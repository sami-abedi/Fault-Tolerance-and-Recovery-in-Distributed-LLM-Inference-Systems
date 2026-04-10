"""
Model worker: a single inference process serving a backend.

Each worker runs as an independent FastAPI process on its own port.
The router dispatches requests to workers and detects failures via
health checks.

Workers support:
  - /infer         : POST — execute one inference request
  - /stream        : POST — SSE streaming inference
  - /health        : GET  — liveness probe
  - /admin/fault   : POST — inject a fault into this worker (for experiments)
  - /admin/reset   : POST — reset fault state
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
import uuid
from typing import AsyncIterator, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from src.backends.base import InferenceRequest, InferenceResponse, Token
from src.backends.factory import create_backend
from src.config import AppConfig, load_config
from src.server.models import (
    ErrorResponse,
    FaultInjectRequest,
    FaultInjectResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    TokenResponse,
)
from src.utils.logging_utils import configure_logging, get_logger

logger = get_logger("worker")


class FaultState:
    """Mutable fault injection state for the worker process."""

    def __init__(self) -> None:
        self.active: bool = False
        self.fault_type: str = "none"
        self.delay_ms: float = 0.0
        self.hang_duration_s: float = 0.0
        self.trigger_count: int = 0
        self.max_triggers: int = 1  # fault fires once then resets

    def reset(self) -> None:
        self.active = False
        self.fault_type = "none"
        self.delay_ms = 0.0
        self.hang_duration_s = 0.0
        self.trigger_count = 0


def create_worker_app(config: AppConfig, worker_id: str) -> FastAPI:
    """
    Create the FastAPI application for a single worker process.

    Args:
        config: Application configuration.
        worker_id: Unique identifier for this worker.

    Returns:
        Configured FastAPI app instance.
    """
    app = FastAPI(
        title=f"CrashSafe Worker [{worker_id}]",
        version="0.1.0",
    )

    backend = create_backend(config)
    fault_state = FaultState()
    start_time = time.time()
    requests_served = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @app.on_event("startup")
    async def startup():
        await backend.load()
        logger.info(
            "worker_started",
            worker_id=worker_id,
            backend=config.backend.type,
        )

    @app.on_event("shutdown")
    async def shutdown():
        await backend.unload()
        logger.info("worker_shutdown", worker_id=worker_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _apply_fault_if_active() -> None:
        """Raise an exception or sleep based on the active fault state."""
        if not fault_state.active:
            return

        ft = fault_state.fault_type
        fault_state.trigger_count += 1

        if ft == "delay":
            logger.warning(
                "fault_delay_applied",
                worker_id=worker_id,
                delay_ms=fault_state.delay_ms,
            )
            await asyncio.sleep(fault_state.delay_ms / 1000.0)
            # Delay does NOT deactivate the fault (it fires every request)

        elif ft == "hang":
            logger.warning(
                "fault_hang_applied",
                worker_id=worker_id,
                duration_s=fault_state.hang_duration_s,
            )
            await asyncio.sleep(fault_state.hang_duration_s)
            raise TimeoutError(f"Worker {worker_id} simulated hang for {fault_state.hang_duration_s}s")

        elif ft == "kill":
            logger.error(
                "fault_kill_applied",
                worker_id=worker_id,
            )
            fault_state.reset()
            # Simulate process crash — hard exit
            os.kill(os.getpid(), signal.SIGKILL)

        elif ft == "graceful_shutdown":
            logger.warning(
                "fault_graceful_shutdown_applied",
                worker_id=worker_id,
            )
            fault_state.reset()
            os.kill(os.getpid(), signal.SIGTERM)

    # ------------------------------------------------------------------
    # Inference endpoints
    # ------------------------------------------------------------------

    @app.post("/infer", response_model=GenerateResponse)
    async def infer(req: GenerateRequest) -> GenerateResponse:
        """Execute a single inference request (blocking)."""
        nonlocal requests_served

        request_id = req.request_id or f"req-{uuid.uuid4().hex[:8]}"

        # Apply fault if armed
        await _apply_fault_if_active()

        inference_req = InferenceRequest(
            request_id=request_id,
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            metadata=req.metadata,
        )

        t_start = time.perf_counter()

        try:
            response: InferenceResponse = await backend.generate(inference_req)
        except Exception as exc:
            logger.error(
                "worker_inference_error",
                worker_id=worker_id,
                request_id=request_id,
                error=str(exc),
            )
            raise HTTPException(status_code=500, detail=str(exc))

        elapsed = time.perf_counter() - t_start
        requests_served += 1

        token_responses = [
            TokenResponse(
                token_id=t.token_id,
                token_text=t.token_text,
                position=t.position,
                is_eos=t.is_eos,
            )
            for t in response.tokens
        ]

        return GenerateResponse(
            request_id=request_id,
            generated_text=response.generated_text,
            tokens=token_responses,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            finish_reason=response.finish_reason,
            latency_s=elapsed,
        )

    @app.post("/stream")
    async def stream_infer(req: GenerateRequest):
        """Execute a streaming inference request (SSE)."""
        nonlocal requests_served

        request_id = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
        await _apply_fault_if_active()

        inference_req = InferenceRequest(
            request_id=request_id,
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            metadata=req.metadata,
        )

        async def token_stream() -> AsyncIterator[str]:
            import json
            try:
                async for token in backend.stream_generate(inference_req):
                    data = {
                        "token_id": token.token_id,
                        "token_text": token.token_text,
                        "position": token.position,
                        "is_eos": token.is_eos,
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    if token.is_eos:
                        break
            except Exception as exc:
                yield f"data: {{\"error\": \"{str(exc)}\"}}\n\n"

        requests_served += 1
        return StreamingResponse(
            token_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
            },
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        is_healthy = await backend.health_check()
        if not is_healthy:
            raise HTTPException(status_code=503, detail="Backend not ready")
        return HealthResponse(status="ok")

    # ------------------------------------------------------------------
    # Admin / fault injection
    # ------------------------------------------------------------------

    @app.post("/admin/fault", response_model=FaultInjectResponse)
    async def inject_fault(req: FaultInjectRequest) -> FaultInjectResponse:
        """Arm a fault to be triggered on the next inference request."""
        fault_state.active = True
        fault_state.fault_type = req.fault_type
        fault_state.delay_ms = req.delay_ms or 500.0
        fault_state.hang_duration_s = req.hang_duration_s or 30.0
        fault_state.trigger_count = 0

        logger.info(
            "fault_armed",
            worker_id=worker_id,
            fault_type=req.fault_type,
        )

        return FaultInjectResponse(
            status="armed",
            fault_type=req.fault_type,
            target_worker=worker_id,
            message=f"Fault '{req.fault_type}' armed on worker {worker_id}",
        )

    @app.post("/admin/reset")
    async def reset_fault() -> Dict:
        """Clear any armed fault."""
        fault_state.reset()
        return {"status": "reset", "worker_id": worker_id}

    @app.get("/admin/status")
    async def worker_status() -> Dict:
        """Return worker status for the router health check."""
        return {
            "worker_id": worker_id,
            "status": "healthy" if await backend.health_check() else "unhealthy",
            "requests_served": requests_served,
            "uptime_s": time.time() - start_time,
            "fault_active": fault_state.active,
            "fault_type": fault_state.fault_type,
            "backend": config.backend.type,
        }

    return app


def run_worker(
    config: AppConfig,
    worker_id: str,
    port: int,
) -> None:
    """
    Start a worker process on the given port.

    This function is intended to be called in a subprocess.
    """
    configure_logging(level=config.server.log_level)
    app = create_worker_app(config, worker_id)
    uvicorn.run(
        app,
        host=config.worker.worker_host,
        port=port,
        log_level=config.server.log_level,
        access_log=False,
    )


# Entry point for running a worker directly
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CrashSafe Worker Process")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_worker(cfg, args.worker_id, args.port)
