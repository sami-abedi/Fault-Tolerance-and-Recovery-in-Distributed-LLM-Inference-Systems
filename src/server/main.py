"""
CrashSafe main FastAPI application (router process).

This is the entry point for the distributed inference system.
It starts the router and worker sub-processes, then serves
the public API and admin endpoints.

Endpoints:
  POST /generate         — Submit inference request (sync or streaming)
  GET  /health           — Liveness probe
  GET  /admin/status     — System-wide status
  POST /admin/inject-fault — Arm a fault on a target worker
  POST /admin/workers/{id}/reset — Reset worker fault state
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.config import AppConfig, load_config
from src.recovery.factory import create_recovery_strategy
from src.server.models import (
    ErrorResponse,
    FaultInjectRequest,
    FaultInjectResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    SystemStatusResponse,
    WorkerStatus,
)
from src.server.router import InferenceRouter
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.metrics import MetricsSink, RequestMetrics

logger = get_logger("main")


# ---------------------------------------------------------------------------
# Worker subprocess launcher
# ---------------------------------------------------------------------------


def _launch_worker_process(
    config: AppConfig,
    worker_id: str,
    port: int,
) -> multiprocessing.Process:
    """
    Start a worker as an independent OS process.

    Args:
        config: Application config passed to the worker.
        worker_id: Unique worker identifier.
        port: Port the worker should listen on.

    Returns:
        Started Process object.
    """
    from src.server.worker import run_worker

    proc = multiprocessing.Process(
        target=run_worker,
        args=(config, worker_id, port),
        daemon=True,
        name=f"crashsafe-worker-{worker_id}",
    )
    proc.start()
    logger.info(
        "worker_process_launched",
        worker_id=worker_id,
        port=port,
        pid=proc.pid,
    )
    return proc


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    """
    Create and configure the main FastAPI application.

    Args:
        config: Optional pre-built config (for testing). If None,
                loads from the default config file.

    Returns:
        Configured FastAPI app instance.
    """
    if config is None:
        config = load_config()

    configure_logging(level=config.server.log_level)

    app = FastAPI(
        title="CrashSafe: Distributed LLM Inference with Fault Tolerance",
        version="0.1.0",
        description=(
            "Research prototype for fault tolerance and recovery "
            "in distributed LLM inference systems."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # State
    worker_processes: List[multiprocessing.Process] = []
    recovery_strategy = create_recovery_strategy(config)
    router = InferenceRouter(config, recovery_strategy)

    # Ensure output directories exist
    Path(config.storage.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.storage.results_dir).mkdir(parents=True, exist_ok=True)
    Path(config.storage.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    metrics_sink = MetricsSink(config.metrics.log_file)
    start_time = time.time()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @app.on_event("startup")
    async def startup() -> None:
        """Launch worker processes and start the router."""
        # Launch worker subprocesses
        for i in range(config.worker.num_workers):
            worker_id = f"worker-{i}"
            port = config.worker.worker_base_port + i
            proc = _launch_worker_process(config, worker_id, port)
            worker_processes.append(proc)
            router.register_worker(
                worker_id=worker_id,
                host=config.worker.worker_host,
                port=port,
            )

        # Start router (HTTP client + health check loop)
        await router.start()

        # Wait for workers to become healthy
        logger.info(
            "waiting_for_workers",
            num_workers=config.worker.num_workers,
            timeout_s=config.worker.startup_timeout_s,
        )
        deadline = time.time() + config.worker.startup_timeout_s
        while time.time() < deadline:
            await router.check_workers_once()
            if router.healthy_worker_count >= 1:
                break
            await asyncio.sleep(1.0)

        if router.healthy_worker_count == 0:
            logger.warning(
                "no_healthy_workers_at_startup",
                note="System may not serve requests until workers are ready",
            )
        else:
            logger.info(
                "system_ready",
                healthy_workers=router.healthy_worker_count,
                recovery_strategy=recovery_strategy.name,
            )

    @app.on_event("shutdown")
    async def shutdown() -> None:
        """Stop router and terminate worker processes."""
        await router.stop()
        for proc in worker_processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
        logger.info("system_shutdown_complete")

    # ------------------------------------------------------------------
    # Inference endpoints
    # ------------------------------------------------------------------

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest) -> GenerateResponse:
        """
        Submit an inference request.

        For streaming responses, set ``stream=True`` in the request body.
        Streaming uses Server-Sent Events (SSE).
        """
        request_id = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
        req.request_id = request_id

        metrics = RequestMetrics(
            request_id=request_id,
            fault_type=config.fault_injection.default_fault_type,
            recovery_strategy=recovery_strategy.name,
            prompt_length=len(req.prompt.split()),
        )

        result = await router.route(req, metrics=metrics)

        metrics_sink.record(metrics)

        if not result.success:
            raise HTTPException(
                status_code=503,
                detail=result.error or "Inference failed after all recovery attempts",
            )

        assert result.response is not None
        response = result.response

        token_responses = [
            {
                "token_id": t.token_id,
                "token_text": t.token_text,
                "position": t.position,
                "is_eos": t.is_eos,
            }
            for t in response.tokens
        ]

        return GenerateResponse(
            request_id=request_id,
            generated_text=response.generated_text,
            tokens=token_responses,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            finish_reason=response.finish_reason,
            latency_s=metrics.latency_s or 0.0,
            recovery_info={
                "num_retries": result.num_retries,
                "tokens_recomputed": result.tokens_recomputed,
                "recovery_time_s": result.recovery_time_s,
            },
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Liveness probe."""
        return HealthResponse(status="ok")

    # ------------------------------------------------------------------
    # Admin endpoints
    # ------------------------------------------------------------------

    @app.get("/admin/status", response_model=SystemStatusResponse)
    async def system_status() -> SystemStatusResponse:
        """Return system-wide status including all worker states."""
        worker_statuses = [
            WorkerStatus(**w) for w in router.get_worker_statuses()
        ]
        return SystemStatusResponse(
            status="ok" if router.healthy_worker_count > 0 else "degraded",
            num_workers=len(router.workers),
            workers=worker_statuses,
            backend_type=config.backend.type,
            recovery_strategy=recovery_strategy.name,
            uptime_s=time.time() - start_time,
        )

    @app.post("/admin/inject-fault", response_model=FaultInjectResponse)
    async def inject_fault(req: FaultInjectRequest) -> FaultInjectResponse:
        """
        Arm a fault on one or all workers.

        The fault will fire on the next inference request received
        by the target worker.
        """
        import httpx as _httpx

        target_workers = []
        if req.worker_id:
            worker = router.workers.get(req.worker_id)
            if not worker:
                raise HTTPException(
                    status_code=404,
                    detail=f"Worker '{req.worker_id}' not found",
                )
            target_workers = [worker]
        else:
            # Target first healthy worker
            healthy = [w for w in router.workers.values() if w.is_healthy]
            if not healthy:
                raise HTTPException(status_code=503, detail="No healthy workers")
            target_workers = [healthy[0]]

        results = []
        async with _httpx.AsyncClient(timeout=5.0) as client:
            for worker in target_workers:
                try:
                    resp = await client.post(
                        f"{worker.base_url}/admin/fault",
                        json=req.model_dump(),
                    )
                    results.append(resp.json())
                except Exception as exc:
                    logger.warning(
                        "fault_inject_failed",
                        worker_id=worker.worker_id,
                        error=str(exc),
                    )

        return FaultInjectResponse(
            status="armed",
            fault_type=req.fault_type,
            target_worker=target_workers[0].worker_id if target_workers else None,
            message=f"Fault '{req.fault_type}' armed on {len(target_workers)} worker(s)",
        )

    @app.post("/admin/workers/{worker_id}/reset")
    async def reset_worker_fault(worker_id: str) -> Dict:
        """Clear armed fault state on a specific worker."""
        import httpx as _httpx

        worker = router.workers.get(worker_id)
        if not worker:
            raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found")

        async with _httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{worker.base_url}/admin/reset")

        return resp.json()

    @app.get("/admin/metrics/export")
    async def export_metrics() -> Dict:
        """Export all collected metrics as JSON."""
        return {
            "records": [r.to_dict() for r in metrics_sink.records],
            "count": len(metrics_sink.records),
        }

    return app


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the CrashSafe server as a standalone process."""
    import argparse

    parser = argparse.ArgumentParser(description="CrashSafe Distributed Inference Server")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    if args.log_level:
        config.server.log_level = args.log_level

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
