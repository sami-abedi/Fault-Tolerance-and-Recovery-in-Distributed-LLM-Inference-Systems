"""
Request router: dispatches inference requests to worker processes.

Implements:
  - Round-robin, least-loaded, and random worker selection
  - Health checking (periodic background task)
  - Automatic worker removal on repeated failures
  - Integration with recovery strategies

The router is the central coordinator of the distributed system.
It receives requests from the main FastAPI app and routes them to
the appropriate worker, applying the configured recovery strategy.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import httpx

from src.backends.base import InferenceRequest, InferenceResponse, Token
from src.config import AppConfig
from src.recovery.base import (
    BaseRecoveryStrategy,
    GenerationCallable,
    RecoveryContext,
    RecoveryResult,
)
from src.server.models import GenerateRequest, GenerateResponse
from src.utils.logging_utils import get_logger
from src.utils.metrics import RequestMetrics

logger = get_logger("router")


# ---------------------------------------------------------------------------
# Worker registry entry
# ---------------------------------------------------------------------------


@dataclass
class WorkerInfo:
    """Metadata and state for a registered worker."""

    worker_id: str
    host: str
    port: int
    status: str = "unknown"  # healthy | unhealthy | unknown
    consecutive_failures: int = 0
    requests_served: int = 0
    current_load: int = 0
    last_health_check: float = field(default_factory=time.time)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def is_healthy(self) -> bool:
        return self.status == "healthy"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class InferenceRouter:
    """
    Load-balancing router for distributed LLM inference.

    Manages a pool of worker processes, performs health checks,
    and routes inference requests according to the configured strategy
    and recovery policy.
    """

    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(
        self,
        config: AppConfig,
        recovery_strategy: BaseRecoveryStrategy,
    ) -> None:
        self._config = config
        self._recovery = recovery_strategy
        self._workers: Dict[str, WorkerInfo] = {}
        self._round_robin_idx: int = 0
        self._client: Optional[httpx.AsyncClient] = None
        self._health_check_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize HTTP client and start background health checks."""
        self._client = httpx.AsyncClient(
            timeout=self._config.router.request_timeout_s,
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info(
            "router_started",
            strategy=self._config.router.strategy,
            recovery=self._recovery.name,
        )

    async def stop(self) -> None:
        """Cancel health checks and close HTTP client."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()
        logger.info("router_stopped")

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    def register_worker(self, worker_id: str, host: str, port: int) -> None:
        """Add a worker to the routing pool."""
        self._workers[worker_id] = WorkerInfo(
            worker_id=worker_id,
            host=host,
            port=port,
            status="unknown",
        )
        logger.info("worker_registered", worker_id=worker_id, host=host, port=port)

    def deregister_worker(self, worker_id: str) -> None:
        """Remove a worker from the routing pool."""
        self._workers.pop(worker_id, None)
        logger.info("worker_deregistered", worker_id=worker_id)

    def _select_worker(self) -> Optional[WorkerInfo]:
        """Select a worker according to the routing strategy."""
        healthy = [w for w in self._workers.values() if w.is_healthy]
        if not healthy:
            return None

        strategy = self._config.router.strategy

        if strategy == "round_robin":
            idx = self._round_robin_idx % len(healthy)
            self._round_robin_idx += 1
            return healthy[idx]

        elif strategy == "least_loaded":
            return min(healthy, key=lambda w: w.current_load)

        elif strategy == "random":
            return random.choice(healthy)

        else:
            return healthy[0]

    def _get_all_healthy_ids(self) -> List[str]:
        return [w.worker_id for w in self._workers.values() if w.is_healthy]

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    async def _health_check_loop(self) -> None:
        """Periodically probe all workers for liveness."""
        interval = self._config.router.health_check_interval_s
        while True:
            try:
                await asyncio.sleep(interval)
                await self._check_all_workers()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("health_check_error", error=str(exc))

    async def _check_all_workers(self) -> None:
        """Run health checks on all registered workers concurrently."""
        tasks = [self._check_worker(w) for w in list(self._workers.values())]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_worker(self, worker: WorkerInfo) -> None:
        """Probe a single worker's /health endpoint."""
        try:
            assert self._client is not None
            resp = await self._client.get(
                f"{worker.base_url}/health", timeout=5.0
            )
            if resp.status_code == 200:
                worker.status = "healthy"
                worker.consecutive_failures = 0
            else:
                self._mark_worker_failure(worker)
        except Exception:
            self._mark_worker_failure(worker)

        worker.last_health_check = time.time()

    def _mark_worker_failure(self, worker: WorkerInfo) -> None:
        worker.consecutive_failures += 1
        if worker.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            if worker.status != "unhealthy":
                logger.warning(
                    "worker_marked_unhealthy",
                    worker_id=worker.worker_id,
                    failures=worker.consecutive_failures,
                )
            worker.status = "unhealthy"
        else:
            worker.status = "unknown"

    async def check_workers_once(self) -> None:
        """Force an immediate health check of all workers."""
        await self._check_all_workers()

    # ------------------------------------------------------------------
    # Request routing
    # ------------------------------------------------------------------

    async def route(
        self,
        request: GenerateRequest,
        metrics: Optional[RequestMetrics] = None,
    ) -> RecoveryResult:
        """
        Route an inference request through the recovery strategy.

        Args:
            request: The incoming GenerateRequest.
            metrics: Optional RequestMetrics object to update in-place.

        Returns:
            RecoveryResult from the recovery strategy.
        """
        request_id = request.request_id or f"req-{uuid.uuid4().hex[:8]}"

        inference_req = InferenceRequest(
            request_id=request_id,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            metadata=request.metadata,
        )

        context = RecoveryContext(
            request=inference_req,
            worker_ids=self._get_all_healthy_ids(),
        )

        async def generate_fn(req: InferenceRequest) -> InferenceResponse:
            return await self._dispatch_to_worker(req, metrics)

        result = await self._recovery.execute(context, generate_fn)

        if metrics:
            metrics.timestamp_end = time.time()
            metrics.success = result.success
            metrics.num_retries = result.num_retries
            metrics.tokens_recomputed = result.tokens_recomputed
            if result.response:
                metrics.total_tokens_generated = result.response.completion_tokens
            if result.recovery_time_s > 0:
                metrics.timestamp_recovery_start = (
                    metrics.timestamp_start + result.recovery_time_s * 0.1
                )  # approximate
                metrics.timestamp_recovery_end = (
                    metrics.timestamp_start + result.recovery_time_s
                )

        return result

    async def _dispatch_to_worker(
        self,
        request: InferenceRequest,
        metrics: Optional[RequestMetrics] = None,
    ) -> InferenceResponse:
        """
        Send an inference request to a selected worker.

        Raises an exception on communication failure so the recovery
        strategy can handle it.
        """
        worker = self._select_worker()
        if worker is None:
            raise RuntimeError("No healthy workers available")

        worker.current_load += 1
        try:
            assert self._client is not None
            t_start = time.perf_counter()

            payload = {
                "request_id": request.request_id,
                "prompt": request.prompt,
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "metadata": request.metadata,
            }

            resp = await self._client.post(
                f"{worker.base_url}/infer",
                json=payload,
                timeout=self._config.router.request_timeout_s,
            )

            elapsed = time.perf_counter() - t_start

            if resp.status_code != 200:
                worker.consecutive_failures += 1
                raise RuntimeError(
                    f"Worker {worker.worker_id} returned HTTP {resp.status_code}: {resp.text}"
                )

            data = resp.json()

            # Record TTFT (approximate: first byte of response)
            if metrics and metrics.timestamp_first_token is None:
                metrics.timestamp_first_token = time.time() - (elapsed * 0.1)

            worker.consecutive_failures = 0
            worker.requests_served += 1

            tokens = [
                Token(
                    token_id=t["token_id"],
                    token_text=t["token_text"],
                    position=t["position"],
                    is_eos=t.get("is_eos", False),
                )
                for t in data.get("tokens", [])
            ]

            return InferenceResponse(
                request_id=data["request_id"],
                tokens=tokens,
                generated_text=data["generated_text"],
                prompt_tokens=data.get("prompt_tokens", 0),
                completion_tokens=data.get("completion_tokens", 0),
                finish_reason=data.get("finish_reason", "length"),
            )

        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            logger.warning(
                "worker_communication_error",
                worker_id=worker.worker_id,
                error=str(exc),
            )
            self._mark_worker_failure(worker)
            raise RuntimeError(f"Worker {worker.worker_id} communication failed: {exc}") from exc

        finally:
            worker.current_load -= 1

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def workers(self) -> Dict[str, WorkerInfo]:
        return dict(self._workers)

    @property
    def healthy_worker_count(self) -> int:
        return sum(1 for w in self._workers.values() if w.is_healthy)

    def get_worker_statuses(self) -> List[Dict]:
        return [
            {
                "worker_id": w.worker_id,
                "host": w.host,
                "port": w.port,
                "status": w.status,
                "requests_served": w.requests_served,
                "current_load": w.current_load,
            }
            for w in self._workers.values()
        ]
