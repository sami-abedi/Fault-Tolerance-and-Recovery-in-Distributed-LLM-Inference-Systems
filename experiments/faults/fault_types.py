"""
Fault injection definitions and orchestration.

Defines the five fault types in the fault model:
  1. NONE          — No fault (baseline)
  2. KILL          — Process crash (SIGKILL via /admin/fault)
  3. GRACEFUL_SHUTDOWN — Clean shutdown (SIGTERM)
  4. DELAY         — Artificial latency injection
  5. HANG          — Deadlock / unresponsive simulation

Each fault is injected via the router's /admin/inject-fault endpoint.
The experiment runner arms a fault before sending requests, then
measures the impact and recovery.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import httpx


class FaultType(str, Enum):
    """Enumeration of supported fault types."""
    NONE = "none"
    KILL = "kill"
    GRACEFUL_SHUTDOWN = "graceful_shutdown"
    DELAY = "delay"
    HANG = "hang"


@dataclass
class FaultSpec:
    """
    Specification for a fault injection.

    Attributes:
        fault_type: The type of fault to inject.
        target_worker_id: Specific worker to target (None = auto-select).
        delay_ms: Delay duration in milliseconds (for DELAY fault).
        hang_duration_s: Hang duration in seconds (for HANG fault).
    """
    fault_type: FaultType
    target_worker_id: Optional[str] = None
    delay_ms: float = 500.0
    hang_duration_s: float = 5.0

    def to_request_body(self) -> Dict:
        return {
            "fault_type": self.fault_type.value,
            "worker_id": self.target_worker_id,
            "delay_ms": self.delay_ms,
            "hang_duration_s": self.hang_duration_s,
        }


class FaultInjector:
    """
    Client for arming faults via the router's admin API.

    Used by the experiment runner to inject faults before sending
    inference requests and measure their impact.
    """

    def __init__(self, router_url: str) -> None:
        """
        Args:
            router_url: Base URL of the router (e.g., http://localhost:8000).
        """
        self._router_url = router_url.rstrip("/")

    async def inject(self, spec: FaultSpec) -> Dict:
        """
        Arm the specified fault on the target worker.

        Args:
            spec: Fault specification.

        Returns:
            JSON response from the admin endpoint.
        """
        if spec.fault_type == FaultType.NONE:
            return {"status": "no_fault", "message": "No fault injected"}

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self._router_url}/admin/inject-fault",
                json=spec.to_request_body(),
            )
            resp.raise_for_status()
            return resp.json()

    async def reset_worker(self, worker_id: str) -> Dict:
        """Clear fault state on a specific worker."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self._router_url}/admin/workers/{worker_id}/reset"
            )
            resp.raise_for_status()
            return resp.json()

    async def system_status(self) -> Dict:
        """Fetch current system status from the router."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self._router_url}/admin/status")
            resp.raise_for_status()
            return resp.json()

    async def wait_for_healthy_workers(
        self, min_workers: int = 1, timeout_s: float = 60.0, poll_interval_s: float = 2.0
    ) -> bool:
        """
        Poll until at least min_workers are healthy or timeout expires.

        Returns:
            True if condition met, False on timeout.
        """
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            try:
                status = await self.system_status()
                healthy = sum(
                    1 for w in status.get("workers", []) if w.get("status") == "healthy"
                )
                if healthy >= min_workers:
                    return True
            except Exception:
                pass
            await asyncio.sleep(poll_interval_s)
        return False


# ---------------------------------------------------------------------------
# Fault parameter grids for experiments
# ---------------------------------------------------------------------------

FAULT_GRID = [
    FaultSpec(fault_type=FaultType.NONE),
    FaultSpec(fault_type=FaultType.DELAY, delay_ms=500.0),
    FaultSpec(fault_type=FaultType.DELAY, delay_ms=2000.0),
    FaultSpec(fault_type=FaultType.HANG, hang_duration_s=5.0),
    FaultSpec(fault_type=FaultType.KILL),
    FaultSpec(fault_type=FaultType.GRACEFUL_SHUTDOWN),
]

FAULT_LABELS = {
    FaultType.NONE: "No Fault",
    FaultType.DELAY: "Latency Injection",
    FaultType.HANG: "Worker Hang",
    FaultType.KILL: "Process Kill (SIGKILL)",
    FaultType.GRACEFUL_SHUTDOWN: "Graceful Shutdown",
}
