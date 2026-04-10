"""
Pydantic request/response models for the FastAPI server.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Inference API
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""

    request_id: Optional[str] = Field(
        default=None,
        description="Optional client-assigned ID. Auto-generated if not provided.",
    )
    prompt: str = Field(
        ...,
        description="The text prompt to complete.",
        min_length=1,
        max_length=32768,
    )
    max_new_tokens: int = Field(
        default=128,
        ge=1,
        le=4096,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0 = greedy decoding.",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability.",
    )
    stream: bool = Field(
        default=False,
        description="If True, use SSE streaming response.",
    )
    recovery_strategy: Optional[str] = Field(
        default=None,
        description="Override default recovery strategy for this request.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata attached to this request.",
    )


class TokenResponse(BaseModel):
    """A single streamed token event."""

    token_id: int
    token_text: str
    position: int
    is_eos: bool = False


class GenerateResponse(BaseModel):
    """Response body for the /generate endpoint (non-streaming)."""

    request_id: str
    generated_text: str
    tokens: List[TokenResponse] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "length"
    latency_s: float = 0.0
    recovery_info: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    request_id: Optional[str] = None
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Admin API
# ---------------------------------------------------------------------------


class FaultInjectRequest(BaseModel):
    """Request body for /admin/inject-fault endpoint."""

    fault_type: str = Field(
        ...,
        description="Fault type: kill | delay | hang | graceful_shutdown",
    )
    worker_id: Optional[str] = Field(
        default=None,
        description="Target worker. If None, pick a random worker.",
    )
    delay_ms: Optional[float] = Field(
        default=None,
        description="Delay in milliseconds (for 'delay' fault type).",
    )
    hang_duration_s: Optional[float] = Field(
        default=None,
        description="Hang duration in seconds (for 'hang' fault type).",
    )


class FaultInjectResponse(BaseModel):
    """Response from fault injection endpoint."""

    status: str
    fault_type: str
    target_worker: Optional[str] = None
    message: str = ""


class WorkerStatus(BaseModel):
    """Status of a single worker."""

    worker_id: str
    host: str
    port: int
    status: str  # healthy | unhealthy | unknown
    requests_served: int = 0
    current_load: int = 0


class SystemStatusResponse(BaseModel):
    """Response from /admin/status endpoint."""

    status: str
    num_workers: int
    workers: List[WorkerStatus]
    backend_type: str
    recovery_strategy: str
    uptime_s: float


class HealthResponse(BaseModel):
    """Response from /health endpoint."""

    status: str = "ok"
    version: str = "0.1.0"
