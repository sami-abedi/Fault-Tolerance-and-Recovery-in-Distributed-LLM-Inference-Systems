"""
Abstract base class for recovery strategies.

A recovery strategy wraps inference execution with fault handling logic.
It receives a GenerationCallable and is responsible for:
  1. Executing the callable
  2. Detecting failures (exceptions / timeouts)
  3. Deciding whether and how to retry / resume
  4. Reporting metrics (recovery time, tokens recomputed, etc.)
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, List, Optional

from src.backends.base import InferenceRequest, InferenceResponse, Token
from src.utils.logging_utils import get_logger

logger = get_logger("recovery")


# ---------------------------------------------------------------------------
# Context and result types
# ---------------------------------------------------------------------------


@dataclass
class RecoveryContext:
    """
    Shared context passed to a recovery strategy.

    Attributes:
        request: The original inference request.
        worker_ids: Ordered list of available worker IDs to try.
        attempt: Current attempt number (0-indexed).
        committed_tokens: Tokens committed so far (for TOKEN_COMMIT_RESUME).
        recovery_start_ts: Timestamp when recovery began.
    """

    request: InferenceRequest
    worker_ids: List[str]
    attempt: int = 0
    committed_tokens: List[Token] = field(default_factory=list)
    recovery_start_ts: Optional[float] = None

    def mark_recovery_start(self) -> None:
        self.recovery_start_ts = time.time()

    def recovery_elapsed_s(self) -> Optional[float]:
        if self.recovery_start_ts is not None:
            return time.time() - self.recovery_start_ts
        return None


@dataclass
class RecoveryResult:
    """
    Outcome of a recovery strategy execution.

    Attributes:
        response: The final InferenceResponse (None on permanent failure).
        success: Whether the request ultimately succeeded.
        num_retries: Number of retry attempts performed.
        tokens_recomputed: How many tokens had to be regenerated.
        recovery_time_s: Total time spent in recovery (s).
        error: Error message if permanently failed.
    """

    response: Optional[InferenceResponse]
    success: bool
    num_retries: int = 0
    tokens_recomputed: int = 0
    recovery_time_s: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Callable types
# ---------------------------------------------------------------------------

# Type alias for the async callable that performs one generation attempt
# Signature: (request: InferenceRequest) -> InferenceResponse
GenerationCallable = Callable[
    [InferenceRequest], Coroutine[Any, Any, InferenceResponse]
]


# ---------------------------------------------------------------------------
# Abstract strategy
# ---------------------------------------------------------------------------


class BaseRecoveryStrategy(abc.ABC):
    """
    Abstract recovery strategy.

    Subclasses implement ``execute`` which wraps a generation callable
    with the appropriate fault handling logic.
    """

    name: str = "base"

    @abc.abstractmethod
    async def execute(
        self,
        context: RecoveryContext,
        generate_fn: GenerationCallable,
    ) -> RecoveryResult:
        """
        Execute a generation attempt with recovery handling.

        Args:
            context: Shared recovery context (request, workers, etc.).
            generate_fn: Async callable that performs one generation attempt.

        Returns:
            RecoveryResult with outcome and metrics.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
