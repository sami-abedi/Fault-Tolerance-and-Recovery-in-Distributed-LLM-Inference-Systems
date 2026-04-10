"""
RETRY_FROM_SCRATCH recovery strategy.

On failure, re-submits the entire generation request to a (possibly
different) worker with exponential backoff.

Characteristics:
  - Higher latency than fail_stop (retries add overhead)
  - Wastes all previously computed tokens on each failure
  - tokens_recomputed = max_new_tokens per failed attempt
  - Reliability scales with number of retries

This establishes the middle ground:
  overhead ≈ k * generation_time, reliability ≈ 1 - (1-p)^(k+1)
  where p = P(single attempt succeeds), k = num retries
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional

from src.recovery.base import (
    BaseRecoveryStrategy,
    GenerationCallable,
    RecoveryContext,
    RecoveryResult,
)
from src.utils.logging_utils import get_logger

logger = get_logger("retry_from_scratch")


class RetryFromScratchStrategy(BaseRecoveryStrategy):
    """
    RETRY_FROM_SCRATCH: Restart full generation on failure.

    Retries up to ``max_retries`` times with exponential backoff.
    Each retry starts a completely fresh generation from the beginning,
    discarding any previously generated tokens.
    """

    name: str = "retry_from_scratch"

    def __init__(
        self,
        max_retries: int = 3,
        backoff_base_s: float = 0.5,
        backoff_max_s: float = 10.0,
    ) -> None:
        """
        Args:
            max_retries: Maximum number of retry attempts after initial failure.
            backoff_base_s: Base sleep duration for exponential backoff.
            backoff_max_s: Maximum sleep duration cap.
        """
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s

    async def execute(
        self,
        context: RecoveryContext,
        generate_fn: GenerationCallable,
    ) -> RecoveryResult:
        """
        Attempt generation up to max_retries + 1 times.

        Each attempt uses the original (unmodified) request — no
        token checkpointing occurs. On failure, exponential backoff
        is applied before the next attempt.

        Returns:
            RecoveryResult with success, retry count, and recomputed tokens.
        """
        max_new_tokens = context.request.max_new_tokens
        total_recomputed = 0
        recovery_start: Optional[float] = None
        last_error: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await generate_fn(context.request)
                recovery_time = (
                    time.time() - recovery_start if recovery_start else 0.0
                )
                logger.info(
                    "retry_success",
                    request_id=context.request.request_id,
                    attempt=attempt,
                    total_recomputed=total_recomputed,
                )
                return RecoveryResult(
                    response=response,
                    success=True,
                    num_retries=attempt,
                    tokens_recomputed=total_recomputed,
                    recovery_time_s=recovery_time,
                )

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "retry_attempt_failed",
                    request_id=context.request.request_id,
                    attempt=attempt,
                    error=last_error,
                )

                if attempt == 0:
                    recovery_start = time.time()
                else:
                    # Each failed attempt wasted max_new_tokens tokens
                    total_recomputed += max_new_tokens

                if attempt < self.max_retries:
                    # Exponential backoff with jitter
                    sleep_s = min(
                        self.backoff_base_s * (2 ** attempt),
                        self.backoff_max_s,
                    )
                    logger.debug(
                        "retry_backoff",
                        request_id=context.request.request_id,
                        sleep_s=sleep_s,
                        next_attempt=attempt + 1,
                    )
                    await asyncio.sleep(sleep_s)

        # All retries exhausted — the first failure also counts as wasted work
        total_recomputed += max_new_tokens

        recovery_time = time.time() - recovery_start if recovery_start else 0.0
        logger.error(
            "retry_exhausted",
            request_id=context.request.request_id,
            max_retries=self.max_retries,
            total_recomputed=total_recomputed,
        )
        return RecoveryResult(
            response=None,
            success=False,
            num_retries=self.max_retries,
            tokens_recomputed=total_recomputed,
            recovery_time_s=recovery_time,
            error=last_error,
        )
