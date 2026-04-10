"""
FAIL_STOP recovery strategy.

The simplest possible strategy: if a fault occurs, the request
immediately fails with no recovery attempt.

Characteristics:
  - Lowest overhead (no retry, no checkpoint I/O)
  - Worst reliability under fault injection
  - Serves as the baseline for comparison

This establishes the lower bound of recovery cost:
  overhead ≈ 0, reliability ≈ (1 - fault_rate)
"""

from __future__ import annotations

import time

from src.recovery.base import (
    BaseRecoveryStrategy,
    GenerationCallable,
    RecoveryContext,
    RecoveryResult,
)
from src.utils.logging_utils import get_logger

logger = get_logger("fail_stop")


class FailStopStrategy(BaseRecoveryStrategy):
    """
    FAIL_STOP: No recovery. Propagate failure immediately.

    On any exception during generation, the request is marked failed
    and control is returned to the caller without any retry.
    """

    name: str = "fail_stop"

    async def execute(
        self,
        context: RecoveryContext,
        generate_fn: GenerationCallable,
    ) -> RecoveryResult:
        """
        Attempt generation exactly once. Fail on any exception.

        Args:
            context: Recovery context with request and worker list.
            generate_fn: Single-attempt generation callable.

        Returns:
            RecoveryResult: success=True on clean run, success=False on fault.
        """
        try:
            response = await generate_fn(context.request)
            logger.info(
                "fail_stop_success",
                request_id=context.request.request_id,
                tokens=response.completion_tokens,
            )
            return RecoveryResult(
                response=response,
                success=True,
                num_retries=0,
                tokens_recomputed=0,
                recovery_time_s=0.0,
            )

        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "fail_stop_failure",
                request_id=context.request.request_id,
                error=err_msg,
            )
            return RecoveryResult(
                response=None,
                success=False,
                num_retries=0,
                tokens_recomputed=0,
                recovery_time_s=0.0,
                error=err_msg,
            )
