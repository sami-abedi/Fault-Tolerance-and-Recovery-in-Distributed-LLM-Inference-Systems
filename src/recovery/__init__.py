"""
Recovery strategy implementations.

Three strategies are provided:

FAIL_STOP
    No recovery. The request fails immediately on any fault.
    Baseline: lowest overhead, worst reliability.

RETRY_FROM_SCRATCH
    Re-submits the entire request to a healthy worker on failure.
    Simple exponential backoff. Wastes all previously computed tokens.

TOKEN_COMMIT_RESUME (KEY CONTRIBUTION)
    Persists generated tokens incrementally to a durable store.
    On failure, loads the checkpoint and resumes generation from
    the last committed token, avoiding redundant computation.
"""

from src.recovery.base import BaseRecoveryStrategy, RecoveryContext, RecoveryResult
from src.recovery.fail_stop import FailStopStrategy
from src.recovery.retry import RetryFromScratchStrategy
from src.recovery.token_resume import TokenCommitResumeStrategy

__all__ = [
    "BaseRecoveryStrategy",
    "RecoveryContext",
    "RecoveryResult",
    "FailStopStrategy",
    "RetryFromScratchStrategy",
    "TokenCommitResumeStrategy",
]
