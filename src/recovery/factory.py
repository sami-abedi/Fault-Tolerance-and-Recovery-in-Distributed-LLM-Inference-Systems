"""
Recovery strategy factory.
"""

from __future__ import annotations

from src.config import AppConfig, RecoveryConfig
from src.recovery.base import BaseRecoveryStrategy
from src.recovery.fail_stop import FailStopStrategy
from src.recovery.retry import RetryFromScratchStrategy
from src.recovery.token_resume import TokenCommitResumeStrategy
from src.storage.token_store import BaseTokenStore, create_token_store


def create_recovery_strategy(
    config: AppConfig,
    token_store: BaseTokenStore | None = None,
) -> BaseRecoveryStrategy:
    """
    Instantiate the configured recovery strategy.

    Args:
        config: Global application config.
        token_store: Optional pre-built token store (used for testing).

    Returns:
        BaseRecoveryStrategy instance.
    """
    rc: RecoveryConfig = config.recovery
    strategy = rc.strategy

    if strategy == "fail_stop":
        return FailStopStrategy()

    elif strategy == "retry_from_scratch":
        return RetryFromScratchStrategy(
            max_retries=rc.max_retries,
            backoff_base_s=rc.retry_backoff_base_s,
            backoff_max_s=rc.retry_backoff_max_s,
        )

    elif strategy == "token_commit_resume":
        if token_store is None:
            token_store = create_token_store(
                backend=config.storage.backend,
                checkpoint_dir=config.storage.checkpoint_dir,
            )
        return TokenCommitResumeStrategy(
            token_store=token_store,
            checkpoint_every_n=rc.checkpoint_every_n_tokens,
            max_retries=rc.max_retries,
            backoff_base_s=rc.retry_backoff_base_s,
            backoff_max_s=rc.retry_backoff_max_s,
        )

    else:
        raise ValueError(
            f"Unknown recovery strategy '{strategy}'. "
            "Valid options: fail_stop, retry_from_scratch, token_commit_resume"
        )
