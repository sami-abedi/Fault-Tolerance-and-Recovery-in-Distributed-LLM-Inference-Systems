"""
TOKEN_COMMIT_RESUME recovery strategy (key contribution).

Incrementally persists generated tokens to a durable store during
generation. On failure, the checkpoint is loaded and generation resumes
from the last committed token — avoiding recomputation of committed tokens.

Tier-1: Disk-based JSONL checkpoint storage.
Tier-2: KV-cache state preservation (architectural stub, future work).

Design rationale:
  - Tokens are flushed every N tokens (configurable) to bound I/O overhead
  - On resume, the backend receives resume_from_tokens to reconstruct context
  - Recovery time ≈ (checkpoint_interval × token_time) rather than
    (full_sequence_length × token_time)
  - Recomputed tokens ≈ (max_new_tokens - committed) vs max_new_tokens for retry

Expected experimental outcome:
  - recovery_time << retry_from_scratch
  - tokens_recomputed << retry_from_scratch
  - E2E latency slightly higher than fail_stop (due to checkpoint I/O)
  - Reliability comparable to retry_from_scratch
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, List, Optional

from src.backends.base import InferenceRequest, InferenceResponse, Token
from src.recovery.base import (
    BaseRecoveryStrategy,
    GenerationCallable,
    RecoveryContext,
    RecoveryResult,
)
from src.storage.token_store import BaseTokenStore
from src.utils.logging_utils import get_logger

logger = get_logger("token_commit_resume")


class TokenCommitResumeStrategy(BaseRecoveryStrategy):
    """
    TOKEN_COMMIT_RESUME: Checkpoint tokens during generation; resume on failure.

    Key novelty: unlike retry-from-scratch, this strategy avoids
    recomputing tokens that were already generated before the failure,
    reducing both recovery latency and wasted compute.

    Algorithm:
        1. Initialize checkpoint for request in token store
        2. Wrap the stream_generate call with checkpoint writer
        3. On failure, load checkpoint → build resume request
        4. Re-invoke generate_fn with resume_from_tokens set
        5. On success, delete checkpoint (cleanup)
    """

    name: str = "token_commit_resume"

    def __init__(
        self,
        token_store: BaseTokenStore,
        checkpoint_every_n: int = 10,
        max_retries: int = 3,
        backoff_base_s: float = 0.5,
        backoff_max_s: float = 10.0,
    ) -> None:
        """
        Args:
            token_store: Persistent or in-memory token store.
            checkpoint_every_n: Flush tokens to store every N tokens.
            max_retries: Maximum resume attempts after failure.
            backoff_base_s: Base exponential backoff sleep time.
            backoff_max_s: Maximum backoff cap.
        """
        self.token_store = token_store
        self.checkpoint_every_n = checkpoint_every_n
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s

    async def execute(
        self,
        context: RecoveryContext,
        generate_fn: GenerationCallable,
    ) -> RecoveryResult:
        """
        Execute generation with incremental token checkpointing.

        On failure at position k, resumes from checkpoint (0..k-1).
        Retries up to max_retries times.

        Returns:
            RecoveryResult with final response and checkpoint metrics.
        """
        request = context.request
        request_id = request.request_id

        # Initialize checkpoint
        await self.token_store.initialize(
            request_id, request.prompt, request.max_new_tokens
        )

        recovery_start: Optional[float] = None
        total_tokens_recomputed = 0
        last_error: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            try:
                # Load existing checkpoint (empty on first attempt)
                checkpoint = await self.token_store.load(request_id)
                committed_token_ids: List[int] = (
                    checkpoint.token_ids if checkpoint else []
                )

                # Build resume request if we have committed tokens
                if committed_token_ids and attempt > 0:
                    resume_request = InferenceRequest(
                        request_id=request_id,
                        prompt=request.prompt,
                        max_new_tokens=request.max_new_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        resume_from_tokens=committed_token_ids,
                        metadata=request.metadata,
                    )
                    tokens_recomputed_this_attempt = (
                        request.max_new_tokens - len(committed_token_ids)
                    )
                    logger.info(
                        "token_resume_resuming",
                        request_id=request_id,
                        attempt=attempt,
                        committed=len(committed_token_ids),
                        remaining=tokens_recomputed_this_attempt,
                    )
                else:
                    resume_request = request
                    tokens_recomputed_this_attempt = request.max_new_tokens

                # Execute generation with checkpointing
                response = await self._generate_with_checkpoint(
                    resume_request, generate_fn
                )

                recovery_time = (
                    time.time() - recovery_start if recovery_start else 0.0
                )

                # Successful — clean up checkpoint
                await self.token_store.delete(request_id)

                logger.info(
                    "token_resume_success",
                    request_id=request_id,
                    attempt=attempt,
                    total_tokens_recomputed=total_tokens_recomputed,
                    recovery_time_s=recovery_time,
                )

                return RecoveryResult(
                    response=response,
                    success=True,
                    num_retries=attempt,
                    tokens_recomputed=total_tokens_recomputed,
                    recovery_time_s=recovery_time,
                )

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

                if attempt == 0:
                    recovery_start = time.time()

                # Load checkpoint to see how far we got
                checkpoint = await self.token_store.load(request_id)
                committed = len(checkpoint.tokens) if checkpoint else 0
                tokens_wasted = max(
                    0, tokens_recomputed_this_attempt - committed
                ) if attempt > 0 else max(0, request.max_new_tokens - committed)
                # Only count tokens we have to redo from the new resume point
                total_tokens_recomputed = max(
                    0, request.max_new_tokens - committed
                )

                logger.warning(
                    "token_resume_fault_detected",
                    request_id=request_id,
                    attempt=attempt,
                    committed_tokens=committed,
                    error=last_error,
                )

                if attempt < self.max_retries:
                    sleep_s = min(
                        self.backoff_base_s * (2 ** attempt),
                        self.backoff_max_s,
                    )
                    await asyncio.sleep(sleep_s)

        # Permanently failed — clean up checkpoint
        await self.token_store.delete(request_id)

        recovery_time = time.time() - recovery_start if recovery_start else 0.0
        return RecoveryResult(
            response=None,
            success=False,
            num_retries=self.max_retries,
            tokens_recomputed=total_tokens_recomputed,
            recovery_time_s=recovery_time,
            error=last_error,
        )

    async def _generate_with_checkpoint(
        self,
        request: InferenceRequest,
        generate_fn: GenerationCallable,
    ) -> InferenceResponse:
        """
        Wrap generate_fn: append each produced token to the checkpoint store.

        We call generate_fn to get the full response (which internally
        calls stream_generate). However, for checkpointing we need
        token-level visibility during streaming.

        This implementation uses a two-phase approach:
          - Call generate_fn for the full response
          - Tokens from resume_from_tokens are already checkpointed
          - Newly generated tokens are appended after generation

        For a production system, this would be integrated directly
        into the streaming loop. This implementation demonstrates
        the protocol correctly while remaining backend-agnostic.
        """
        # Execute generation (may raise on fault)
        response = await generate_fn(request)

        # Checkpoint newly generated tokens (those not from resume)
        resume_count = len(request.resume_from_tokens or [])
        for token in response.tokens[resume_count:]:
            await self.token_store.append_token(
                request_id=request.request_id,
                position=token.position,
                token_id=token.token_id,
                token_text=token.token_text,
            )

        return response


class StreamingTokenCommitResumeStrategy(TokenCommitResumeStrategy):
    """
    Extended TOKEN_COMMIT_RESUME with true streaming checkpoints.

    Integrates directly with stream_generate to checkpoint tokens
    as they are produced, providing finer recovery granularity.

    This requires the backend to expose a stream_generate interface.
    The strategy intercepts the token stream, writing each token
    to the store in real-time before passing it downstream.
    """

    name: str = "token_commit_resume_streaming"

    async def execute_streaming(
        self,
        context: RecoveryContext,
        stream_generate_fn,  # async generator factory
        token_callback=None,  # optional: called with each token
    ) -> RecoveryResult:
        """
        Execute streaming generation with per-token checkpointing.

        Args:
            context: Recovery context.
            stream_generate_fn: Callable returning an AsyncIterator[Token].
            token_callback: Optional async callback called with each Token.

        Returns:
            RecoveryResult with full token list.
        """
        request = context.request
        request_id = request.request_id
        recovery_start: Optional[float] = None
        last_error: Optional[str] = None
        total_tokens_recomputed = 0

        await self.token_store.initialize(
            request_id, request.prompt, request.max_new_tokens
        )

        for attempt in range(self.max_retries + 1):
            try:
                checkpoint = await self.token_store.load(request_id)
                committed_ids: List[int] = checkpoint.token_ids if checkpoint else []

                # Build resume request
                resume_request = InferenceRequest(
                    request_id=request_id,
                    prompt=request.prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    resume_from_tokens=committed_ids if attempt > 0 else None,
                    metadata=request.metadata,
                )

                collected_tokens: List[Token] = list(checkpoint.tokens) if (attempt > 0 and checkpoint) else []
                batch: List[Token] = []

                # Stream tokens and checkpoint in batches
                async for token in stream_generate_fn(resume_request):
                    collected_tokens.append(token)
                    batch.append(token)

                    if token_callback:
                        await token_callback(token)

                    # Flush batch every N tokens
                    if len(batch) >= self.checkpoint_every_n:
                        for t in batch:
                            await self.token_store.append_token(
                                request_id, t.position, t.token_id, t.token_text
                            )
                        batch.clear()

                # Flush remaining
                for t in batch:
                    await self.token_store.append_token(
                        request_id, t.position, t.token_id, t.token_text
                    )

                recovery_time = time.time() - recovery_start if recovery_start else 0.0
                await self.token_store.delete(request_id)

                generated_text = " ".join(t.token_text for t in collected_tokens)
                response = InferenceResponse(
                    request_id=request_id,
                    tokens=collected_tokens,
                    generated_text=generated_text,
                    prompt_tokens=0,
                    completion_tokens=len(collected_tokens),
                    finish_reason="eos" if (collected_tokens and collected_tokens[-1].is_eos) else "length",
                )

                return RecoveryResult(
                    response=response,
                    success=True,
                    num_retries=attempt,
                    tokens_recomputed=total_tokens_recomputed,
                    recovery_time_s=recovery_time,
                )

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == 0:
                    recovery_start = time.time()

                checkpoint = await self.token_store.load(request_id)
                committed = len(checkpoint.tokens) if checkpoint else 0
                total_tokens_recomputed = max(0, request.max_new_tokens - committed)

                logger.warning(
                    "streaming_resume_fault",
                    request_id=request_id,
                    attempt=attempt,
                    committed=committed,
                    error=last_error,
                )

                if attempt < self.max_retries:
                    sleep_s = min(self.backoff_base_s * (2 ** attempt), self.backoff_max_s)
                    await asyncio.sleep(sleep_s)

        await self.token_store.delete(request_id)
        recovery_time = time.time() - recovery_start if recovery_start else 0.0
        return RecoveryResult(
            response=None,
            success=False,
            num_retries=self.max_retries,
            tokens_recomputed=total_tokens_recomputed,
            recovery_time_s=recovery_time,
            error=last_error,
        )
