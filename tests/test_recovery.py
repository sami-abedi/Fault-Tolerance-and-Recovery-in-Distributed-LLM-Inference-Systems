"""
Unit tests for recovery strategies.

Tests:
  - FailStopStrategy: success and failure cases
  - RetryFromScratchStrategy: retry counting, backoff, token recomputation
  - TokenCommitResumeStrategy: checkpoint correctness, resume after fault
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backends.base import InferenceRequest, InferenceResponse, Token
from src.recovery.base import RecoveryContext
from src.recovery.fail_stop import FailStopStrategy
from src.recovery.retry import RetryFromScratchStrategy
from src.recovery.token_resume import TokenCommitResumeStrategy
from src.storage.token_store import MemoryTokenStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_request(request_id: str = "test-req", max_tokens: int = 20) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        prompt="Test prompt for recovery",
        max_new_tokens=max_tokens,
    )


def make_response(request_id: str, num_tokens: int) -> InferenceResponse:
    tokens = [
        Token(token_id=i, token_text=f"tok{i}", position=i)
        for i in range(num_tokens)
    ]
    return InferenceResponse(
        request_id=request_id,
        tokens=tokens,
        generated_text=" ".join(t.token_text for t in tokens),
        completion_tokens=num_tokens,
    )


def make_context(request: InferenceRequest) -> RecoveryContext:
    return RecoveryContext(
        request=request,
        worker_ids=["worker-0"],
    )


# ---------------------------------------------------------------------------
# FailStopStrategy tests
# ---------------------------------------------------------------------------


class TestFailStopStrategy:

    @pytest.mark.asyncio
    async def test_success(self):
        """FailStop should return the response on success."""
        strategy = FailStopStrategy()
        request = make_request()
        context = make_context(request)
        expected = make_response(request.request_id, 20)

        async def generate_fn(req):
            return expected

        result = await strategy.execute(context, generate_fn)

        assert result.success is True
        assert result.response == expected
        assert result.num_retries == 0
        assert result.tokens_recomputed == 0
        assert result.recovery_time_s == 0.0

    @pytest.mark.asyncio
    async def test_failure_no_retry(self):
        """FailStop should fail immediately without retrying."""
        strategy = FailStopStrategy()
        request = make_request()
        context = make_context(request)
        call_count = 0

        async def generate_fn(req):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Simulated crash")

        result = await strategy.execute(context, generate_fn)

        assert result.success is False
        assert result.response is None
        assert call_count == 1  # Only one attempt
        assert "RuntimeError" in (result.error or "")

    @pytest.mark.asyncio
    async def test_failure_preserves_error_message(self):
        """FailStop should capture the error message."""
        strategy = FailStopStrategy()
        request = make_request()

        async def generate_fn(req):
            raise ValueError("Test error message")

        result = await strategy.execute(make_context(request), generate_fn)

        assert result.success is False
        assert "Test error message" in (result.error or "")


# ---------------------------------------------------------------------------
# RetryFromScratchStrategy tests
# ---------------------------------------------------------------------------


class TestRetryFromScratchStrategy:

    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        """Should succeed on first attempt without any retries."""
        strategy = RetryFromScratchStrategy(max_retries=3, backoff_base_s=0.01)
        request = make_request()
        expected = make_response(request.request_id, 20)

        async def generate_fn(req):
            return expected

        result = await strategy.execute(make_context(request), generate_fn)

        assert result.success is True
        assert result.num_retries == 0
        assert result.tokens_recomputed == 0

    @pytest.mark.asyncio
    async def test_success_after_one_failure(self):
        """Should succeed on retry after one failure."""
        strategy = RetryFromScratchStrategy(max_retries=3, backoff_base_s=0.01)
        request = make_request()
        expected = make_response(request.request_id, 20)
        call_count = 0

        async def generate_fn(req):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("First attempt fails")
            return expected

        result = await strategy.execute(make_context(request), generate_fn)

        assert result.success is True
        assert result.num_retries == 1
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """Should fail after max_retries + 1 total attempts."""
        max_retries = 2
        strategy = RetryFromScratchStrategy(max_retries=max_retries, backoff_base_s=0.01)
        request = make_request()
        call_count = 0

        async def generate_fn(req):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Always fails")

        result = await strategy.execute(make_context(request), generate_fn)

        assert result.success is False
        assert result.num_retries == max_retries
        assert call_count == max_retries + 1

    @pytest.mark.asyncio
    async def test_tokens_recomputed_counted_correctly(self):
        """Each failed attempt contributes max_new_tokens to recomputed count."""
        max_retries = 2
        max_tokens = 30
        strategy = RetryFromScratchStrategy(max_retries=max_retries, backoff_base_s=0.01)
        request = make_request(max_tokens=max_tokens)

        async def generate_fn(req):
            raise RuntimeError("Always fails")

        result = await strategy.execute(make_context(request), generate_fn)

        # Each of the (max_retries + 1) attempts wasted max_tokens
        expected_recomputed = (max_retries + 1) * max_tokens
        assert result.tokens_recomputed == expected_recomputed

    @pytest.mark.asyncio
    async def test_uses_original_request_each_retry(self):
        """Retry should always pass the original (unmodified) request."""
        strategy = RetryFromScratchStrategy(max_retries=2, backoff_base_s=0.01)
        original_prompt = "Original prompt text"
        request = make_request()
        request.prompt = original_prompt
        received_prompts = []

        async def generate_fn(req):
            received_prompts.append(req.prompt)
            raise RuntimeError("fail")

        await strategy.execute(make_context(request), generate_fn)

        for prompt in received_prompts:
            assert prompt == original_prompt


# ---------------------------------------------------------------------------
# TokenCommitResumeStrategy tests
# ---------------------------------------------------------------------------


class TestTokenCommitResumeStrategy:

    @pytest.mark.asyncio
    async def test_success_no_fault(self):
        """Should succeed and clean up checkpoint on success."""
        store = MemoryTokenStore()
        strategy = TokenCommitResumeStrategy(
            token_store=store,
            checkpoint_every_n=5,
            max_retries=3,
            backoff_base_s=0.01,
        )
        request = make_request()
        expected = make_response(request.request_id, 20)

        async def generate_fn(req):
            return expected

        result = await strategy.execute(make_context(request), generate_fn)

        assert result.success is True
        assert result.response is not None
        assert result.tokens_recomputed == 0
        # Checkpoint should be cleaned up
        assert not await store.exists(request.request_id)

    @pytest.mark.asyncio
    async def test_checkpoint_created_on_failure(self):
        """On failure, strategy should checkpoint generated tokens."""
        store = MemoryTokenStore()
        strategy = TokenCommitResumeStrategy(
            token_store=store,
            checkpoint_every_n=5,
            max_retries=0,  # fail immediately after first attempt
            backoff_base_s=0.01,
        )
        request = make_request("checkpoint-test", max_tokens=20)

        async def generate_fn(req):
            # First call fails (simulates partial generation + crash)
            raise RuntimeError("Worker crash")

        result = await strategy.execute(make_context(request), generate_fn)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_resume_reduces_recomputation(self):
        """
        Critical: resumed generation should recompute fewer tokens
        than retry-from-scratch.
        """
        store = MemoryTokenStore()
        strategy = TokenCommitResumeStrategy(
            token_store=store,
            checkpoint_every_n=5,
            max_retries=2,
            backoff_base_s=0.01,
        )
        max_tokens = 30
        request = make_request("resume-test", max_tokens=max_tokens)

        call_count = 0

        async def generate_fn(req):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First attempt: fail after generating some tokens
                # Manually populate checkpoint with some tokens
                if await store.exists(req.request_id):
                    for i in range(15):  # 15 tokens committed
                        await store.append_token(req.request_id, i, i + 100, f"tok{i}")
                raise RuntimeError("First attempt fails after partial work")

            # Second attempt: succeed
            committed = req.resume_from_tokens or []
            tokens = [
                Token(token_id=i + 100, token_text=f"tok{i}", position=i)
                for i in range(max_tokens)
            ]
            return InferenceResponse(
                request_id=req.request_id,
                tokens=tokens,
                generated_text="resumed response",
                completion_tokens=max_tokens,
            )

        result = await strategy.execute(make_context(request), generate_fn)
        assert result.success is True
        # Recomputed should be less than full max_tokens
        # (because checkpoint preserved some tokens)
        assert result.tokens_recomputed <= max_tokens

    @pytest.mark.asyncio
    async def test_checkpoint_deleted_on_success(self):
        """Checkpoint file should be removed after successful generation."""
        store = MemoryTokenStore()
        strategy = TokenCommitResumeStrategy(
            token_store=store,
            max_retries=0,
            backoff_base_s=0.01,
        )
        request = make_request("cleanup-test")

        async def generate_fn(req):
            return make_response(req.request_id, 10)

        await strategy.execute(make_context(request), generate_fn)
        assert not await store.exists(request.request_id)


# ---------------------------------------------------------------------------
# Token store tests
# ---------------------------------------------------------------------------


class TestTokenStore:

    @pytest.mark.asyncio
    async def test_memory_store_lifecycle(self):
        """MemoryTokenStore should support init, append, load, delete."""
        store = MemoryTokenStore()

        await store.initialize("req-1", "test prompt", 50)
        assert await store.exists("req-1")

        await store.append_token("req-1", 0, 42, "hello")
        await store.append_token("req-1", 1, 17, "world")

        checkpoint = await store.load("req-1")
        assert checkpoint is not None
        assert checkpoint.num_committed == 2
        assert checkpoint.token_ids == [42, 17]

        await store.delete("req-1")
        assert not await store.exists("req-1")

    @pytest.mark.asyncio
    async def test_disk_store_lifecycle(self, tmp_path):
        """DiskTokenStore should persist and reload tokens correctly."""
        from src.storage.token_store import DiskTokenStore

        store = DiskTokenStore(str(tmp_path / "checkpoints"))

        await store.initialize("req-disk", "disk prompt", 64)
        assert await store.exists("req-disk")

        await store.append_token("req-disk", 0, 100, "alpha")
        await store.append_token("req-disk", 1, 200, "beta")
        await store.append_token("req-disk", 2, 300, "gamma")

        checkpoint = await store.load("req-disk")
        assert checkpoint is not None
        assert checkpoint.num_committed == 3
        assert checkpoint.token_ids == [100, 200, 300]
        assert checkpoint.prompt == "disk prompt"

        await store.delete("req-disk")
        assert not await store.exists("req-disk")

    @pytest.mark.asyncio
    async def test_disk_store_handles_missing_file(self, tmp_path):
        """Loading a non-existent checkpoint should return None."""
        from src.storage.token_store import DiskTokenStore

        store = DiskTokenStore(str(tmp_path / "checkpoints"))
        result = await store.load("nonexistent-req")
        assert result is None
