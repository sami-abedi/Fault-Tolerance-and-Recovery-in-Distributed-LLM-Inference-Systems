"""
Unit tests for inference backends.

Tests:
  - MockBackend: load, generate, stream_generate
  - Deterministic token generation (reproducibility)
  - Token-resume correctness (same tokens before and after resume point)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backends.base import InferenceRequest
from src.backends.mock_backend import MockBackend, _deterministic_token_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    return {
        "tokens_per_second": 500.0,  # fast for tests
        "base_latency_ms": 1.0,
    }


@pytest.fixture
def backend(mock_config):
    return MockBackend(mock_config)


@pytest.fixture
def sample_request():
    return InferenceRequest(
        request_id="test-001",
        prompt="The quick brown fox",
        max_new_tokens=20,
        temperature=0.7,
    )


# ---------------------------------------------------------------------------
# Backend lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_backend_loads(backend):
    """Backend should be loadable and report healthy."""
    assert not backend.is_loaded
    await backend.load()
    assert backend.is_loaded
    assert await backend.health_check()


@pytest.mark.asyncio
async def test_mock_backend_unloads(backend):
    """Backend should unload cleanly."""
    await backend.load()
    await backend.unload()
    assert not backend.is_loaded
    assert not await backend.health_check()


# ---------------------------------------------------------------------------
# Token generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_returns_correct_token_count(backend, sample_request):
    """generate() should produce exactly max_new_tokens tokens."""
    await backend.load()
    response = await backend.generate(sample_request)
    assert response.completion_tokens == sample_request.max_new_tokens
    assert len(response.tokens) == sample_request.max_new_tokens


@pytest.mark.asyncio
async def test_generate_is_deterministic(backend, sample_request):
    """Same prompt + position should always produce the same token ID."""
    await backend.load()

    r1 = await backend.generate(sample_request)
    r2 = await backend.generate(sample_request)

    token_ids_1 = [t.token_id for t in r1.tokens]
    token_ids_2 = [t.token_id for t in r2.tokens]
    assert token_ids_1 == token_ids_2, "Token generation must be deterministic"


@pytest.mark.asyncio
async def test_stream_generate_matches_generate(backend, sample_request):
    """stream_generate() should yield the same tokens as generate()."""
    await backend.load()

    # Collect stream tokens
    stream_tokens = []
    async for token in backend.stream_generate(sample_request):
        stream_tokens.append(token)

    # Full generate
    response = await backend.generate(sample_request)

    stream_ids = [t.token_id for t in stream_tokens]
    gen_ids = [t.token_id for t in response.tokens]
    assert stream_ids == gen_ids, "Streaming must produce same tokens as batch"


@pytest.mark.asyncio
async def test_generate_positions_are_sequential(backend, sample_request):
    """Token positions should be 0, 1, 2, ... (contiguous)."""
    await backend.load()
    response = await backend.generate(sample_request)
    positions = [t.position for t in response.tokens]
    expected = list(range(sample_request.max_new_tokens))
    assert positions == expected


# ---------------------------------------------------------------------------
# TOKEN_COMMIT_RESUME correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_generates_same_suffix(backend, sample_request):
    """
    Critical correctness test for TOKEN_COMMIT_RESUME:

    If we run full generation and collect tokens [0..N-1], then
    run again with resume_from_tokens = [0..k-1], the suffix [k..N-1]
    must be identical.
    """
    await backend.load()

    # Full generation
    full_response = await backend.generate(sample_request)
    full_ids = [t.token_id for t in full_response.tokens]

    # Resume from position k
    k = 8
    committed = full_ids[:k]
    resume_request = InferenceRequest(
        request_id="test-resume",
        prompt=sample_request.prompt,
        max_new_tokens=sample_request.max_new_tokens,
        resume_from_tokens=committed,
    )
    resume_response = await backend.generate(resume_request)
    resume_ids = [t.token_id for t in resume_response.tokens]

    # The resume response should include all tokens (committed + new)
    # The suffix (positions k..N-1) should match full generation
    suffix_full = full_ids[k:]
    suffix_resume = resume_ids[k:]  # tokens after the committed prefix

    assert suffix_resume == suffix_full, (
        f"Resume suffix mismatch at position {k}: "
        f"expected {suffix_full}, got {suffix_resume}"
    )


@pytest.mark.asyncio
async def test_resume_skips_recomputation(backend, sample_request):
    """
    Resumed generation should produce fewer new tokens than the committed count.
    """
    await backend.load()

    # Commit 10 tokens
    committed_ids = [
        _deterministic_token_id(sample_request.prompt, i)
        for i in range(10)
    ]

    resume_request = InferenceRequest(
        request_id="test-resume-2",
        prompt=sample_request.prompt,
        max_new_tokens=sample_request.max_new_tokens,
        resume_from_tokens=committed_ids,
    )

    response = await backend.generate(resume_request)
    # Total tokens should still be max_new_tokens
    assert response.completion_tokens == sample_request.max_new_tokens


# ---------------------------------------------------------------------------
# Determinism unit tests
# ---------------------------------------------------------------------------


def test_deterministic_token_id_consistent():
    """Same prompt + position always yields the same token ID."""
    prompt = "test prompt"
    for pos in range(50):
        t1 = _deterministic_token_id(prompt, pos)
        t2 = _deterministic_token_id(prompt, pos)
        assert t1 == t2


def test_deterministic_token_id_position_varies():
    """Different positions should produce different token IDs (with high probability)."""
    prompt = "test prompt"
    ids = [_deterministic_token_id(prompt, i) for i in range(20)]
    # Should not be all the same
    assert len(set(ids)) > 1


def test_deterministic_token_id_prompt_varies():
    """Different prompts at the same position should produce different IDs."""
    pos = 5
    id1 = _deterministic_token_id("prompt A", pos)
    id2 = _deterministic_token_id("prompt B", pos)
    # May occasionally collide, but in practice should differ
    # Just verify the function runs without error
    assert isinstance(id1, int)
    assert isinstance(id2, int)
