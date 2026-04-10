"""
Mock inference backend for CPU-only / CI environments.

Simulates LLM token generation with:
  - Configurable tokens-per-second rate
  - Deterministic token IDs (hash-based, reproducible)
  - Faithful TOKEN_COMMIT_RESUME support (resumes from committed prefix)
  - Optional artificial delays for fault injection testing

This backend generates no real language but produces a consistent,
measurable token stream that exercises all system paths correctly.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import AsyncIterator, Dict, List, Optional

from src.backends.base import BaseBackend, InferenceRequest, InferenceResponse, Token
from src.utils.logging_utils import get_logger

logger = get_logger("mock_backend")

# Shared mock vocabulary (common English words for readable output)
_MOCK_VOCAB: List[str] = [
    "the", "a", "is", "was", "in", "of", "and", "to", "for", "with",
    "that", "this", "it", "be", "on", "are", "at", "by", "from", "as",
    "model", "token", "inference", "latency", "fault", "recovery",
    "system", "distributed", "request", "response", "checkpoint",
    "worker", "router", "stream", "generate", "compute", "cache",
    "failure", "retry", "strategy", "overhead", "throughput", "delay",
    "process", "node", "replica", "batch", "queue", "timeout", "error",
    "success", "resume", "commit", "persist", "disk", "memory", "gpu",
]
_VOCAB_SIZE = len(_MOCK_VOCAB)


def _deterministic_token_id(prompt: str, position: int) -> int:
    """
    Generate a deterministic token ID based on prompt and position.

    This ensures reproducibility across runs and enables checkpointing
    to be validated: if we resume from position k, the tokens at k+1..N
    are identical to a fresh run.
    """
    h = hashlib.md5(f"{prompt}:{position}".encode()).digest()
    return int.from_bytes(h[:4], "little") % _VOCAB_SIZE


def _token_text(token_id: int) -> str:
    return _MOCK_VOCAB[token_id % _VOCAB_SIZE]


class MockBackend(BaseBackend):
    """
    Deterministic mock inference backend.

    Generates tokens at a configurable rate without any real model.
    Supports all recovery strategies including TOKEN_COMMIT_RESUME.
    """

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self._tokens_per_second: float = config.get("tokens_per_second", 50.0)
        self._base_latency_ms: float = config.get("base_latency_ms", 10.0)
        # per-token sleep in seconds
        self._token_interval_s: float = 1.0 / max(self._tokens_per_second, 0.1)

    async def load(self) -> None:
        """Mock load — instantaneous."""
        logger.info(
            "mock_backend_loaded",
            tokens_per_second=self._tokens_per_second,
            base_latency_ms=self._base_latency_ms,
        )
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False
        logger.info("mock_backend_unloaded")

    async def health_check(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """
        Generate a full sequence and return the complete response.

        If ``request.resume_from_tokens`` is set, those tokens are
        treated as already generated and only the remaining tokens
        are computed (simulating KV-cache resume).
        """
        tokens: List[Token] = []
        async for token in self.stream_generate(request):
            tokens.append(token)

        generated_text = " ".join(t.token_text for t in tokens)
        return InferenceResponse(
            request_id=request.request_id,
            tokens=tokens,
            generated_text=generated_text,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(tokens),
            finish_reason="eos" if (tokens and tokens[-1].is_eos) else "length",
        )

    async def stream_generate(
        self, request: InferenceRequest
    ) -> AsyncIterator[Token]:
        """
        Stream tokens one by one with simulated inter-token delay.

        Handles TOKEN_COMMIT_RESUME: if ``resume_from_tokens`` contains
        previously committed token IDs, we skip regenerating them and
        start from the resume position.
        """
        # Base latency (TTFT simulation)
        await asyncio.sleep(self._base_latency_ms / 1000.0)

        resume_tokens = request.resume_from_tokens or []
        start_position = len(resume_tokens)

        # Yield resumed tokens immediately (they came from checkpoint)
        for pos, tok_id in enumerate(resume_tokens):
            yield Token(
                token_id=tok_id,
                token_text=_token_text(tok_id),
                position=pos,
                is_eos=False,
            )

        # Generate remaining tokens
        for pos in range(start_position, request.max_new_tokens):
            token_id = _deterministic_token_id(request.prompt, pos)
            is_eos = pos == request.max_new_tokens - 1

            yield Token(
                token_id=token_id,
                token_text=_token_text(token_id),
                position=pos,
                is_eos=is_eos,
            )

            if not is_eos:
                await asyncio.sleep(self._token_interval_s)

    def __repr__(self) -> str:
        return (
            f"MockBackend("
            f"tps={self._tokens_per_second}, "
            f"base_latency_ms={self._base_latency_ms})"
        )
