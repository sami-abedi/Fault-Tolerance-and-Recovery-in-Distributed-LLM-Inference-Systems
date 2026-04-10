"""
Abstract base class for all inference backends.

Every backend must implement:
  - generate()         : blocking full-sequence generation
  - stream_generate()  : async streaming token-by-token generation
  - health_check()     : liveness probe
  - load()             : model initialization
  - unload()           : cleanup
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------


@dataclass
class InferenceRequest:
    """
    Represents a single inference request to a backend.

    Attributes:
        request_id: Unique identifier for this request (set by router).
        prompt: The input text prompt.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_p: Nucleus sampling probability.
        resume_from_tokens: Optional list of already-committed tokens
            (for TOKEN_COMMIT_RESUME recovery). Backend should skip
            recomputing these and resume from this prefix.
        metadata: Arbitrary key-value metadata for tracing.
    """

    request_id: str
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    resume_from_tokens: Optional[List[int]] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class Token:
    """A single generated token with its metadata."""

    token_id: int
    token_text: str
    log_prob: Optional[float] = None
    is_eos: bool = False
    position: int = 0  # position index in the generated sequence


@dataclass
class InferenceResponse:
    """
    Full response from a backend after generation completes.

    Attributes:
        request_id: Echoes the request id.
        tokens: List of generated Token objects.
        generated_text: Decoded full string.
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens generated.
        finish_reason: Why generation stopped (length / eos / error).
        error: Error message if generation failed.
    """

    request_id: str
    tokens: List[Token] = field(default_factory=list)
    generated_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "length"  # length | eos | error
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class BaseBackend(abc.ABC):
    """
    Abstract inference backend.

    Subclasses must implement ``generate`` and ``stream_generate``.
    The ``load`` method is called once at startup; ``unload`` at shutdown.
    """

    def __init__(self, config: Dict) -> None:
        self._config = config
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """Initialize the backend (load model weights, allocate resources)."""
        self._loaded = True

    async def unload(self) -> None:
        """Release backend resources."""
        self._loaded = False

    async def health_check(self) -> bool:
        """Return True if the backend is ready to serve requests."""
        return self._loaded

    # ------------------------------------------------------------------
    # Inference interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """
        Generate a complete response (blocking until EOS or max_new_tokens).

        Args:
            request: The inference request.

        Returns:
            Completed InferenceResponse.
        """
        ...

    @abc.abstractmethod
    async def stream_generate(
        self, request: InferenceRequest
    ) -> AsyncIterator[Token]:
        """
        Stream tokens one by one as they are generated.

        Args:
            request: The inference request.

        Yields:
            Token objects in generation order.
        """
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(loaded={self._loaded})"
