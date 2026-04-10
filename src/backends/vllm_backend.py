"""
vLLM inference backend (GPU only, optional).

High-throughput serving via PagedAttention and continuous batching.
Requires: pip install vllm (CUDA GPU required)

This backend is a stub that integrates with the vLLM AsyncLLMEngine.
"""

from __future__ import annotations

from typing import AsyncIterator, Dict, List, Optional

from src.backends.base import BaseBackend, InferenceRequest, InferenceResponse, Token
from src.utils.logging_utils import get_logger

logger = get_logger("vllm_backend")


class VLLMBackend(BaseBackend):
    """
    vLLM-based inference backend for GPU-accelerated serving.

    Wraps vLLM's AsyncLLMEngine for async token streaming.
    TOKEN_COMMIT_RESUME is supported via prompt prefix injection:
    committed tokens are decoded and prepended to the prompt so vLLM
    naturally continues from that point (no true KV-cache resume).
    """

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self._model_name: str = config.get("model_name", "facebook/opt-125m")
        self._max_new_tokens: int = config.get("max_new_tokens", 128)
        self._temperature: float = config.get("temperature", 0.7)
        self._top_p: float = config.get("top_p", 0.9)
        self._engine = None
        self._tokenizer = None

    async def load(self) -> None:
        """Initialize vLLM AsyncLLMEngine."""
        logger.info("loading_vllm_backend", model=self._model_name)
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs
            from vllm.transformers_utils.tokenizer import get_tokenizer

            engine_args = AsyncEngineArgs(
                model=self._model_name,
                dtype="auto",
                max_model_len=2048,
            )
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._tokenizer = get_tokenizer(self._model_name)
            self._loaded = True
            logger.info("vllm_backend_loaded", model=self._model_name)
        except ImportError as exc:
            raise RuntimeError(
                "vLLM backend requires: pip install vllm (GPU required)"
            ) from exc

    async def unload(self) -> None:
        self._engine = None
        self._tokenizer = None
        self._loaded = False

    async def health_check(self) -> bool:
        return self._loaded and self._engine is not None

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        tokens: List[Token] = []
        async for token in self.stream_generate(request):
            tokens.append(token)
        generated_text = "".join(t.token_text for t in tokens)
        return InferenceResponse(
            request_id=request.request_id,
            tokens=tokens,
            generated_text=generated_text,
            prompt_tokens=0,  # filled in by engine
            completion_tokens=len(tokens),
            finish_reason="eos" if (tokens and tokens[-1].is_eos) else "length",
        )

    async def stream_generate(
        self, request: InferenceRequest
    ) -> AsyncIterator[Token]:
        """Stream tokens via vLLM AsyncLLMEngine."""
        from vllm import SamplingParams

        assert self._engine is not None, "vLLM engine not loaded"
        assert self._tokenizer is not None

        resume_tokens = request.resume_from_tokens or []

        # Yield committed tokens from checkpoint
        for pos, tok_id in enumerate(resume_tokens):
            tok_text = self._tokenizer.decode([tok_id])
            yield Token(token_id=tok_id, token_text=tok_text, position=pos)

        # Decode prefix and append to prompt
        prefix_text = (
            self._tokenizer.decode(resume_tokens, skip_special_tokens=True)
            if resume_tokens
            else ""
        )
        full_prompt = request.prompt + (" " + prefix_text if prefix_text else "")

        remaining = request.max_new_tokens - len(resume_tokens)
        if remaining <= 0:
            return

        sampling_params = SamplingParams(
            max_tokens=remaining,
            temperature=self._temperature,
            top_p=self._top_p,
        )

        pos = len(resume_tokens)
        async for output in self._engine.generate(
            full_prompt, sampling_params, request_id=request.request_id
        ):
            if output.outputs:
                out = output.outputs[0]
                for tok in out.token_ids[pos - len(resume_tokens):]:
                    tok_text = self._tokenizer.decode([tok])
                    yield Token(
                        token_id=tok,
                        token_text=tok_text,
                        position=pos,
                        is_eos=(out.finish_reason == "stop"),
                    )
                    pos += 1
