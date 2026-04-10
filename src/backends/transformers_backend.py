"""
HuggingFace Transformers inference backend.

Real model inference using the ``transformers`` library.
Supports CPU and GPU (CUDA/MPS) inference.

Requires: pip install transformers torch accelerate
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Dict, List, Optional

from src.backends.base import BaseBackend, InferenceRequest, InferenceResponse, Token
from src.utils.logging_utils import get_logger

logger = get_logger("transformers_backend")


class TransformersBackend(BaseBackend):
    """
    Inference backend using HuggingFace Transformers.

    Loads the model and tokenizer at startup. Generation is performed
    synchronously in a thread pool to avoid blocking the event loop.
    """

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self._model_name: str = config.get("model_name", "gpt2")
        self._device: str = config.get("device", "cpu")
        self._max_new_tokens: int = config.get("max_new_tokens", 128)
        self._temperature: float = config.get("temperature", 0.7)
        self._top_p: float = config.get("top_p", 0.9)

        self._model = None
        self._tokenizer = None

    async def load(self) -> None:
        """Load model and tokenizer (runs in thread pool to avoid blocking)."""
        logger.info("loading_transformers_model", model=self._model_name, device=self._device)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)
        self._loaded = True
        logger.info("transformers_model_loaded", model=self._model_name)

    def _load_sync(self) -> None:
        """Synchronous model load (called in executor)."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=torch.float32 if self._device == "cpu" else torch.float16,
            )
            self._model.to(self._device)
            self._model.eval()
        except ImportError as e:
            raise RuntimeError(
                "transformers backend requires: pip install transformers torch"
            ) from e

    async def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        logger.info("transformers_model_unloaded")

    async def health_check(self) -> bool:
        return self._loaded and self._model is not None

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate full sequence via transformers pipeline."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self._generate_sync, request)
        return response

    def _generate_sync(self, request: InferenceRequest) -> InferenceResponse:
        """Synchronous generation (called in thread pool executor)."""
        import torch

        assert self._tokenizer is not None, "Model not loaded"
        assert self._model is not None, "Model not loaded"

        # Handle resume: prepend committed tokens to prompt
        prefix_text = ""
        resume_tokens = request.resume_from_tokens or []
        if resume_tokens:
            prefix_ids = torch.tensor([resume_tokens])
            prefix_text = self._tokenizer.decode(resume_tokens, skip_special_tokens=True)

        full_prompt = request.prompt + (" " + prefix_text if prefix_text else "")
        inputs = self._tokenizer(full_prompt, return_tensors="pt").to(self._device)

        remaining = request.max_new_tokens - len(resume_tokens)
        if remaining <= 0:
            return InferenceResponse(
                request_id=request.request_id,
                tokens=[],
                generated_text=prefix_text,
                prompt_tokens=inputs["input_ids"].shape[1],
                completion_tokens=len(resume_tokens),
                finish_reason="length",
            )

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=remaining,
                temperature=self._temperature if self._temperature > 0 else None,
                top_p=self._top_p,
                do_sample=self._temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        new_token_ids = output[0][prompt_len:].tolist()
        generated_text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True)

        tokens = []
        for pos, tok_id in enumerate(new_token_ids):
            tok_text = self._tokenizer.decode([tok_id], skip_special_tokens=True)
            tokens.append(Token(
                token_id=tok_id,
                token_text=tok_text,
                position=len(resume_tokens) + pos,
                is_eos=(tok_id == self._tokenizer.eos_token_id),
            ))

        return InferenceResponse(
            request_id=request.request_id,
            tokens=tokens,
            generated_text=generated_text,
            prompt_tokens=prompt_len,
            completion_tokens=len(tokens) + len(resume_tokens),
            finish_reason="eos" if (tokens and tokens[-1].is_eos) else "length",
        )

    async def stream_generate(
        self, request: InferenceRequest
    ) -> AsyncIterator[Token]:
        """
        Stream tokens using TextIteratorStreamer.

        Note: True token streaming requires transformers ≥ 4.35.
        """
        try:
            from transformers import TextIteratorStreamer
            import torch
            import threading
        except ImportError:
            raise RuntimeError("transformers backend requires transformers + torch")

        assert self._tokenizer is not None
        assert self._model is not None

        resume_tokens = request.resume_from_tokens or []

        # Yield resumed tokens immediately
        for pos, tok_id in enumerate(resume_tokens):
            tok_text = self._tokenizer.decode([tok_id], skip_special_tokens=True)
            yield Token(token_id=tok_id, token_text=tok_text, position=pos)

        remaining = request.max_new_tokens - len(resume_tokens)
        if remaining <= 0:
            return

        inputs = self._tokenizer(request.prompt, return_tensors="pt").to(self._device)
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generate_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=remaining,
            temperature=self._temperature if self._temperature > 0 else None,
            top_p=self._top_p,
            do_sample=self._temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        loop = asyncio.get_event_loop()
        thread = threading.Thread(
            target=lambda: self._model.generate(**generate_kwargs)
        )
        thread.start()

        pos = len(resume_tokens)
        for text_chunk in streamer:
            # Encode back to get token IDs (approximate)
            chunk_ids = self._tokenizer(text_chunk, add_special_tokens=False)["input_ids"]
            for tok_id in chunk_ids:
                yield Token(
                    token_id=tok_id,
                    token_text=self._tokenizer.decode([tok_id]),
                    position=pos,
                )
                pos += 1
            await asyncio.sleep(0)  # yield control

        thread.join()
