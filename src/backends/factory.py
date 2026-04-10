"""
Backend factory: instantiate the configured backend type.
"""

from __future__ import annotations

from typing import Dict

from src.backends.base import BaseBackend
from src.config import AppConfig


def create_backend(config: AppConfig) -> BaseBackend:
    """
    Instantiate and return the appropriate inference backend.

    Args:
        config: Global application configuration.

    Returns:
        Unloaded BaseBackend instance (caller must call ``await backend.load()``).

    Raises:
        ValueError: If ``config.backend.type`` is not recognized.
    """
    backend_type = config.backend.type

    if backend_type == "mock":
        from src.backends.mock_backend import MockBackend

        backend_cfg: Dict = {
            "tokens_per_second": config.mock_backend.tokens_per_second,
            "base_latency_ms": config.mock_backend.base_latency_ms,
            "token_vocab_size": config.mock_backend.token_vocab_size,
        }
        return MockBackend(backend_cfg)

    elif backend_type == "transformers":
        from src.backends.transformers_backend import TransformersBackend

        backend_cfg = {
            "model_name": config.backend.model_name,
            "device": config.backend.device,
            "max_new_tokens": config.backend.max_new_tokens,
            "temperature": config.backend.temperature,
            "top_p": config.backend.top_p,
        }
        return TransformersBackend(backend_cfg)

    elif backend_type == "vllm":
        from src.backends.vllm_backend import VLLMBackend

        backend_cfg = {
            "model_name": config.backend.model_name,
            "max_new_tokens": config.backend.max_new_tokens,
            "temperature": config.backend.temperature,
            "top_p": config.backend.top_p,
        }
        return VLLMBackend(backend_cfg)

    else:
        raise ValueError(
            f"Unknown backend type '{backend_type}'. "
            "Valid options: mock, transformers, vllm"
        )
