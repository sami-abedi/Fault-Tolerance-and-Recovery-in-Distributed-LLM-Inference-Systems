"""
Backend abstraction layer for LLM inference.

Provides a unified interface over:
  - MockBackend   : deterministic CPU-based token simulation (CI/testing)
  - TransformersBackend : HuggingFace transformers (real model, CPU/GPU)
  - VLLMBackend   : vLLM high-throughput serving (GPU only, optional)
"""

from src.backends.base import BaseBackend, InferenceRequest, InferenceResponse
from src.backends.mock_backend import MockBackend

__all__ = [
    "BaseBackend",
    "InferenceRequest",
    "InferenceResponse",
    "MockBackend",
]
