"""
Token checkpoint store for TOKEN_COMMIT_RESUME recovery.

Tier-1 implementation: JSONL-based disk persistence.
Each request gets its own checkpoint file. Tokens are appended
incrementally, enabling partial recovery after mid-generation failures.

Tier-2 stub: KV-cache state placeholder (not yet implemented at model level).

Design:
  - Each checkpoint file: <checkpoint_dir>/<request_id>.jsonl
  - Each line is a JSON object: {"position": int, "token_id": int, "token_text": str}
  - Atomic reads: load the file, parse complete lines only
  - Cleanup: delete file after successful request completion
"""

from __future__ import annotations

import abc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TokenCheckpoint:
    """
    Snapshot of committed tokens for a single request.

    Attributes:
        request_id: The request this checkpoint belongs to.
        tokens: List of (position, token_id) tuples, in order.
        prompt: The original prompt (needed for model-level resume).
        max_new_tokens: Original generation budget.
    """

    request_id: str
    tokens: List[Dict]  # [{"position": int, "token_id": int, "token_text": str}]
    prompt: str = ""
    max_new_tokens: int = 128

    @property
    def token_ids(self) -> List[int]:
        return [t["token_id"] for t in self.tokens]

    @property
    def num_committed(self) -> int:
        return len(self.tokens)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class BaseTokenStore(abc.ABC):
    """Abstract token checkpoint store."""

    @abc.abstractmethod
    async def initialize(self, request_id: str, prompt: str, max_new_tokens: int) -> None:
        """Create a new empty checkpoint for a request."""
        ...

    @abc.abstractmethod
    async def append_token(
        self, request_id: str, position: int, token_id: int, token_text: str
    ) -> None:
        """Append a single committed token to the checkpoint."""
        ...

    @abc.abstractmethod
    async def load(self, request_id: str) -> Optional[TokenCheckpoint]:
        """Load an existing checkpoint for a request, or None if not found."""
        ...

    @abc.abstractmethod
    async def delete(self, request_id: str) -> None:
        """Delete the checkpoint (called on successful completion)."""
        ...

    @abc.abstractmethod
    async def exists(self, request_id: str) -> bool:
        """Return True if a checkpoint exists for request_id."""
        ...


# ---------------------------------------------------------------------------
# Disk-based JSONL store (Tier-1)
# ---------------------------------------------------------------------------


class DiskTokenStore(BaseTokenStore):
    """
    JSONL-based token checkpoint store on local disk.

    Each request checkpoint is stored as a separate JSONL file.
    Lines are appended synchronously (using aiofiles in async context)
    for minimal latency impact.

    File format (one JSON object per line):
        {"type": "header", "request_id": "...", "prompt": "...", "max_new_tokens": 128}
        {"type": "token", "position": 0, "token_id": 42, "token_text": "the"}
        {"type": "token", "position": 1, "token_id": 17, "token_text": "model"}
        ...
    """

    def __init__(self, checkpoint_dir: str) -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, request_id: str) -> Path:
        # Sanitize request_id to be filesystem-safe
        safe_id = request_id.replace("/", "_").replace(":", "_")
        return self._dir / f"{safe_id}.jsonl"

    async def initialize(
        self, request_id: str, prompt: str, max_new_tokens: int
    ) -> None:
        """Create a checkpoint file with the header record."""
        path = self._path(request_id)
        header = {
            "type": "header",
            "request_id": request_id,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        }
        with open(path, "w") as f:
            f.write(json.dumps(header) + "\n")

    async def append_token(
        self,
        request_id: str,
        position: int,
        token_id: int,
        token_text: str,
    ) -> None:
        """Append a token record to the checkpoint file."""
        path = self._path(request_id)
        record = {
            "type": "token",
            "position": position,
            "token_id": token_id,
            "token_text": token_text,
        }
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    async def load(self, request_id: str) -> Optional[TokenCheckpoint]:
        """
        Load and parse the checkpoint file for a given request.

        Handles partial writes gracefully: only complete lines are parsed.
        """
        path = self._path(request_id)
        if not path.exists():
            return None

        tokens: List[Dict] = []
        prompt = ""
        max_new_tokens = 128

        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        # Skip incomplete/corrupt lines (partial write)
                        continue

                    if obj.get("type") == "header":
                        prompt = obj.get("prompt", "")
                        max_new_tokens = obj.get("max_new_tokens", 128)
                    elif obj.get("type") == "token":
                        tokens.append({
                            "position": obj["position"],
                            "token_id": obj["token_id"],
                            "token_text": obj.get("token_text", ""),
                        })
        except OSError:
            return None

        # Sort by position to handle any out-of-order writes
        tokens.sort(key=lambda t: t["position"])

        return TokenCheckpoint(
            request_id=request_id,
            tokens=tokens,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

    async def delete(self, request_id: str) -> None:
        """Remove the checkpoint file."""
        path = self._path(request_id)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    async def exists(self, request_id: str) -> bool:
        return self._path(request_id).exists()


# ---------------------------------------------------------------------------
# In-memory store (for testing / low-latency mode)
# ---------------------------------------------------------------------------


class MemoryTokenStore(BaseTokenStore):
    """
    In-memory token checkpoint store (non-persistent).

    Tokens are stored in a dict keyed by request_id.
    Useful for unit testing and benchmarking the overhead of
    disk I/O by comparison.
    """

    def __init__(self) -> None:
        self._store: Dict[str, TokenCheckpoint] = {}

    async def initialize(
        self, request_id: str, prompt: str, max_new_tokens: int
    ) -> None:
        self._store[request_id] = TokenCheckpoint(
            request_id=request_id,
            tokens=[],
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

    async def append_token(
        self,
        request_id: str,
        position: int,
        token_id: int,
        token_text: str,
    ) -> None:
        if request_id not in self._store:
            return
        self._store[request_id].tokens.append({
            "position": position,
            "token_id": token_id,
            "token_text": token_text,
        })

    async def load(self, request_id: str) -> Optional[TokenCheckpoint]:
        return self._store.get(request_id)

    async def delete(self, request_id: str) -> None:
        self._store.pop(request_id, None)

    async def exists(self, request_id: str) -> bool:
        return request_id in self._store


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_token_store(backend: str, checkpoint_dir: str = "./checkpoints") -> BaseTokenStore:
    """
    Instantiate a token store based on the configured backend.

    Args:
        backend: "disk" or "memory".
        checkpoint_dir: Directory for disk-based store.

    Returns:
        BaseTokenStore instance.
    """
    if backend == "disk":
        return DiskTokenStore(checkpoint_dir)
    elif backend == "memory":
        return MemoryTokenStore()
    else:
        raise ValueError(f"Unknown token store backend: '{backend}'")
