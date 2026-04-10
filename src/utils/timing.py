"""
Timing helpers and async sleep utilities.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Optional


@contextmanager
def timed(label: str = "") -> Generator[dict, None, None]:
    """
    Synchronous context manager that measures elapsed time.

    Usage::

        with timed("my_operation") as t:
            do_work()
        print(t["elapsed_s"])
    """
    result: dict = {"elapsed_s": 0.0, "label": label}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_s"] = time.perf_counter() - start


@asynccontextmanager
async def async_timed(label: str = "") -> AsyncGenerator[dict, None]:
    """
    Async context manager that measures elapsed time.

    Usage::

        async with async_timed("my_op") as t:
            await do_async_work()
        print(t["elapsed_s"])
    """
    result: dict = {"elapsed_s": 0.0, "label": label}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_s"] = time.perf_counter() - start


async def async_sleep_ms(ms: float) -> None:
    """Sleep for ``ms`` milliseconds asynchronously."""
    await asyncio.sleep(ms / 1000.0)


def now_ts() -> float:
    """Return current UNIX timestamp (float seconds)."""
    return time.time()
