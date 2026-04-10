"""
Structured JSON logging for CrashSafe.

Provides a consistent logging interface backed by structlog with
JSON output suitable for machine processing and analysis.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

import structlog


def configure_logging(level: str = "info", json_output: bool = True) -> None:
    """
    Configure the global structlog + stdlib logging pipeline.

    Args:
        level: Log level string (debug, info, warning, error).
        json_output: If True, emit JSON lines; else emit human-readable text.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: str = "crashsafe") -> structlog.BoundLogger:
    """Return a named structlog logger."""
    return structlog.get_logger(name)
