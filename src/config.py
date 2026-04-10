"""
Central configuration management for CrashSafe.

Loads configuration from YAML files with environment variable overrides.
All configuration is strongly typed via Pydantic BaseSettings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"


class RouterConfig(BaseModel):
    strategy: Literal["round_robin", "least_loaded", "random"] = "round_robin"
    health_check_interval_s: float = 5.0
    request_timeout_s: float = 60.0
    max_queue_depth: int = 100


class WorkerConfig(BaseModel):
    num_workers: int = 2
    worker_host: str = "127.0.0.1"
    worker_base_port: int = 8100
    startup_timeout_s: float = 30.0


class BackendConfig(BaseModel):
    type: Literal["mock", "transformers", "vllm"] = "mock"
    model_name: str = "gpt2"
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "cpu"


class MockBackendConfig(BaseModel):
    tokens_per_second: float = 50.0
    base_latency_ms: float = 10.0
    token_vocab_size: int = 50257


class RecoveryConfig(BaseModel):
    strategy: Literal[
        "fail_stop", "retry_from_scratch", "token_commit_resume"
    ] = "token_commit_resume"
    max_retries: int = 3
    retry_backoff_base_s: float = 0.5
    retry_backoff_max_s: float = 10.0
    checkpoint_every_n_tokens: int = 10


class StorageConfig(BaseModel):
    backend: Literal["disk", "memory"] = "disk"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    log_dir: str = "./logs"


class MetricsConfig(BaseModel):
    enabled: bool = True
    log_file: str = "./logs/metrics.jsonl"
    flush_interval_s: float = 1.0


class FaultInjectionConfig(BaseModel):
    enabled: bool = False
    default_fault_type: Literal[
        "none", "kill", "delay", "hang", "graceful_shutdown"
    ] = "none"
    delay_ms: float = 500.0
    hang_duration_s: float = 30.0


class ExperimentConfig(BaseModel):
    concurrency_levels: List[int] = [1, 4, 8]
    prompt_lengths: Dict[str, int] = {
        "short": 32,
        "medium": 128,
        "long": 512,
    }
    fault_types: List[str] = ["none", "kill", "delay", "hang"]
    recovery_strategies: List[str] = [
        "fail_stop",
        "retry_from_scratch",
        "token_commit_resume",
    ]
    requests_per_cell: int = 20
    output_dir: str = "./results"
    figures_dir: str = "./figures"


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Root application configuration loaded from YAML."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    mock_backend: MockBackendConfig = Field(default_factory=MockBackendConfig)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    fault_injection: FaultInjectionConfig = Field(
        default_factory=FaultInjectionConfig
    )
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load application configuration from a YAML file.

    Falls back to built-in defaults if no path is provided.
    Environment variable ``CRASHSAFE_CONFIG`` can also specify the path.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Fully populated AppConfig instance.
    """
    if config_path is None:
        config_path = os.environ.get(
            "CRASHSAFE_CONFIG",
            str(Path(__file__).parent.parent / "configs" / "default.yaml"),
        )

    path = Path(config_path)
    if path.exists():
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    return AppConfig(**raw)


# Module-level singleton (lazy-loaded)
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Return the global AppConfig singleton, loading it if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(cfg: AppConfig) -> None:
    """Override the global config singleton (useful for testing)."""
    global _config
    _config = cfg
