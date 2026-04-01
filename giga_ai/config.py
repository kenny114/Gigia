"""
config.py – Config dataclass loaded from env vars + config.yaml.

Priority order (highest → lowest):
  1. Environment variables  (prefixed GIGA_, nested with __)
  2. config.yaml in the project root
  3. Hardcoded defaults below
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    provider: str = "openai"        # "openai" | "mock"
    model: str = "gpt-4o"
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_seconds: int = 60


@dataclass
class DatabaseConfig:
    sqlite_path: str = "giga_ai.db"


@dataclass
class ProxiesConfig:
    list: List[str] = field(default_factory=lambda: [""])
    rotation_strategy: str = "round_robin"   # "round_robin" | "random"


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay_seconds: float = 2.0
    backoff_factor: float = 2.0
    max_delay_seconds: float = 30.0


@dataclass
class ManagerBotConfig:
    max_concurrent_sub_bots: int = 5
    sub_bot_timeout_seconds: int = 120


@dataclass
class SubBotConfig:
    default_user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
    request_timeout_seconds: int = 30
    screenshot_on_error: bool = True
    screenshot_dir: str = "screenshots/"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"


@dataclass
class EventBusConfig:
    queue_max_size: int = 1000


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    proxies: ProxiesConfig = field(default_factory=ProxiesConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    manager_bot: ManagerBotConfig = field(default_factory=ManagerBotConfig)
    sub_bot: SubBotConfig = field(default_factory=SubBotConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    event_bus: EventBusConfig = field(default_factory=EventBusConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _merge_dict(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _env_overrides() -> dict:
    """
    Collect GIGA_* env vars and turn them into a nested dict.
    E.g. GIGA_LLM__API_KEY="sk-..." → {"llm": {"api_key": "sk-..."}}
    """
    overrides: dict = {}
    prefix = "GIGA_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        # strip prefix, lower-case, split on double-underscore
        parts = key[len(prefix):].lower().split("__")
        node = overrides
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return overrides


def _apply_dict_to_config(cfg: Config, data: dict) -> None:
    """Apply a (possibly nested) dict onto the Config dataclass tree."""
    section_map = {
        "llm": ("llm", LLMConfig),
        "database": ("database", DatabaseConfig),
        "proxies": ("proxies", ProxiesConfig),
        "retry": ("retry", RetryConfig),
        "manager_bot": ("manager_bot", ManagerBotConfig),
        "sub_bot": ("sub_bot", SubBotConfig),
        "logging": ("logging", LoggingConfig),
        "event_bus": ("event_bus", EventBusConfig),
    }
    for section, (attr, klass) in section_map.items():
        if section in data:
            sub_data = data[section]
            sub_obj = getattr(cfg, attr)
            for k, v in sub_data.items():
                if hasattr(sub_obj, k):
                    # Attempt type coercion to the field's current type
                    current = getattr(sub_obj, k)
                    try:
                        if isinstance(current, bool):
                            v = str(v).lower() in ("1", "true", "yes")
                        elif current is not None:
                            v = type(current)(v)
                    except (ValueError, TypeError):
                        pass
                    setattr(sub_obj, k, v)


def load_config(yaml_path: Optional[str] = None) -> Config:
    """
    Build and return a Config instance.

    Parameters
    ----------
    yaml_path:
        Path to the YAML config file.  Defaults to ``config.yaml`` in the
        current working directory (or the directory two levels above this
        module file).
    """
    cfg = Config()

    # 1. YAML
    if yaml_path is None:
        candidates = [
            Path("config.yaml"),
            Path(__file__).parent.parent / "config.yaml",
        ]
        for p in candidates:
            if p.exists():
                yaml_path = str(p)
                break

    if yaml_path and Path(yaml_path).exists():
        with open(yaml_path, "r", encoding="utf-8") as fh:
            yaml_data: dict = yaml.safe_load(fh) or {}
        _apply_dict_to_config(cfg, yaml_data)

    # 2. Environment variable overrides
    env_data = _env_overrides()
    _apply_dict_to_config(cfg, env_data)

    # 3. Fallback: plain OPENAI_API_KEY
    if not cfg.llm.api_key:
        cfg.llm.api_key = os.getenv("OPENAI_API_KEY", "")

    return cfg


# Singleton – importable directly
_config: Optional[Config] = None


def get_config() -> Config:
    """Return the process-level singleton Config (lazy-initialised)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
