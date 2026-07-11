"""YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from toks_bench.providers import Provider, ToolConfig


class ConfigError(Exception):
    """Raised when the configuration file is invalid."""


def load_config(path: Path | str = Path("config.yaml")) -> dict[str, Any]:
    """Load and return the raw YAML configuration."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _require_mapping(data: Any, name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{name} must be a mapping")
    return data


def get_providers(config: dict[str, Any]) -> list[Provider]:
    """Return validated provider list from config."""
    raw_providers = _require_mapping(config.get("providers"), "providers")
    providers: list[Provider] = []
    for name, fields in raw_providers.items():
        fields = _require_mapping(fields, f"providers.{name}")
        for key in ("kind", "base_url", "model"):
            if key not in fields:
                raise ConfigError(f"providers.{name} missing required key '{key}'")
        providers.append(
            Provider(
                name=name,
                kind=fields["kind"],
                base_url=fields["base_url"],
                model=fields["model"],
            )
        )
    return providers


def get_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Return benchmark defaults, falling back to sensible values."""
    defaults = _require_mapping(config.get("defaults", {}), "defaults")
    return {
        "runs": defaults.get("runs", 5),
        "max_tokens": defaults.get("max_tokens", 512),
        "temperature": defaults.get("temperature", 0.7),
        "top_p": defaults.get("top_p", 0.9),
    }


def get_prompt(
    config: dict[str, Any],
    name: str,
    *,
    config_path: Path | str = Path("config.yaml"),
) -> list[dict[str, str]]:
    """Return the message list for a named prompt.

    Relative prompt file paths are resolved against the directory containing
    the configuration file.
    """
    prompts = _require_mapping(config.get("prompts"), "prompts")
    if name not in prompts:
        raise ConfigError(f"prompt '{name}' not found")
    prompt = _require_mapping(prompts[name], f"prompts.{name}")

    if "messages" in prompt:
        return [
            {"role": str(m["role"]), "content": str(m["content"])}
            for m in prompt["messages"]
        ]

    if "file" in prompt:
        config_dir = Path(config_path).resolve().parent
        prompt_path = config_dir / prompt["file"]
        if not prompt_path.exists():
            raise ConfigError(f"prompt file not found: {prompt_path}")
        return [{"role": "user", "content": prompt_path.read_text(encoding="utf-8")}]

    raise ConfigError(f"prompts.{name} must define 'messages' or 'file'")


def get_tools(
    config: dict[str, Any],
    name: str,
    *,
    tool_choice: object = "auto",
) -> ToolConfig:
    """Return tool configuration for a named prompt.

    Raises ConfigError if the prompt has no ``tools`` section.
    """
    prompts = _require_mapping(config.get("prompts"), "prompts")
    if name not in prompts:
        raise ConfigError(f"prompt '{name}' not found")
    prompt = _require_mapping(prompts[name], f"prompts.{name}")
    tools = prompt.get("tools")
    if not tools:
        raise ConfigError(f"prompts.{name} has no 'tools' section")
    if not isinstance(tools, list):
        raise ConfigError(f"prompts.{name}.tools must be a list")
    return ToolConfig(tools=list(tools), tool_choice=tool_choice)


def is_tool_prompt(config: dict[str, Any], name: str) -> bool:
    """Return True if the named prompt has a tools section."""
    prompts = _require_mapping(config.get("prompts"), "prompts")
    if name not in prompts:
        return False
    prompt = _require_mapping(prompts[name], f"prompts.{name}")
    return bool(prompt.get("tools"))
