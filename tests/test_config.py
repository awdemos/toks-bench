"""Tests for configuration loading."""

from pathlib import Path

import pytest

from toks_bench.config import ConfigError, get_defaults, get_prompt, get_providers, load_config

SAMPLE_CONFIG = """
defaults:
  runs: 3
  max_tokens: 128
  temperature: 0.5
  top_p: 0.95

providers:
  a:
    kind: llama-server
    base_url: http://localhost:8080/v1
    model: model-a
  b:
    kind: vllm
    base_url: http://localhost:8000/v1
    model: model-b

prompts:
  short:
    messages:
      - role: user
        content: hello
  file_prompt:
    file: prompts/file.txt
"""


def test_load_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(SAMPLE_CONFIG, encoding="utf-8")
    config = load_config(cfg_path)
    assert config["defaults"]["runs"] == 3


def test_get_providers(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(SAMPLE_CONFIG, encoding="utf-8")
    config = load_config(cfg_path)
    providers = get_providers(config)
    assert [p.name for p in providers] == ["a", "b"]
    assert providers[0].kind == "llama-server"


def test_get_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(SAMPLE_CONFIG, encoding="utf-8")
    defaults = get_defaults(load_config(cfg_path))
    assert defaults == {"runs": 3, "max_tokens": 128, "temperature": 0.5, "top_p": 0.95}


def test_get_prompt_messages(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(SAMPLE_CONFIG, encoding="utf-8")
    messages = get_prompt(load_config(cfg_path), "short")
    assert messages == [{"role": "user", "content": "hello"}]


def test_get_prompt_file(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "file.txt").write_text("file content", encoding="utf-8")
    cfg_path.write_text(SAMPLE_CONFIG, encoding="utf-8")
    messages = get_prompt(load_config(cfg_path), "file_prompt", config_path=cfg_path)
    assert messages == [{"role": "user", "content": "file content"}]


def test_missing_provider_key(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "providers:\n  a:\n    kind: llama-server\n    base_url: http://x/v1\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="missing required key 'model'"):
        get_providers(load_config(cfg_path))


def test_missing_prompt(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("prompts:\n  short:\n    messages: []\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="prompt 'missing' not found"):
        get_prompt(load_config(cfg_path), "missing")
