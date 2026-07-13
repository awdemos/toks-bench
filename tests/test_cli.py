"""Tests for CLI security behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toks_bench.cli import aggregate_cli, main
from toks_bench.security import SecurityError


def test_main_rejects_traversal_output(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
defaults:
  runs: 1
providers:
  p:
    kind: llama-server
    base_url: http://localhost:8080/v1
    model: m
prompts:
  short:
    messages:
      - role: user
        content: hi
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SecurityError, match="contains '\\.\\.'"):
        main(["--config", str(cfg_path), "--provider", "p", "--prompt", "short", "--output", "../evil.json"])


def test_main_rejects_negative_runs(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
defaults:
  runs: 1
providers:
  p:
    kind: llama-server
    base_url: http://localhost:8080/v1
    model: m
prompts:
  short:
    messages:
      - role: user
        content: hi
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        main(["--config", str(cfg_path), "--provider", "p", "--prompt", "short", "--runs", "-1"])


def test_aggregate_csv_sanitizes_formula_injection(tmp_path: Path, monkeypatch) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    payload = {
        "=cmd|'/C calc'!A0": {
            "aggregate": {
                "runs": 1,
                "output_tokens_mean": 1.0,
                "output_tokens_p95": 1.0,
                "ttft_ms_mean": 1.0,
                "ttft_ms_p95": 1.0,
                "tpot_ms_mean": 1.0,
                "tpot_ms_p95": 1.0,
                "tok_per_sec_mean": 1.0,
                "tok_per_sec_median": 1.0,
                "tok_per_sec_p95": 1.0,
                "tok_per_sec_std": 0.0,
                "tool_latency_ms_mean": None,
                "tool_latency_ms_p95": None,
                "prompt_tokens_mean": None,
                "tool_calls_mean": None,
                "finish_reasons": {"stop": 1},
            },
            "runs": [],
        }
    }
    (results_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert aggregate_cli([str(results_dir)]) == 0
    csv_text = (tmp_path / "results" / "aggregate.csv").read_text(encoding="utf-8")
    assert "'=cmd|'/C calc'!A0" in csv_text
