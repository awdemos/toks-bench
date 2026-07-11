"""Tests for metrics aggregation."""

from toks_bench.metrics import AggregateResult, RunResult, aggregate


def test_run_result_tok_per_sec() -> None:
    run = RunResult(output_tokens=100, ttft_ms=50.0, total_ms=1000.0)
    assert run.tok_per_sec == 100.0
    assert run.tpot_ms == (1000.0 - 50.0) / 99


def test_run_result_single_token() -> None:
    run = RunResult(output_tokens=1, ttft_ms=50.0, total_ms=100.0)
    assert run.tok_per_sec == 10.0
    assert run.tpot_ms == 0.0


def test_aggregate_basic() -> None:
    results = [
        RunResult(output_tokens=10, ttft_ms=100.0, total_ms=1000.0),
        RunResult(output_tokens=20, ttft_ms=200.0, total_ms=2000.0),
    ]
    agg = aggregate(results)
    assert agg.runs == 2
    assert agg.output_tokens_mean == 15.0
    assert agg.tok_per_sec_mean == (10.0 + 10.0) / 2


def test_aggregate_empty() -> None:
    agg = aggregate([])
    assert agg == AggregateResult(
        runs=0,
        output_tokens_mean=0.0,
        output_tokens_p95=0.0,
        ttft_ms_mean=0.0,
        ttft_ms_p95=0.0,
        tpot_ms_mean=0.0,
        tpot_ms_p95=0.0,
        tok_per_sec_mean=0.0,
        tok_per_sec_median=0.0,
        tok_per_sec_p95=0.0,
        tok_per_sec_std=0.0,
        tool_latency_ms_mean=None,
        tool_latency_ms_p95=None,
        prompt_tokens_mean=None,
        tool_calls_mean=None,
        finish_reasons={},
    )
