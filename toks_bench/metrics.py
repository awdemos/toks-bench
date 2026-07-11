"""Benchmark result metrics and aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median, stdev
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class TokenTimestamp:
    """A single generated token with its arrival time."""

    index: int
    elapsed_ms: float
    text: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "index": self.index,
            "elapsed_ms": self.elapsed_ms,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenTimestamp:
        return cls(
            index=int(data["index"]),
            elapsed_ms=float(data["elapsed_ms"]),
            text=str(data.get("text", "")),
        )


@dataclass(frozen=True)
class RunResult:
    """Metrics for a single benchmark request."""

    output_tokens: int
    ttft_ms: float
    total_ms: float
    token_timestamps: tuple[TokenTimestamp, ...] = field(default_factory=tuple)
    tool_calls: tuple[dict[str, object], ...] = field(default_factory=tuple)
    tool_latency_ms: float | None = None
    prompt_tokens: int | None = None
    finish_reason: str | None = None

    @property
    def tok_per_sec(self) -> float:
        """Tokens generated per second over the full response."""
        return self.output_tokens / (self.total_ms / 1000.0)

    @property
    def tpot_ms(self) -> float:
        """Mean time per output token after the first token."""
        if self.output_tokens <= 1:
            return 0.0
        return (self.total_ms - self.ttft_ms) / (self.output_tokens - 1)

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "output_tokens": self.output_tokens,
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            "token_timestamps": [ts.as_dict() for ts in self.token_timestamps],
            "tool_calls": list(self.tool_calls),
            "tool_latency_ms": self.tool_latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "finish_reason": self.finish_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        return cls(
            output_tokens=int(data["output_tokens"]),
            ttft_ms=float(data["ttft_ms"]),
            total_ms=float(data["total_ms"]),
            token_timestamps=tuple(
                TokenTimestamp.from_dict(ts) for ts in data.get("token_timestamps", [])
            ),
            tool_calls=tuple(data.get("tool_calls", [])),
            tool_latency_ms=data.get("tool_latency_ms"),
            prompt_tokens=data.get("prompt_tokens"),
            finish_reason=data.get("finish_reason"),
        )


@dataclass(frozen=True)
class AggregateResult:
    """Aggregated metrics across multiple runs."""

    runs: int
    output_tokens_mean: float
    output_tokens_p95: float
    ttft_ms_mean: float
    ttft_ms_p95: float
    tpot_ms_mean: float
    tpot_ms_p95: float
    tok_per_sec_mean: float
    tok_per_sec_median: float
    tok_per_sec_p95: float
    tok_per_sec_std: float
    tool_latency_ms_mean: float | None
    tool_latency_ms_p95: float | None
    prompt_tokens_mean: float | None
    tool_calls_mean: float | None
    finish_reasons: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        data: dict[str, Any] = {
            "runs": self.runs,
            "output_tokens_mean": self.output_tokens_mean,
            "output_tokens_p95": self.output_tokens_p95,
            "ttft_ms_mean": self.ttft_ms_mean,
            "ttft_ms_p95": self.ttft_ms_p95,
            "tpot_ms_mean": self.tpot_ms_mean,
            "tpot_ms_p95": self.tpot_ms_p95,
            "tok_per_sec_mean": self.tok_per_sec_mean,
            "tok_per_sec_median": self.tok_per_sec_median,
            "tok_per_sec_p95": self.tok_per_sec_p95,
            "tok_per_sec_std": self.tok_per_sec_std,
        }
        if self.tool_latency_ms_mean is not None:
            data["tool_latency_ms_mean"] = self.tool_latency_ms_mean
        if self.tool_latency_ms_p95 is not None:
            data["tool_latency_ms_p95"] = self.tool_latency_ms_p95
        if self.prompt_tokens_mean is not None:
            data["prompt_tokens_mean"] = self.prompt_tokens_mean
        if self.tool_calls_mean is not None:
            data["tool_calls_mean"] = self.tool_calls_mean
        if self.finish_reasons:
            data["finish_reasons"] = self.finish_reasons
        return data


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregate plus individual runs, suitable for JSON export."""

    aggregate: AggregateResult
    runs: tuple[RunResult, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "aggregate": self.aggregate.as_dict(),
            "runs": [run.as_dict() for run in self.runs],
        }


def _percentile(values: Sequence[float], p: float) -> float:
    """Return the p-th percentile using nearest-rank."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = int((p / 100.0) * (len(sorted_values) - 1))
    return sorted_values[rank]


def aggregate(results: Sequence[RunResult]) -> AggregateResult:
    """Aggregate a sequence of RunResult objects."""
    if not results:
        return AggregateResult(
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

    tokens = [r.output_tokens for r in results]
    ttfts = [r.ttft_ms for r in results]
    tpots = [r.tpot_ms for r in results]
    tok_s = [r.tok_per_sec for r in results]
    tool_latencies = [r.tool_latency_ms for r in results if r.tool_latency_ms is not None]
    prompt_tokens = [r.prompt_tokens for r in results if r.prompt_tokens is not None]
    tool_call_counts = [len(r.tool_calls) for r in results]
    finish_reason_counts: dict[str, int] = {}
    for r in results:
        reason = r.finish_reason or "unknown"
        finish_reason_counts[reason] = finish_reason_counts.get(reason, 0) + 1

    return AggregateResult(
        runs=len(results),
        output_tokens_mean=mean(tokens),
        output_tokens_p95=_percentile(tokens, 95.0),
        ttft_ms_mean=mean(ttfts),
        ttft_ms_p95=_percentile(ttfts, 95.0),
        tpot_ms_mean=mean(tpots),
        tpot_ms_p95=_percentile(tpots, 95.0),
        tok_per_sec_mean=mean(tok_s),
        tok_per_sec_median=median(tok_s),
        tok_per_sec_p95=_percentile(tok_s, 95.0),
        tok_per_sec_std=stdev(tok_s) if len(tok_s) > 1 else 0.0,
        tool_latency_ms_mean=mean(tool_latencies) if tool_latencies else None,
        tool_latency_ms_p95=_percentile(tool_latencies, 95.0) if tool_latencies else None,
        prompt_tokens_mean=mean(prompt_tokens) if prompt_tokens else None,
        tool_calls_mean=mean(tool_call_counts) if tool_call_counts else None,
        finish_reasons=finish_reason_counts,
    )
