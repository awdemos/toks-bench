"""Core benchmark loop."""

from __future__ import annotations

import signal
import sys
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from toks_bench.metrics import (
    BenchmarkResult,
    RunResult,
    TokenTimestamp,
    aggregate,
)
from toks_bench.providers import complete_stream, complete_tool, create_client

if TYPE_CHECKING:
    from toks_bench.metrics import AggregateResult as AggregateResultT
    from toks_bench.providers import Provider, ToolConfig


class _RunTimeoutError(TimeoutError):
    """Raised when a single benchmark run exceeds its wall-clock budget."""


_RUN_TIMEOUT_SECONDS = 180.0


def _with_run_timeout(func: Callable[[], RunResult]) -> RunResult:
    """Wrap a single benchmark run with a wall-clock timeout.

    The OpenAI client timeout guards individual HTTP operations, but some
    servers keep a streaming connection open without producing further chunks.
    This wrapper aborts the whole run after ``_RUN_TIMEOUT_SECONDS`` so the
    benchmark loop can continue with the next run.
    """

    def _handler(_signum: int, _frame: object) -> None:
        raise _RunTimeoutError(
            f"Run exceeded {_RUN_TIMEOUT_SECONDS}s wall-clock timeout"
        )

    old_handler = signal.signal(signal.SIGALRM, _handler)
    old_alarm = signal.alarm(int(_RUN_TIMEOUT_SECONDS))
    try:
        return func()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_alarm:
            signal.alarm(old_alarm)


def _run_generation(
    provider: Provider,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> RunResult:
    """Execute a single generation request and collect per-token timestamps."""

    def _inner() -> RunResult:
        client = create_client(provider)
        start = time.perf_counter()
        first_token_time: float | None = None
        output_tokens = 0
        usage_tokens: int | None = None
        prompt_tokens: int | None = None
        finish_reason: str | None = None
        timestamps: list[TokenTimestamp] = []

        for delta, is_first, usage, prompt, finish in complete_stream(
            client,
            provider,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            now = time.perf_counter()
            if is_first:
                first_token_time = now
            if delta:
                idx = output_tokens
                output_tokens += 1
                timestamps.append(
                    TokenTimestamp(
                        index=idx,
                        elapsed_ms=(now - start) * 1000.0,
                        text=delta,
                    )
                )
            if usage is not None:
                usage_tokens = usage
            if prompt is not None:
                prompt_tokens = prompt
            if finish is not None:
                finish_reason = finish

        end = time.perf_counter()
        total_ms = (end - start) * 1000.0
        ttft_ms = (
            (first_token_time - start) * 1000.0
            if first_token_time is not None
            else total_ms
        )

        final_tokens = usage_tokens if usage_tokens is not None else output_tokens
        return RunResult(
            output_tokens=final_tokens,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            token_timestamps=tuple(timestamps),
            prompt_tokens=prompt_tokens,
            finish_reason=finish_reason,
        )

    return _with_run_timeout(_inner)


def _run_tool(
    provider: Provider,
    messages: list[dict[str, str]],
    tools: ToolConfig,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> RunResult:
    """Execute a single tool-calling request and return parsed tool calls."""

    def _inner() -> RunResult:
        client = create_client(provider)
        start = time.perf_counter()
        calls, completion_tokens, prompt_tokens, finish_reason = complete_tool(
            client,
            provider,
            messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        end = time.perf_counter()
        total_ms = (end - start) * 1000.0

        return RunResult(
            output_tokens=completion_tokens if completion_tokens is not None else 0,
            ttft_ms=total_ms,
            total_ms=total_ms,
            tool_calls=tuple(calls),
            tool_latency_ms=total_ms,
            prompt_tokens=prompt_tokens,
            finish_reason=finish_reason,
        )

    return _with_run_timeout(_inner)


def _failed_run_result(finish_reason: str) -> RunResult:
    """Return a sentinel RunResult for a run that timed out or otherwise failed.

    Uses a tiny positive ``total_ms`` so aggregate tok/s calculations do not
    divide by zero and produce NaN.
    """
    return RunResult(
        output_tokens=0,
        ttft_ms=0.0,
        total_ms=1.0,
        token_timestamps=(),
        tool_calls=(),
        tool_latency_ms=1.0,
        prompt_tokens=None,
        finish_reason=finish_reason,
    )


def benchmark_provider_full(
    provider: Provider,
    messages: list[dict[str, str]],
    *,
    runs: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> BenchmarkResult:
    """Run a generation benchmark and return aggregate plus individual runs."""
    results: list[RunResult] = []
    for i in range(runs):
        try:
            results.append(
                _run_generation(
                    provider,
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            )
        except Exception as exc:
            print(
                f"WARNING: run {i + 1}/{runs} for {provider.name} failed: {exc}",
                file=sys.stderr,
            )
            results.append(_failed_run_result(f"error: {type(exc).__name__}: {exc}"))
    return BenchmarkResult(aggregate=aggregate(results), runs=tuple(results))


def benchmark_provider(
    provider: Provider,
    messages: list[dict[str, str]],
    *,
    runs: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> AggregateResultT:
    """Run a generation benchmark and return aggregated metrics only."""
    return benchmark_provider_full(
        provider,
        messages,
        runs=runs,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    ).aggregate


def benchmark_tool_provider_full(
    provider: Provider,
    messages: list[dict[str, str]],
    *,
    tools: ToolConfig,
    runs: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> BenchmarkResult:
    """Run a tool-calling benchmark and return aggregate plus individual runs."""
    results: list[RunResult] = []
    for i in range(runs):
        try:
            results.append(
                _run_tool(
                    provider,
                    messages,
                    tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            )
        except Exception as exc:
            print(
                f"WARNING: run {i + 1}/{runs} for {provider.name} failed: {exc}",
                file=sys.stderr,
            )
            results.append(_failed_run_result(f"error: {type(exc).__name__}: {exc}"))
    return BenchmarkResult(aggregate=aggregate(results), runs=tuple(results))


def benchmark_tool_provider(
    provider: Provider,
    messages: list[dict[str, str]],
    *,
    tools: ToolConfig,
    runs: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> AggregateResultT:
    """Run a tool-calling benchmark and return aggregated metrics only."""
    return benchmark_tool_provider_full(
        provider,
        messages,
        tools=tools,
        runs=runs,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    ).aggregate
