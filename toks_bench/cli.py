"""Command-line interface for toks-bench."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tabulate import tabulate

from toks_bench.bench import benchmark_provider_full, benchmark_tool_provider_full
from toks_bench.config import (
    get_defaults,
    get_prompt,
    get_providers,
    get_tools,
    is_tool_prompt,
    load_config,
)
from toks_bench.metrics import AggregateResult, BenchmarkResult
from toks_bench.security import resolve_contained_path, sanitize_csv_field

_MAX_RUNS = 10_000
_MAX_TOKENS = 1_000_000


def _format_table(results: list[tuple[str, AggregateResult]]) -> str:
    headers = [
        "Provider",
        "Runs",
        "Tok mean",
        "TTFT mean (ms)",
        "TPOT mean (ms)",
        "tok/s mean",
        "tok/s median",
        "tok/s p95",
        "tok/s std",
    ]
    rows = []
    for name, agg in results:
        rows.append(
            [
                name,
                agg.runs,
                f"{agg.output_tokens_mean:.1f}",
                f"{agg.ttft_ms_mean:.1f}",
                f"{agg.tpot_ms_mean:.1f}",
                f"{agg.tok_per_sec_mean:.1f}",
                f"{agg.tok_per_sec_median:.1f}",
                f"{agg.tok_per_sec_p95:.1f}",
                f"{agg.tok_per_sec_std:.1f}",
            ]
        )
    return tabulate(rows, headers=headers, tablefmt="github")


def _serialize_aggregate(agg: AggregateResult) -> dict[str, Any]:
    """Convert an AggregateResult to a JSON-serializable dict."""
    return agg.as_dict()


def _format_json(results: list[tuple[str, BenchmarkResult]]) -> str:
    payload: dict[str, Any] = {
        name: {
            "aggregate": _serialize_aggregate(result.aggregate),
            "runs": [run.as_dict() for run in result.runs],
        }
        for name, result in results
    }
    return json.dumps(payload, indent=2)


def _make_positive_int(name: str, *, maximum: int | None = None) -> Callable[[str], int]:
    def _validate(value: str) -> int:
        ivalue = _validate_positive_int(int(value), name, maximum=maximum)
        return ivalue

    return _validate


def _validate_positive_int(value: int, name: str, *, maximum: int | None = None) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return value


def _run_benchmark(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> list[tuple[str, BenchmarkResult]]:
    """Run generation or tool benchmark based on prompt type."""
    defaults = get_defaults(config)
    providers = get_providers(config, allow_internal_urls=args.allow_internal_urls)
    messages = get_prompt(config, args.prompt, config_path=args.config)

    if not providers:
        print("No providers configured", file=sys.stderr)
        sys.exit(1)

    if args.all:
        selected = providers
    elif args.provider:
        selected = [p for p in providers if p.name == args.provider]
        if not selected:
            print(f"Provider '{args.provider}' not found", file=sys.stderr)
            sys.exit(1)
    else:
        print("Specify --provider or --all", file=sys.stderr)
        sys.exit(1)

    runs = _validate_positive_int(
        args.runs if args.runs is not None else defaults["runs"], "--runs", maximum=_MAX_RUNS
    )
    max_tokens = _validate_positive_int(
        args.max_tokens if args.max_tokens is not None else defaults["max_tokens"],
        "--max-tokens",
        maximum=_MAX_TOKENS,
    )
    tool_max_tokens = _validate_positive_int(
        args.tool_max_tokens if args.tool_max_tokens is not None else 1024,
        "--tool-max-tokens",
        maximum=_MAX_TOKENS,
    )

    tool_mode = args.mode == "tool" or (args.mode is None and is_tool_prompt(config, args.prompt))
    tools = get_tools(config, args.prompt) if tool_mode else None

    results: list[tuple[str, BenchmarkResult]] = []
    for provider in selected:
        print(f"Benchmarking {provider.name} ({provider.model}) ...", file=sys.stderr)
        if tool_mode:
            if tools is None:
                raise ValueError(f"Tool prompt {args.prompt!r} has no tools configured")
            result = benchmark_tool_provider_full(
                provider,
                messages,
                tools=tools,
                runs=runs,
                max_tokens=tool_max_tokens,
                temperature=defaults["temperature"],
                top_p=defaults["top_p"],
            )
        else:
            result = benchmark_provider_full(
                provider,
                messages,
                runs=runs,
                max_tokens=max_tokens,
                temperature=defaults["temperature"],
                top_p=defaults["top_p"],
            )
        results.append((provider.name, result))

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark LLM token throughput")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--provider", type=str, help="Provider name from config")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt name from config")
    parser.add_argument("--all", action="store_true", help="Benchmark all providers")
    parser.add_argument(
        "--runs", type=_make_positive_int("--runs", maximum=_MAX_RUNS), help="Override number of runs"
    )
    parser.add_argument(
        "--max-tokens",
        type=_make_positive_int("--max-tokens", maximum=_MAX_TOKENS),
        help="Override max_tokens for generation",
    )
    parser.add_argument(
        "--tool-max-tokens",
        type=_make_positive_int("--tool-max-tokens", maximum=_MAX_TOKENS),
        help="Override max_tokens for tool-calling mode",
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "tool", "auto"],
        default="auto",
        help="Benchmark mode: auto detects from prompt config",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format",
    )
    parser.add_argument("--output", type=Path, help="Write results to file")
    parser.add_argument(
        "--allow-internal-urls",
        action="store_true",
        help="Allow provider base_url values pointing to private/internal hosts",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    results = _run_benchmark(args, config)

    output_json = _format_json(results)
    output_table = _format_table([(name, result.aggregate) for name, result in results])

    print(output_table if args.format == "table" else output_json)

    if args.output:
        workspace = Path.cwd()
        output_path = resolve_contained_path(args.output, workspace)
        output_path.write_text(output_json, encoding="utf-8")

    return 0


def aggregate_cli(argv: list[str] | None = None) -> int:
    """Aggregate individual JSON result files into a CSV and markdown report."""
    parser = argparse.ArgumentParser(description="Aggregate toks-bench JSON results")
    parser.add_argument("results_dir", type=Path, default=Path("results"))
    parser.add_argument("--csv", type=Path, default=Path("results/aggregate.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/aggregate-report.md"))
    args = parser.parse_args(argv)

    workspace = Path.cwd()
    csv_path = resolve_contained_path(args.csv, workspace)
    report_path = resolve_contained_path(args.report, workspace)

    rows: list[dict[str, Any]] = []
    for path in sorted(args.results_dir.glob("*.json")):
        if path.name in (args.csv.name, args.report.name):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        for provider, payload in data.items():
            if not isinstance(payload, dict) or "aggregate" not in payload:
                continue
            metrics = payload["aggregate"]
            row: dict[str, Any] = {"file": path.name, "provider": provider}
            row.update(metrics)
            rows.append(row)

    if not rows:
        print("No result JSON files found", file=sys.stderr)
        return 1

    fieldnames = sorted({key for row in rows for key in row})

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            safe_row = {k: sanitize_csv_field(v) for k, v in row.items()}
            writer.writerow(safe_row)

    headers = fieldnames
    table_rows = [[str(row.get(h, "")) for h in headers] for row in rows]
    report = "# Aggregate Benchmark Results\n\n"
    report += tabulate(table_rows, headers=headers, tablefmt="github")
    report += "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"Wrote {csv_path} and {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
