"""Generate a Spark-RAPIDS-style profiling report from toks-bench aggregate data."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tabulate import tabulate

from toks_bench.config import load_config
from toks_bench.security import escape_markdown


def _load_provider_models(config_path: Path) -> dict[str, str]:
    """Return a mapping from provider name to model name."""
    try:
        config = load_config(config_path)
    except (FileNotFoundError, OSError):
        return {}
    providers = config.get("providers", {})
    if not isinstance(providers, dict):
        return {}
    return {
        name: str(fields.get("model", "unknown"))
        for name, fields in providers.items()
        if isinstance(fields, dict)
    }

def _parse_finish_reasons(text: str) -> dict[str, int]:
    """Parse a finish_reasons cell like {'length': 3, 'error: ...': 1}."""
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return {}


def _success_rate(finish_reasons: dict[str, int], runs: int) -> float:
    """Fraction of runs that did not report an error."""
    if runs == 0:
        return 0.0
    errors = sum(count for key, count in finish_reasons.items() if key.startswith("error:"))
    return (runs - errors) / runs


def _load_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["finish_reasons"] = _parse_finish_reasons(row.get("finish_reasons", ""))
            for key in (
                "output_tokens_mean",
                "output_tokens_p95",
                "runs",
                "tok_per_sec_mean",
                "tok_per_sec_median",
                "tok_per_sec_p95",
                "tok_per_sec_std",
                "tool_calls_mean",
                "tool_latency_ms_mean",
                "tool_latency_ms_p95",
                "tpot_ms_mean",
                "tpot_ms_p95",
                "ttft_ms_mean",
                "ttft_ms_p95",
            ):
                try:
                    row[key] = float(row[key]) if "." in row[key] else int(row[key])
                except (KeyError, ValueError):
                    row[key] = 0.0
            rows.append(row)
    return rows


def _load_json_metadata(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load provider metadata from result JSON files."""
    meta: dict[str, dict[str, Any]] = defaultdict(dict)
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        for provider, payload in data.items():
            if not isinstance(payload, dict):
                continue
            # prefer model name from first file seen
            if "model" not in meta[provider] and isinstance(payload.get("provider"), dict):
                meta[provider]["model"] = payload["provider"].get("model", "unknown")
    return meta


def _group_by_provider(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["provider"]].append(row)
    return dict(groups)


def _provider_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_runs = sum(int(row["runs"]) for row in rows)
    successful_runs = sum(
        int(row["runs"]) * _success_rate(row["finish_reasons"], int(row["runs"])) for row in rows
    )
    mean_tok_per_sec = sum(float(r["tok_per_sec_mean"]) for r in rows) / len(rows)
    mean_ttft = sum(float(r["ttft_ms_mean"]) for r in rows) / len(rows)
    mean_tpot = sum(float(r["tpot_ms_mean"]) for r in rows) / len(rows)
    return {
        "prompts": len(rows),
        "total_runs": total_runs,
        "successful_runs": successful_runs,
        "success_rate": successful_runs / total_runs if total_runs else 0.0,
        "mean_tok_per_sec": mean_tok_per_sec,
        "mean_ttft_ms": mean_ttft,
        "mean_tpot_ms": mean_tpot,
    }


def _prompt_from_file(file_name: str) -> str:
    """Extract prompt name from result file like provider-prompt.json."""
    # strip provider prefix and .json suffix; provider names may contain hyphens.
    if file_name.endswith(".json"):
        file_name = file_name[:-5]
    # The last segment after the final hyphen is the prompt name in this repo.
    if "-" in file_name:
        return file_name.rsplit("-", 1)[-1]
    return file_name


def _generate_report(
    rows: list[dict[str, Any]],
    results_dir: Path,
    csv_path: Path,
    config_path: Path,
) -> str:
    meta = _load_json_metadata(results_dir)
    provider_models = _load_provider_models(config_path)
    grouped = _group_by_provider(rows)
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = []
    lines.append("# LLM Inference Profiling Report")
    lines.append("")
    lines.append("> Generated in the style of the Spark-RAPIDS Profiling Tool.")
    lines.append(f"> Source: `{csv_path}`")
    lines.append(f"> Generated at: {now}")
    lines.append("")

    # 1. Application Summary
    lines.append("## 1. Application Summary")
    lines.append("")
    lines.append(
        "This section mirrors the Spark-RAPIDS *Application Summary*: one row per "
        "inference backend (application), with aggregate duration and success metrics."
    )
    lines.append("")

    app_rows = []
    for provider in sorted(grouped):
        summary = _provider_summary(grouped[provider])
        model = provider_models.get(provider, meta.get(provider, {}).get("model", "unknown"))
        app_rows.append(
            [
                escape_markdown(provider),
                escape_markdown(model),
                summary["prompts"],
                f"{summary['successful_runs']:.0f}/{summary['total_runs']}",
                f"{summary['success_rate']:.1%}",
                f"{summary['mean_tok_per_sec']:.1f}",
                f"{summary['mean_ttft_ms']:.0f}",
                f"{summary['mean_tpot_ms']:.0f}",
            ]
        )
    lines.append(
        tabulate(
            app_rows,
            headers=[
                "Provider",
                "Model",
                "Prompts",
                "Runs (ok/total)",
                "Success rate",
                "Mean tok/s",
                "Mean TTFT (ms)",
                "Mean TPOT (ms)",
            ],
            tablefmt="github",
        )
    )
    lines.append("")

    # 2. Stage Statistics (per provider x prompt)
    lines.append("## 2. Stage Statistics")
    lines.append("")
    lines.append(
        "This section mirrors *Stage Statistics*: each (provider, prompt) combination "
        "is a stage, with duration metrics and throughput."
    )
    lines.append("")

    stage_rows = []
    for provider in sorted(grouped):
        for row in grouped[provider]:
            prompt = _prompt_from_file(row["file"])
            stage_rows.append(
                [
                    escape_markdown(provider),
                    escape_markdown(prompt),
                    int(row["runs"]),
                    f"{_success_rate(row['finish_reasons'], int(row['runs'])):.1%}",
                    f"{float(row['output_tokens_mean']):.0f}",
                    f"{float(row['tok_per_sec_mean']):.1f}",
                    f"{float(row['ttft_ms_mean']):.0f}",
                    f"{float(row['tpot_ms_mean']):.0f}",
                    f"{float(row['tok_per_sec_std']):.1f}",
                ]
            )
    lines.append(
        tabulate(
            stage_rows,
            headers=[
                "Provider",
                "Prompt",
                "Runs",
                "Success",
                "Out tok mean",
                "tok/s mean",
                "TTFT mean (ms)",
                "TPOT mean (ms)",
                "tok/s std",
            ],
            tablefmt="github",
        )
    )
    lines.append("")

    # 3. Duration Analysis
    lines.append("## 3. Duration Analysis")
    lines.append("")
    lines.append("Rankings of providers by the three primary latency dimensions.")
    lines.append("")

    providers_rank = [
        (provider, _provider_summary(grouped[provider])) for provider in grouped
    ]
    # tok/s: higher is better
    top_tok = sorted(providers_rank, key=lambda x: x[1]["mean_tok_per_sec"], reverse=True)[:3]
    # TTFT: lower is better
    top_ttft = sorted(providers_rank, key=lambda x: x[1]["mean_ttft_ms"])[:3]
    # TPOT: lower is better
    top_tpot = sorted(providers_rank, key=lambda x: x[1]["mean_tpot_ms"])[:3]

    lines.append("### 3.1 Highest throughput (tok/s)")
    lines.append("")
    for i, (provider, summary) in enumerate(top_tok, 1):
        lines.append(
            f"{i}. **{provider}** — {summary['mean_tok_per_sec']:.1f} tok/s mean, "
            f"TTFT {summary['mean_ttft_ms']:.0f} ms, TPOT {summary['mean_tpot_ms']:.0f} ms"
        )
    lines.append("")
    lines.append("### 3.2 Lowest time-to-first-token (TTFT)")
    lines.append("")
    for i, (provider, summary) in enumerate(top_ttft, 1):
        lines.append(
            f"{i}. **{provider}** — TTFT {summary['mean_ttft_ms']:.0f} ms, "
            f"{summary['mean_tok_per_sec']:.1f} tok/s"
        )
    lines.append("")
    lines.append("### 3.3 Lowest time-per-output-token (TPOT)")
    lines.append("")
    for i, (provider, summary) in enumerate(top_tpot, 1):
        lines.append(
            f"{i}. **{provider}** — TPOT {summary['mean_tpot_ms']:.0f} ms, "
            f"{summary['mean_tok_per_sec']:.1f} tok/s"
        )
    lines.append("")

    # 4. Bottleneck / Issue Summary
    lines.append("## 4. Bottleneck / Issue Summary")
    lines.append("")
    lines.append(
        "This section mirrors the Spark-RAPIDS *Potential Problems* view. "
        "Each entry flags an observation that impacts performance or correctness."
    )
    lines.append("")

    issues: list[tuple[str, str, str]] = []
    for row in rows:
        provider = row["provider"]
        prompt = _prompt_from_file(row["file"])
        runs = int(row["runs"])
        success = _success_rate(row["finish_reasons"], runs)
        if success < 1.0:
            reasons = ", ".join(
                f"{escape_markdown(str(k))}: {escape_markdown(str(v))}"
                for k, v in row["finish_reasons"].items()
            )
            issues.append((provider, prompt, f"partial failure ({reasons})"))
        if float(row["ttft_ms_mean"]) > 2000:
            issues.append(
                (provider, prompt, f"high TTFT: {float(row['ttft_ms_mean']):.0f} ms")
            )
        if float(row["tpot_ms_mean"]) > 100:
            issues.append(
                (
                    provider,
                    prompt,
                    f"high TPOT: {float(row['tpot_ms_mean']):.0f} ms/token",
                )
            )
        if float(row["tok_per_sec_std"]) > 5.0:
            issues.append(
                (
                    provider,
                    prompt,
                    f"high throughput variance: "
                    f"σ={float(row['tok_per_sec_std']):.1f} tok/s",
                )
            )
        if "ollama" in provider and float(row["tok_per_sec_mean"]) < 15:
            issues.append((provider, prompt, "Ollama throughput is CPU-bound; run on GPU"))

    if issues:
        issue_rows = [
            [escape_markdown(provider), escape_markdown(prompt), escape_markdown(desc)]
            for provider, prompt, desc in sorted(issues)
        ]
        lines.append(
            tabulate(
                issue_rows,
                headers=["Provider", "Prompt", "Observation"],
                tablefmt="github",
            )
        )
    else:
        lines.append("No major issues detected.")
    lines.append("")

    # 5. Tuning Recommendations
    lines.append("## 5. Tuning Recommendations")
    lines.append("")
    lines.append(
        "Spark-RAPIDS-style actionable recommendations derived from the duration and "
        "bottleneck analysis."
    )
    lines.append("")

    recs: list[str] = []
    for provider in sorted(grouped):
        safe_provider = escape_markdown(provider)
        summary = _provider_summary(grouped[provider])
        rows_p = grouped[provider]
        has_errors = any(
            _success_rate(r["finish_reasons"], int(r["runs"])) < 1.0 for r in rows_p
        )
        if summary["success_rate"] < 1.0:
            recs.append(
                f"**{safe_provider}**: {1 - summary['success_rate']:.1%} "
                "of runs failed or timed out. Increase `--max-model-len` or per-run "
                "timeout; verify server health before benchmarking."
            )
        if summary["mean_tok_per_sec"] < 10:
            recs.append(
                f"**{safe_provider}**: throughput ({summary['mean_tok_per_sec']:.1f} tok/s) "
                "is very low. Ensure the model is fully GPU-offloaded (`-ngl 99`) "
                "and not falling back to CPU."
            )
        if summary["mean_ttft_ms"] > 2000:
            recs.append(
                f"**{safe_provider}**: TTFT is {summary['mean_ttft_ms']:.0f} ms on average. "
                "Consider enabling prefix caching, reducing context size, or using a "
                "smaller draft model."
            )
        if summary["mean_tpot_ms"] > 100:
            recs.append(
                f"**{safe_provider}**: TPOT is {summary['mean_tpot_ms']:.0f} ms/token. "
                "Profile with `nsys` to identify whether the bottleneck is memory "
                "bandwidth or kernel launch."
            )
        if not has_errors and summary["mean_tok_per_sec"] >= 30:
            recs.append(
                f"**{safe_provider}**: healthy high-throughput configuration. "
                "Capture as baseline."
            )

    for rec in recs:
        lines.append(f"- {rec}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "*Report format inspired by the Spark-RAPIDS Profiling Tool. Metrics are produced by "
        "`toks-bench`; this report does not instrument the GPU directly.*"
    )
    lines.append("")

    return "\n".join(lines)


def profiling_report_cli(argv: list[str] | None = None) -> int:
    """CLI entry point for the profiling report generator."""
    parser = argparse.ArgumentParser(
        description="Generate a Spark-RAPIDS-style profiling report from toks-bench results",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/full"),
        help="Directory containing per-run JSON result files",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/aggregate.csv"),
        help="Path to aggregate CSV produced by toks-bench-aggregate",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to toks-bench config.yaml for provider metadata",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/profiling-report.md"),
        help="Output Markdown report path",
    )
    args = parser.parse_args(argv)

    if not args.csv.exists():
        print(f"Aggregate CSV not found: {args.csv}", file=sys.stderr)
        return 1
    if not args.results_dir.exists():
        print(f"Results directory not found: {args.results_dir}", file=sys.stderr)
        return 1

    rows = _load_csv(args.csv)
    if not rows:
        print("No rows found in aggregate CSV", file=sys.stderr)
        return 1

    report = _generate_report(rows, args.results_dir, args.csv, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(profiling_report_cli())
