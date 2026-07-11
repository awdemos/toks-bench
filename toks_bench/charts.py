"""Generate comparison charts from toks-bench aggregate data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd

_PROMPT_ORDER = ["short", "medium", "long", "code", "tool"]


def _load_csv(path: Path) -> pd.DataFrame:
    """Load aggregate CSV into a DataFrame with consistent types."""
    df = pd.read_csv(path)
    numeric_cols = [
        "output_tokens_mean",
        "output_tokens_p95",
        "runs",
        "tok_per_sec_mean",
        "tok_per_sec_median",
        "tok_per_sec_p95",
        "tok_per_sec_std",
        "tpot_ms_mean",
        "tpot_ms_p95",
        "ttft_ms_mean",
        "ttft_ms_p95",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["prompt"] = df["file"].apply(
        lambda x: str(x).replace(".json", "").rsplit("-", 1)[-1]
    )
    df["provider"] = df["provider"].astype(str)
    return df


def _write_sidecar(output_path: Path, metadata: dict[str, Any]) -> None:
    """Write a JSON sidecar describing the chart source data."""
    sidecar_path = output_path.with_suffix("").with_suffix(output_path.suffix + ".json")
    sidecar_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _plot_grouped_bar(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    lower_is_better: bool = False,
) -> dict[str, Any]:
    """Create a grouped bar chart of metric by provider and prompt."""
    pivot = df.pivot_table(
        index="provider",
        columns="prompt",
        values=metric,
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=[p for p in _PROMPT_ORDER if p in pivot.columns])

    # Sort providers so the best values appear at the top of the horizontal bars.
    # matplotlib's barh plots the first DataFrame row at the bottom, so we reverse
    # the sorted order so the best provider is last (top of the chart).
    provider_order = pivot.mean(axis=1).sort_values(ascending=not lower_is_better).index[::-1]
    pivot = pivot.loc[provider_order]

    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns) * 1.2), max(6, len(pivot.index) * 0.4))
    )
    pivot.plot(kind="barh", ax=ax, width=0.8)
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.set_ylabel("Provider")
    ax.legend(title="Prompt", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    if lower_is_better:
        ax.invert_xaxis()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    metadata: dict[str, Any] = {
        "chart_type": "grouped_bar",
        "title": title,
        "metric": metric,
        "ylabel": ylabel,
        "lower_is_better": lower_is_better,
        "aggregation": "mean",
        "prompt_order": list(pivot.columns),
        "providers": [],
    }
    for provider in pivot.index:
        row: dict[str, Any] = {"provider": provider}
        for prompt in pivot.columns:
            value = pivot.loc[provider, prompt]
            row[prompt] = round(float(value), 2) if pd.notna(value) else None
        metadata["providers"].append(row)

    _write_sidecar(output_path, metadata)
    return metadata


def _plot_heatmap(df: pd.DataFrame, metric: str, title: str, output_path: Path) -> dict[str, Any]:
    """Create a heatmap of metric by provider (rows) and prompt (columns)."""
    pivot = df.pivot_table(
        index="provider",
        columns="prompt",
        values=metric,
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=[p for p in _PROMPT_ORDER if p in pivot.columns])

    # Sort providers so the highest mean values appear at the top.
    provider_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[provider_order]

    fig, ax = plt.subplots(
        figsize=(max(8, len(pivot.columns) * 1.2), max(6, len(pivot.index) * 0.5))
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            text = f"{value:.1f}" if pd.notna(value) else "N/A"
            ax.text(j, i, text, ha="center", va="center", color="white", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    metadata: dict[str, Any] = {
        "chart_type": "heatmap",
        "title": title,
        "metric": metric,
        "lower_is_better": False,
        "aggregation": "mean",
        "prompt_order": list(pivot.columns),
        "providers": [],
    }
    for provider in pivot.index:
        row: dict[str, Any] = {"provider": provider}
        for prompt in pivot.columns:
            value = pivot.loc[provider, prompt]
            row[prompt] = round(float(value), 2) if pd.notna(value) else None
        metadata["providers"].append(row)

    _write_sidecar(output_path, metadata)
    return metadata


def _plot_overall_ranking(df: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Create a horizontal bar chart of overall mean tok/s per provider."""
    ranking = cast(pd.Series, df.groupby("provider")["tok_per_sec_mean"].mean())
    ranking = ranking.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(ranking) * 0.4)))
    ranking.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Overall mean throughput by provider")
    ax.set_xlabel("Mean tok/s")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    for i, v in enumerate(ranking):
        ax.text(v, i, f" {v:.1f}", va="center")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    metadata: dict[str, Any] = {
        "chart_type": "ranking",
        "title": "Overall mean throughput by provider",
        "metric": "tok_per_sec_mean",
        "ylabel": "Mean tok/s (higher is better)",
        "lower_is_better": False,
        "aggregation": "mean of per-prompt tok_per_sec_mean",
        "providers": [
            {"provider": provider, "mean_tok_per_sec": round(float(value), 2)}
            for provider, value in ranking.items()
        ],
    }

    _write_sidecar(output_path, metadata)
    return metadata


def generate_charts(csv_path: Path, output_dir: Path) -> list[Path]:
    """Generate all comparison charts and return the list of output paths."""
    df = _load_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

    # Throughput
    p = output_dir / "tok-per-sec-by-prompt.png"
    _plot_grouped_bar(
        df,
        "tok_per_sec_mean",
        "Mean throughput (tok/s) by provider and prompt",
        "tok/s (higher is better)",
        p,
    )
    paths.append(p)

    p = output_dir / "tok-per-sec-ranking.png"
    _plot_overall_ranking(df, p)
    paths.append(p)

    p = output_dir / "tok-per-sec-heatmap.png"
    _plot_heatmap(df, "tok_per_sec_mean", "Mean tok/s heatmap", p)
    paths.append(p)

    # TTFT
    p = output_dir / "ttft-by-prompt.png"
    _plot_grouped_bar(
        df,
        "ttft_ms_mean",
        "Mean time-to-first-token (TTFT) by provider and prompt",
        "TTFT ms (lower is better)",
        p,
        lower_is_better=True,
    )
    paths.append(p)

    # TPOT
    p = output_dir / "tpot-by-prompt.png"
    _plot_grouped_bar(
        df,
        "tpot_ms_mean",
        "Mean time-per-output-token (TPOT) by provider and prompt",
        "TPOT ms/token (lower is better)",
        p,
        lower_is_better=True,
    )
    paths.append(p)

    return paths


def charts_cli(argv: list[str] | None = None) -> int:
    """CLI entry point for chart generation."""
    parser = argparse.ArgumentParser(
        description="Generate comparison charts from toks-bench aggregate CSV",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/aggregate.csv"),
        help="Path to aggregate CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/charts"),
        help="Directory for output charts",
    )
    args = parser.parse_args(argv)

    if not args.csv.exists():
        print(f"Aggregate CSV not found: {args.csv}", file=sys.stderr)
        return 1

    paths = generate_charts(args.csv, args.output_dir)
    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(charts_cli())
