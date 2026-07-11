"""Generate a rich, self-contained HTML dashboard from toks-bench aggregate data."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
OUTPUT = RESULTS_DIR / "dashboard.html"


def _display_path(path: Path) -> str:
    """Return a path string with the user's home directory shortened to ~."""
    try:
        rel = path.relative_to(Path.home())
        return f"~/{rel.as_posix()}"
    except ValueError:
        return str(path)


def _load_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
                    row[key] = float(row[key])
                except (KeyError, ValueError):
                    row[key] = 0.0
            rows.append(row)
    return rows


def _embed_image(path: Path) -> str | None:
    if not path.exists():
        return None
    data = path.read_bytes()
    ext = path.suffix.lstrip(".")
    mime = f"image/{ext}"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _embed_csv_download(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:text/csv;base64,{b64}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _provider_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["provider"]].append(row)
    stats: dict[str, dict[str, Any]] = {}
    for provider, prov_rows in groups.items():
        mean_tok = sum(r["tok_per_sec_mean"] for r in prov_rows) / len(prov_rows)
        mean_ttft = sum(r["ttft_ms_mean"] for r in prov_rows) / len(prov_rows)
        mean_tpot = sum(r["tpot_ms_mean"] for r in prov_rows) / len(prov_rows)
        stats[provider] = {
            "prompts": len(prov_rows),
            "mean_tok_per_sec": mean_tok,
            "mean_ttft_ms": mean_ttft,
            "mean_tpot_ms": mean_tpot,
            "best_prompt": max(prov_rows, key=lambda r: r["tok_per_sec_mean"]),
        }
    return stats


def _fmt(num: float) -> str:
    return f"{num:.1f}"


def _class_for_rank(i: int) -> str:
    if i == 0:
        return "rank-gold"
    if i == 1:
        return "rank-silver"
    if i == 2:
        return "rank-bronze"
    return ""


def _read_chart_sidecar(chart_path: Path) -> dict[str, Any] | None:
    sidecar = chart_path.with_suffix("").with_suffix(chart_path.suffix + ".json")
    if not sidecar.exists():
        return None
    return json.loads(sidecar.read_text(encoding="utf-8"))


def _render_metrics_glossary() -> str:
    return """
    <section id="definitions" class="card glossary">
        <h2>Metrics Explained</h2>
        <p class="glossary-intro">
            Every number on this page is computed from client-side measurements of streaming
            chat-completion requests. Higher is better for throughput; lower is better for latency.
        </p>
        <dl class="glossary-list">
            <dt><span class="metric-name">tok/s</span> <span class="tag better-high">higher is better</span></dt>
            <dd>
                Tokens generated per second: <code>output_tokens / (total_ms / 1000)</code>.
                This is the headline throughput metric. It depends on both generation speed
                and how many tokens the model actually emits.
            </dd>

            <dt><span class="metric-name">TTFT</span> <span class="tag">time to first token</span> <span class="tag better-low">lower is better</span></dt>
            <dd>
                Milliseconds from the start of the request until the first content chunk arrives.
                Dominated by prompt prefill, queueing, and network serialization.
                Longer prompts generally increase TTFT.
            </dd>

            <dt><span class="metric-name">TPOT</span> <span class="tag">time per output token</span> <span class="tag better-low">lower is better</span></dt>
            <dd>
                Average milliseconds per generated token after the first one:
                <code>(total_ms - ttft_ms) / (output_tokens - 1)</code> for runs that
                produced more than one token. Reflects raw generation latency.
            </dd>

            <dt><span class="metric-name">Output tokens</span></dt>
            <dd>
                Mean number of completion tokens produced per run. Reported by the server
                when usage data is available, otherwise counted from streaming deltas.
            </dd>

            <dt><span class="metric-name">Runs</span></dt>
            <dd>
                Number of independent requests issued for a given provider + prompt combination.
                Means, medians, p95s, and standard deviations are computed across these runs.
            </dd>

            <dt><span class="metric-name">Finish reasons</span></dt>
            <dd>
                How the server ended each run. <code>length</code> means generation was cut off
                by <code>max_tokens</code>; <code>stop</code> means the model emitted an end-of-sequence
                token. Error entries indicate timeouts or connection failures.
            </dd>

            <dt><span class="metric-name">Mean / Median / P95 / Std</span></dt>
            <dd>
                Aggregate statistics across runs. Mean is the arithmetic average, median is the
                50th percentile, p95 is the 95th percentile, and std is the standard deviation.
            </dd>
        </dl>
    </section>
    """


def _render_provenance(rows: list[dict[str, Any]], csv_path: Path) -> str:
    total_runs = sum(int(r["runs"]) for r in rows)
    providers = sorted({r["provider"] for r in rows})
    sha256 = _sha256_file(csv_path)
    csv_b64 = _embed_csv_download(csv_path)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return f"""
    <section id="provenance" class="card provenance">
        <h2>Data Provenance</h2>
        <div class="provenance-grid">
            <div>
                <div class="prov-label">Source file</div>
                <div class="prov-value mono">{_display_path(csv_path)}</div>
            </div>
            <div>
                <div class="prov-label">Rows</div>
                <div class="prov-value">{len(rows)}</div>
            </div>
            <div>
                <div class="prov-label">Providers</div>
                <div class="prov-value">{len(providers)}</div>
            </div>
            <div>
                <div class="prov-label">Total runs</div>
                <div class="prov-value">{total_runs}</div>
            </div>
            <div>
                <div class="prov-label">Generated at</div>
                <div class="prov-value">{generated_at}</div>
            </div>
            <div class="prov-wide">
                <div class="prov-label">SHA-256 of aggregate.csv</div>
                <div class="prov-value mono">{sha256}</div>
            </div>
            <div class="prov-wide">
                <div class="prov-label">Reproduction commands</div>
                <pre class="code-block">cd {_display_path(csv_path.parent.parent)}
.venv/bin/python -m toks_bench.charts --csv results/aggregate.csv --output-dir results/charts
.venv/bin/python scripts/generate_dashboard.py</pre>
            </div>
            <div class="prov-wide">
                <a class="download-btn" href="{csv_b64}" download="aggregate.csv">
                    Download aggregate.csv
                </a>
                <p class="download-hint">
                    The raw aggregate CSV is embedded above. You can re-generate every chart from
                    this file using the commands shown.
                </p>
            </div>
        </div>
    </section>
    """


def _render_verification_table(metadata: dict[str, Any]) -> str:
    chart_type = metadata.get("chart_type")
    metric = metadata.get("metric", "")
    ylabel = metadata.get("ylabel", "")
    lower_is_better = metadata.get("lower_is_better", False)
    aggregation = metadata.get("aggregation", "mean")
    direction = "lower is better" if lower_is_better else "higher is better"

    rows_html = []
    if chart_type == "grouped_bar":
        prompts = metadata.get("prompt_order", [])
        headers = ["Provider"] + [p.title() for p in prompts]
        for prov in metadata.get("providers", []):
            cells = [f'<td class="provider">{prov["provider"]}</td>']
            for p in prompts:
                val = prov.get(p)
                cells.append(f"<td>{val if val is not None else 'N/A'}</td>")
            rows_html.append("<tr>" + "".join(cells) + "</tr>")
        thead = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    elif chart_type == "heatmap":
        prompts = metadata.get("prompt_order", [])
        headers = ["Provider"] + [p.title() for p in prompts]
        for prov in metadata.get("providers", []):
            cells = [f'<td class="provider">{prov["provider"]}</td>']
            for p in prompts:
                val = prov.get(p)
                cells.append(f"<td>{val if val is not None else 'N/A'}</td>")
            rows_html.append("<tr>" + "".join(cells) + "</tr>")
        thead = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    elif chart_type == "ranking":
        headers = ["Provider", "Mean tok/s"]
        for prov in metadata.get("providers", []):
            rows_html.append(
                f'<tr><td class="provider">{prov["provider"]}</td>'
                f'<td>{prov.get("mean_tok_per_sec", "N/A")}</td></tr>'
            )
        thead = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    else:
        return ""

    return f"""
    <details class="verify-details">
        <summary>Verify this chart</summary>
        <div class="verify-body">
            <p>
                <strong>Metric:</strong> <code>{metric}</code> ({ylabel}) — <em>{direction}</em>.<br>
                <strong>Aggregation:</strong> {aggregation}.
            </p>
            <div class="verify-table-wrapper">
                <table class="verify-table">
                    <thead>{thead}</thead>
                    <tbody>{''.join(rows_html)}</tbody>
                </table>
            </div>
        </div>
    </details>
    """


def generate_dashboard() -> None:
    rows = _load_csv(RESULTS_DIR / "aggregate.csv")
    stats = _provider_stats(rows)
    ranked = sorted(stats.items(), key=lambda kv: kv[1]["mean_tok_per_sec"], reverse=True)

    total_runs = sum(int(r["runs"]) for r in rows)
    providers_count = len(stats)
    top = ranked[0]
    bottom = ranked[-1]

    chart_files = sorted(CHARTS_DIR.glob("*.png"))
    chart_cards = []
    for chart in chart_files:
        b64 = _embed_image(chart)
        if b64 is None:
            continue
        title = chart.stem.replace("-", " ").replace("_", " ").title()
        metadata = _read_chart_sidecar(chart)
        verify_html = _render_verification_table(metadata) if metadata else ""
        chart_cards.append(
            f"""
            <div class="card chart-card" id="chart-{chart.stem}">
                <h3>{title}</h3>
                <img src="{b64}" alt="{title}" loading="lazy" />
                {verify_html}
            </div>
            """
        )

    leaderboard_rows = []
    for i, (provider, s) in enumerate(ranked):
        cls = _class_for_rank(i)
        leaderboard_rows.append(
            f"""
            <tr class="{cls}">
                <td class="rank">#{i + 1}</td>
                <td class="provider">{provider}</td>
                <td>{s['prompts']}</td>
                <td>{_fmt(s['mean_tok_per_sec'])}</td>
                <td>{_fmt(s['mean_ttft_ms'])}</td>
                <td>{_fmt(s['mean_tpot_ms'])}</td>
                <td>{s['best_prompt']['file']}</td>
            </tr>
            """
        )

    table_rows = []
    for r in rows:
        table_rows.append(
            f"""
            <tr>
                <td>{r['provider']}</td>
                <td>{Path(r['file']).stem.rsplit('-', 1)[-1]}</td>
                <td>{int(r['runs'])}</td>
                <td>{_fmt(r['tok_per_sec_mean'])}</td>
                <td>{_fmt(r['ttft_ms_mean'])}</td>
                <td>{_fmt(r['tpot_ms_mean'])}</td>
                <td>{_fmt(r['output_tokens_mean'])}</td>
                <td>{r['finish_reasons']}</td>
            </tr>
            """
        )

    glossary_html = _render_metrics_glossary()
    provenance_html = _render_provenance(rows, RESULTS_DIR / "aggregate.csv")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>toks-bench Dashboard</title>
<style>
:root {{
    --bg: #0f172a;
    --panel: #1e293b;
    --panel-2: #334155;
    --panel-3: #475569;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --accent-2: #818cf8;
    --gold: #facc15;
    --silver: #cbd5e1;
    --bronze: #fb923c;
    --danger: #f87171;
    --success: #4ade80;
    --warning: #fbbf24;
    --radius: 1rem;
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
    margin: 0;
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    color: var(--text);
    line-height: 1.6;
}}
nav {{
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(15, 23, 42, 0.92);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--panel-2);
}}
nav ul {{
    list-style: none;
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.5rem;
    margin: 0;
    padding: 0.75rem 1rem;
    max-width: 1400px;
    margin-inline: auto;
}}
nav a {{
    color: var(--muted);
    text-decoration: none;
    font-size: 0.875rem;
    padding: 0.35rem 0.75rem;
    border-radius: 9999px;
    transition: all 0.15s;
}}
nav a:hover {{
    color: var(--text);
    background: var(--panel-2);
}}
.container {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
}}
header {{
    text-align: center;
    margin-bottom: 2.5rem;
    padding-top: 1rem;
}}
header h1 {{
    font-size: 3rem;
    margin: 0;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
header p {{
    color: var(--muted);
    font-size: 1.125rem;
    margin-top: 0.5rem;
}}
.metrics {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.25rem;
    margin-bottom: 2.5rem;
}}
.metric {{
    background: var(--panel);
    border: 1px solid var(--panel-2);
    border-radius: var(--radius);
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}}
.metric .value {{
    font-size: 2.25rem;
    font-weight: 800;
    color: var(--accent);
}}
.metric .label {{
    color: var(--muted);
    font-size: 0.875rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}}
.card {{
    background: var(--panel);
    border: 1px solid var(--panel-2);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}}
.card h2, .card h3 {{
    margin-top: 0;
    color: var(--text);
}}
.leaderboard table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.95rem;
}}
.leaderboard th, .leaderboard td {{
    padding: 0.85rem 0.75rem;
    text-align: left;
    border-bottom: 1px solid var(--panel-2);
}}
.leaderboard th {{
    color: var(--muted);
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    cursor: pointer;
    user-select: none;
}}
.leaderboard th:hover {{
    color: var(--text);
}}
.leaderboard th.sort-asc::after {{ content: " ▲"; }}
.leaderboard th.sort-desc::after {{ content: " ▼"; }}
.leaderboard tr:hover {{
    background: rgba(255,255,255,0.03);
}}
.rank-gold .rank {{ color: var(--gold); font-weight: 800; }}
.rank-silver .rank {{ color: var(--silver); font-weight: 800; }}
.rank-bronze .rank {{ color: var(--bronze); font-weight: 800; }}
.provider {{ font-weight: 600; }}
.charts-grid {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 1.5rem;
}}
@media (min-width: 1200px) {{
    .charts-grid {{
        grid-template-columns: 1fr 1fr;
    }}
}}
.chart-card img {{
    width: 100%;
    height: auto;
    border-radius: 0.5rem;
    margin-top: 1rem;
    border: 1px solid var(--panel-2);
    transition: transform 0.2s ease;
    cursor: zoom-in;
}}
.chart-card img:hover {{
    transform: scale(1.02);
}}
.chart-card img.expanded {{
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: min(95vw, 1600px);
    height: auto;
    max-height: 95vh;
    object-fit: contain;
    z-index: 10000;
    cursor: zoom-out;
    box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.85);
}}
.verify-details {{
    margin-top: 1rem;
    border: 1px solid var(--panel-2);
    border-radius: 0.5rem;
    background: rgba(0,0,0,0.15);
}}
.verify-details summary {{
    padding: 0.75rem 1rem;
    cursor: pointer;
    font-weight: 600;
    color: var(--accent);
}}
.verify-body {{
    padding: 0 1rem 1rem;
}}
.verify-table-wrapper {{
    overflow-x: auto;
}}
.verify-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}}
.verify-table th, .verify-table td {{
    padding: 0.5rem;
    text-align: left;
    border-bottom: 1px solid var(--panel-2);
}}
.verify-table th {{
    color: var(--muted);
    text-transform: uppercase;
    font-size: 0.7rem;
}}
.data-table-wrapper {{
    overflow-x: auto;
}}
.data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}}
.data-table th, .data-table td {{
    padding: 0.65rem 0.5rem;
    text-align: left;
    border-bottom: 1px solid var(--panel-2);
    white-space: nowrap;
}}
.data-table th {{
    color: var(--muted);
    text-transform: uppercase;
    font-size: 0.7rem;
    letter-spacing: 0.05em;
}}
.data-table tr:hover {{ background: rgba(255,255,255,0.03); }}
.glossary-intro {{
    color: var(--muted);
    margin-top: -0.5rem;
}}
.glossary-list {{
    display: grid;
    gap: 1rem;
}}
.glossary-list dt {{
    font-weight: 700;
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.glossary-list dd {{
    margin: 0;
    color: var(--muted);
    padding-left: 0.5rem;
    border-left: 2px solid var(--panel-2);
}}
.metric-name {{
    color: var(--text);
    font-size: 1.05rem;
}}
.tag {{
    font-size: 0.7rem;
    padding: 0.15rem 0.5rem;
    border-radius: 9999px;
    background: var(--panel-2);
    color: var(--muted);
    font-weight: 600;
    text-transform: uppercase;
}}
.better-high {{
    background: rgba(74, 222, 128, 0.15);
    color: var(--success);
}}
.better-low {{
    background: rgba(248, 113, 113, 0.15);
    color: var(--danger);
}}
.provenance-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.25rem;
}}
.prov-wide {{
    grid-column: 1 / -1;
}}
.prov-label {{
    color: var(--muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}}
.prov-value {{
    font-weight: 600;
    word-break: break-word;
}}
.mono {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.85rem;
}}
.code-block {{
    background: #0b1120;
    border: 1px solid var(--panel-2);
    border-radius: 0.5rem;
    padding: 1rem;
    overflow-x: auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.85rem;
    margin: 0;
}}
.download-btn {{
    display: inline-block;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    color: #0f172a;
    font-weight: 700;
    padding: 0.75rem 1.25rem;
    border-radius: 0.5rem;
    text-decoration: none;
    margin-bottom: 0.5rem;
}}
.download-hint {{
    color: var(--muted);
    font-size: 0.9rem;
    margin: 0;
}}
.notes {{
    color: var(--muted);
    font-size: 0.95rem;
}}
.notes ul {{
    padding-left: 1.25rem;
}}
.notes li {{
    margin-bottom: 0.5rem;
}}
.back-to-top {{
    text-align: center;
    margin-top: 2rem;
}}
.back-to-top a {{
    color: var(--accent);
    text-decoration: none;
}}
footer {{
    text-align: center;
    color: var(--muted);
    font-size: 0.875rem;
    margin-top: 3rem;
}}
code {{
    background: rgba(0,0,0,0.25);
    padding: 0.15rem 0.35rem;
    border-radius: 0.25rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.9em;
}}
</style>
</head>
<body>
<nav>
    <ul>
        <li><a href="#overview">Overview</a></li>
        <li><a href="#leaderboard">Leaderboard</a></li>
        <li><a href="#charts">Charts</a></li>
        <li><a href="#definitions">Definitions</a></li>
        <li><a href="#raw-data">Raw Data</a></li>
        <li><a href="#provenance">Provenance</a></li>
        <li><a href="#notes">Notes</a></li>
    </ul>
</nav>

<div class="container">
    <header id="overview">
        <h1>toks-bench Dashboard</h1>
        <p>Token-throughput benchmark results across local OpenAI-compatible inference servers</p>
    </header>

    <section class="metrics">
        <div class="metric">
            <div class="value">{providers_count}</div>
            <div class="label">Providers</div>
        </div>
        <div class="metric">
            <div class="value">{total_runs}</div>
            <div class="label">Total runs</div>
        </div>
        <div class="metric">
            <div class="value">{_fmt(top[1]['mean_tok_per_sec'])}</div>
            <div class="label">Best tok/s<br><small>{top[0]}</small></div>
        </div>
        <div class="metric">
            <div class="value">{_fmt(bottom[1]['mean_tok_per_sec'])}</div>
            <div class="label">Lowest tok/s<br><small>{bottom[0]}</small></div>
        </div>
    </section>

    {glossary_html}

    <section id="leaderboard" class="card leaderboard">
        <h2>Provider Leaderboard</h2>
        <p style="color: var(--muted); margin-top: -0.75rem; margin-bottom: 1rem;">
            Ranked by mean tok/s across all prompts. Click column headers to sort. Lower TTFT/TPOT is better.
        </p>
        <table id="leaderboard-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Provider</th>
                    <th>Prompts</th>
                    <th>Mean tok/s</th>
                    <th>Mean TTFT (ms)</th>
                    <th>Mean TPOT (ms)</th>
                    <th>Best run</th>
                </tr>
            </thead>
            <tbody>
                {''.join(leaderboard_rows)}
            </tbody>
        </table>
    </section>

    <section id="charts" class="card">
        <h2>Charts</h2>
        <p style="color: var(--muted); margin-top: -0.75rem; margin-bottom: 1rem;">
            Each chart has an expandable "Verify this chart" section showing the exact data used to render it.
        </p>
        <div class="charts-grid">
            {''.join(chart_cards)}
        </div>
    </section>

    <section id="raw-data" class="card data-table-wrapper">
        <h2>Detailed Results</h2>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Provider</th>
                    <th>Prompt</th>
                    <th>Runs</th>
                    <th>tok/s mean</th>
                    <th>TTFT mean</th>
                    <th>TPOT mean</th>
                    <th>Out tokens</th>
                    <th>Finish reasons</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
    </section>

    {provenance_html}

    <section id="notes" class="card notes">
        <h2>Notes</h2>
        <ul>
            <li><strong>Low-bit models:</strong> five 2026 models were added; the 1.58-bit ternary model (prism-ml/Ternary-Bonsai-8B-Q2_0) failed to load because upstream llama.cpp does not recognize its GGML tensor type.</li>
            <li><strong>TensorRT-LLM:</strong> the published <code>tensorrt_llm-1.2.1</code> aarch64 wheel has an ABI mismatch with every publicly installable PyTorch wheel for CUDA 13.0, so it could not be launched. A vLLM benchmark of the same Qwen3-4B checkpoint was used as the GPU-serving comparison.</li>
            <li><strong>Ollama runs:</strong> Ollama was CPU-bound during these sweeps because the single GB10 GPU was occupied by other servers, producing low throughput and timeouts.</li>
        </ul>
    </section>

    <div class="back-to-top">
        <a href="#overview">Back to top ↑</a>
    </div>

    <footer>
        © 2026 VibeCodingAgency.com · Generated by toks-bench · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
    </footer>
</div>

<script>
(function() {{
    const table = document.getElementById('leaderboard-table');
    if (!table) return;
    const headers = table.querySelectorAll('thead th');
    const tbody = table.querySelector('tbody');
    let sortCol = -1;
    let sortAsc = true;

    function parseValue(cell) {{
        const text = cell.textContent.trim();
        const num = parseFloat(text);
        return isNaN(num) ? text.toLowerCase() : num;
    }}

    headers.forEach((header, index) => {{
        header.addEventListener('click', () => {{
            const rows = Array.from(tbody.querySelectorAll('tr'));
            if (sortCol === index) {{
                sortAsc = !sortAsc;
            }} else {{
                sortCol = index;
                sortAsc = true;
            }}
            headers.forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
            header.classList.add(sortAsc ? 'sort-asc' : 'sort-desc');

            rows.sort((a, b) => {{
                const av = parseValue(a.children[index]);
                const bv = parseValue(b.children[index]);
                if (typeof av === 'number' && typeof bv === 'number') {{
                    return sortAsc ? av - bv : bv - av;
                }}
                return sortAsc ? (av > bv ? 1 : -1) : (av > bv ? -1 : 1);
            }});

            rows.forEach(row => tbody.appendChild(row));
        }});
    }});

    // Click any chart image to expand it to full-screen; click again to collapse.
    document.querySelectorAll('.chart-card img').forEach(img => {{
        img.addEventListener('click', () => {{
            if (img.classList.contains('expanded')) {{
                img.classList.remove('expanded');
            }} else {{
                document.querySelectorAll('.chart-card img.expanded').forEach(i => i.classList.remove('expanded'));
                img.classList.add('expanded');
            }}
        }});
    }});
    document.addEventListener('click', (e) => {{
        if (!e.target.closest('.chart-card img')) {{
            document.querySelectorAll('.chart-card img.expanded').forEach(i => i.classList.remove('expanded'));
        }}
    }});
}})();
</script>
</body>
</html>
"""

    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    generate_dashboard()
