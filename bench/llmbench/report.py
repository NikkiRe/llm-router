"""Summaries, tables and the chart for a benchmark run."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from .client import Result

# Categorical slots 1-3 of the reference palette: validated for colour-vision
# deficiency across all pairs. Colour follows the policy, not its rank.
COLORS = {"round_robin": "#2a78d6", "least_loaded": "#eb6834", "prefix": "#1baf7a"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def percentile(values: list[float], p: float) -> float | None:
    """Linear interpolation between closest ranks; values must be sorted."""
    if not values:
        return None
    k = (len(values) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def summarize(results: list[Result], wall: float) -> dict:
    ok = [r for r in results if r.status == 200 and r.latency is not None]
    ttft = sorted(r.ttft for r in ok if r.ttft is not None)
    latency = sorted(r.latency for r in ok)
    prompt = sum(r.prompt_tokens for r in ok)
    cached = sum(r.cached_tokens for r in ok)
    return {
        "requests": len(results),
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "ttft_p50": percentile(ttft, 50),
        "ttft_p90": percentile(ttft, 90),
        "ttft_p99": percentile(ttft, 99),
        "ttft_mean": sum(ttft) / len(ttft) if ttft else None,
        "latency_p50": percentile(latency, 50),
        "latency_p99": percentile(latency, 99),
        "throughput": len(ok) / wall if wall else 0,
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "cache_hit": cached / prompt if prompt else 0,
        "per_backend": dict(sorted(Counter(r.backend for r in ok).items())),
        "ttft_sorted": ttft,
    }


def _s(v: float | None) -> str:
    if v is None:
        return "–"
    return f"{v * 1000:.0f} ms" if v < 1 else f"{v:.2f} s"


def one_line(s: dict) -> str:
    return (
        f"ok {s['ok']}/{s['requests']}, TTFT p50 {_s(s['ttft_p50'])} p90 {_s(s['ttft_p90'])} "
        f"p99 {_s(s['ttft_p99'])}, cache hit {s['cache_hit']:.0%}, {s['throughput']:.2f} req/s"
    )


def markdown(summaries: dict[str, dict], title: str) -> str:
    rows = [
        f"### {title}",
        "",
        "| Policy | TTFT p50 | TTFT p90 | TTFT p99 | Latency p50 | Prompt tokens from cache "
        "| Errors | Requests per backend |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for policy, s in summaries.items():
        spread = ", ".join(str(n) for n in s["per_backend"].values())
        rows.append(
            f"| {policy} | {_s(s['ttft_p50'])} | {_s(s['ttft_p90'])} | {_s(s['ttft_p99'])} "
            f"| {_s(s['latency_p50'])} | {s['cache_hit']:.0%} | {s['errors']} | {spread} |"
        )
    return "\n".join(rows) + "\n"


def chart(summaries: dict[str, dict], path: Path, title: str) -> None:
    """TTFT distribution per policy as a CDF: the share of requests that got
    their first token within x seconds. Further left is better."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for policy, s in summaries.items():
        xs = s["ttft_sorted"]
        if not xs:
            continue
        ys = [(i + 1) / len(xs) * 100 for i in range(len(xs))]
        # The legend carries the numbers; medians of close policies would
        # collide as labels on the plot.
        label = f"{policy}: p50 {_s(s['ttft_p50'])}, p90 {_s(s['ttft_p90'])}"
        ax.step(xs, ys, where="post", color=COLORS.get(policy, INK_2), linewidth=2, label=label)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v * 1000:g} ms" if v < 1 else f"{v:g} s")
    )
    ax.set_ylim(0, 101)
    ax.set_xlabel("Time to first token (log scale)", color=INK_2, fontsize=9)
    ax.set_ylabel("Requests, %", color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    legend = ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
