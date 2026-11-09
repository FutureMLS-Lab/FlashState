#!/usr/bin/env python3
"""Draw the README's figures from collected readings (analysis/collect.py --json).

    python3 analysis/plot.py serving ROWS.json CONFIG[,CONFIG...] OUT.png   the paper's serving metrics against sessions:
        one column per configuration, rows decode TPM, prefill TPM, uncached tokens p50 and TTFT p90 per GPU, with
        FlashStates' change against vLLM at the largest session count (the GPU-only figure)
    python3 analysis/plot.py tradeoff ROWS.json CONFIG OUT.png   input tokens/min per GPU (or the figure's LAYOUT "y"
        metric) against p50 decode speed, bubble area = p90 TTFT (the Mooncake store figure)
Needs matplotlib.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from display import METRICS, SHORT, change, title  # noqa: E402

# Engine order is the legend's; "sglang" rows come only from readings published with a figure.
STYLE = {"flashstate": ("FlashStates", "#2A78D6", "o"), "official": ("vLLM", "#E66A32", "s"),
         "sglang": ("SGLang", "#28786F", "^")}
# Trade-off label and legend placement per figure where the defaults (the GLM figure's) collide: the FlashStates peak
# label (dx, dy, ha), other FlashStates session labels by session count, vLLM's end label, the bubble legend's corner,
# and the baseline label below its line. "y" swaps the y metric (key, scale, unit, label).
INPUT_TPM = ("prompt_tpm_per_gpu", 1e6, "M", "Input tokens per minute per GPU")
OUTPUT_TPM = ("output_tpm_per_gpu", 1e3, "K", "Output tokens per minute per GPU")
LAYOUT = {"kimi-k31-pro-gb300-mooncake-dspark": {"top": (12, 0, "left"),
                                                 "fs": {24: (0, -14, "center"), 64: (-14, 0, "right")},
                                                 "vllm": (-10, -16, "right"), "keys": "upper right", "y": OUTPUT_TPM,
                                                 "best_below": True}}


def bubble(ttft_p90_s: float) -> float:
    # Marker area grows with p90 TTFT as t^0.75, as in the paper's and the website's figures, so 100-s bubbles fit.
    return 0.6 * (14 + 34 * ttft_p90_s**0.75)


def series(rows: list[dict], config: str) -> dict[str, list[dict]]:
    out = {e: sorted((r for r in rows if r["config"] == config and r["engine"] == e), key=lambda r: r["sessions"])
           for e in STYLE}
    return {e: v for e, v in out.items() if v}


def serving(rows: list[dict], configs: list[str], out: Path) -> None:
    fig, axes = plt.subplots(len(METRICS), len(configs), figsize=(3.2 * len(configs) + 1.2, 6.4), squeeze=False)
    for col, config in enumerate(configs):
        data = series(rows, config)
        for row, (key, label, unit, scale, _, better) in enumerate(METRICS):
            ax = axes[row][col]
            for engine in ("official", "flashstate"):
                name, color, marker = STYLE[engine]
                points = data.get(engine, [])
                ax.plot([p["sessions"] for p in points], [p[key] / scale for p in points], color=color, marker=marker,
                        ms=4, lw=1.4, label=name)
            common = sorted({p["sessions"] for p in data.get("official", [])}
                            & {p["sessions"] for p in data.get("flashstate", [])})
            if common:
                o, f = ({p["sessions"]: p for p in data[e]}[common[-1]] for e in ("official", "flashstate"))
                ax.annotate("", (common[-1], f[key] / scale), (common[-1], o[key] / scale), xycoords="data",
                            arrowprops={"arrowstyle": "->", "color": "black", "shrinkA": 4, "shrinkB": 4,
                                        "connectionstyle": "arc3"}, annotation_clip=False)
                ax.text(1.02, 0.5, change(o[key], f[key], better), transform=ax.transAxes, color=STYLE["flashstate"][1],
                        fontsize=9, fontweight="bold", va="center")
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:g}K" if v >= 1000 else f"{v:g}")
            ax.grid(alpha=0.3)
            ax.set_xticks(sorted({p["sessions"] for points in data.values() for p in points}))
            if row == 0:
                ax.set_title(f"({chr(97 + col)}) {SHORT.get(config, title(config))}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{label}\n({unit})" if unit else label, fontsize=9)
            if row == len(METRICS) - 1:
                ax.set_xlabel("Concurrent sessions")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=200, bbox_inches="tight")


def _site_style() -> dict:
    """The project page's look: Inter when matplotlib can find it (or FS_PLOT_FONT_DIR points at its .ttf files),
    light horizontal grid, no box, muted ink."""
    import os
    from matplotlib import font_manager as fm
    for d in filter(None, [os.environ.get("FS_PLOT_FONT_DIR")]):
        for f in Path(d).rglob("Inter-*.ttf"):
            fm.fontManager.addfont(str(f))
    names = {f.name for f in fm.fontManager.ttflist}
    family = next((n for n in ("Inter", "Helvetica Neue", "Helvetica", "Arial") if n in names), "DejaVu Sans")
    return {"font.family": family, "font.size": 10, "axes.edgecolor": "#D6D3CC", "axes.linewidth": 0.8,
            "axes.labelcolor": "#5F5E5A", "xtick.color": "#5F5E5A", "ytick.color": "#5F5E5A",
            "xtick.major.size": 0, "ytick.major.size": 0, "text.color": "#1F1F1D"}


def tradeoff(rows: list[dict], config: str, out: Path) -> None:
    ink, muted, grid = "#1F1F1D", "#6B6A66", "#E9E7E1"
    layout = LAYOUT.get(config, {})
    key, scale, unit, ylabel = layout.get("y", INPUT_TPM)
    with plt.rc_context(_site_style()):
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        data = series(rows, config)
        order = [e for e in ("official", "sglang", "flashstate") if e in data]   # FlashStates drawn on top
        for engine in order:
            label, color, _ = STYLE[engine]
            points = data[engine]
            x = [p["decode_p50_tok_s"] for p in points]
            y = [p[key] / scale for p in points]
            ax.plot(x, y, color=color, lw=2.0 if engine == "flashstate" else 1.6, zorder=3, solid_capstyle="round")
            ax.scatter(x, y, s=[bubble(p["ttft_p90_s"]) * 1.6 for p in points], facecolor=color, alpha=0.18,
                       edgecolors="none", zorder=2)
            ax.scatter(x, y, s=[bubble(p["ttft_p90_s"]) * 1.6 for p in points], facecolor="none",
                       edgecolors=color, linewidths=1.1, zorder=4)
            ax.scatter(x, y, s=9, color=color, zorder=5)

        def tag(p, text, dx, dy, color, ha="center", weight="normal"):
            ax.annotate(text, (p["decode_p50_tok_s"], p[key] / scale), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va="center", fontsize=9.5, color=color,
                        fontweight=weight, zorder=6)

        others = [p for e in order if e != "flashstate" for p in data[e]]
        best = "best of vLLM or SGLang" if "sglang" in data else "best of vLLM"
        if "flashstate" in data:
            fs = data["flashstate"]
            top = max(fs, key=lambda p: p[key])
            dx, dy, ha = layout.get("top", (-12, 0, "right"))
            tag(top, f"FlashStates · {top['sessions']} sessions", dx, dy, STYLE["flashstate"][1], ha, "semibold")
            for p in fs:
                if p is not top and p["sessions"] in (fs[0]["sessions"], 48, 64):
                    dx, dy, ha = layout.get("fs", {}).get(p["sessions"], (0, 12, "center"))
                    tag(p, str(p["sessions"]), dx, dy, muted, ha)
        peaks = {e: max(data[e], key=lambda p: p[key]) for e in order if e != "flashstate"}
        close = len(peaks) == 2 and abs(peaks["official"]["decode_p50_tok_s"] - peaks["sglang"]["decode_p50_tok_s"]) < 12
        for e, pk in peaks.items():
            name, color, _ = STYLE[e]
            if pk is not data[e][-1]:
                left = e == "official"
                tag(pk, f"peak · {pk['sessions']}", (-14 if left else 14) if close else 0, 14,
                    color, ("right" if left else "left") if close else "center", "semibold")
            last = data[e][-1]
            if e == "official":
                dx, dy, ha = layout.get("vllm", (-16, 0, "right"))
                tag(last, f"{name} · {last['sessions']} sessions", dx, dy, color, ha, "semibold")
            else:
                tag(last, f"{name} · {last['sessions']} sessions", 22, -32, color, "center", "semibold")
            tag(data[e][0], str(data[e][0]["sessions"]), 0, -14, muted)
        if "flashstate" in data and others:
            base = max(others, key=lambda p: p[key])
            y0, y1, x = base[key] / scale, top[key] / scale, top["decode_p50_tok_s"]
            ax.axhline(y0, color="#8F8D87", lw=1, ls=(0, (4, 3)), zorder=1)
            below = layout.get("best_below", False)
            ax.annotate(best, (0.995, y0), xycoords=ax.get_yaxis_transform(), xytext=(0, -3 if below else 0),
                        textcoords="offset points", ha="right", va="top" if below else "bottom", fontsize=9, color=muted)
            if y1 > y0:
                ax.annotate("", (x, y1), (x, y0), arrowprops={"arrowstyle": "-|>,head_width=0.25,head_length=0.5",
                                                              "color": ink, "lw": 1.2, "shrinkA": 0, "shrinkB": 6},
                            zorder=6)
                ax.text(x, (y0 + y1) / 2, f"  {y1 / y0 - 1:+.0%}", fontsize=11, fontweight="bold", color=ink,
                        va="center", zorder=6)

        ax.set_xlabel("Decode speed per request, p50 (tokens/s)", labelpad=8)
        ax.set_ylabel(ylabel, labelpad=8)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}{unit}")
        ax.grid(axis="y", color=grid, lw=0.9)
        ax.grid(axis="x", color=grid, lw=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        xs = [p["decode_p50_tok_s"] for e in order for p in data[e]]
        span = max(xs) - min(xs)
        ax.set_xlim(min(xs) - 0.25 * span, max(xs) + 0.1 * span)
        ax.margins(y=0.12)

        handles = [plt.Line2D([], [], color=STYLE[e][1], marker="o", ms=6, lw=2, label=STYLE[e][0])
                   for e in ("flashstate", "official", "sglang") if e in data]
        handles.append(plt.Line2D([], [], color="#8F8D87", lw=1, ls=(0, (4, 3)), label=best))
        ax.add_artist(ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=len(handles), frameon=False,
                  fontsize=9.5, handlelength=1.8, columnspacing=1.6, borderaxespad=0))
        keys = [ax.scatter([], [], s=bubble(t) * 1.6, facecolor="none", edgecolors="#8F8D87", label=f"{t} s")
                for t in (1, 5, 20, 100) if t <= 1.25 * max(p["ttft_p90_s"] for e in order for p in data[e])]
        ax.legend(handles=keys, loc=layout.get("keys", "lower left"), ncol=4, frameon=False, fontsize=8.5,
                                title="p90 time to first token", title_fontsize=9, columnspacing=1.2,
                                handletextpad=0.4, borderaxespad=0.6)
        ax.set_title(title(config), loc="left", fontsize=12, fontweight="semibold", color=ink, pad=30)
        fig.tight_layout()
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")


def main() -> None:
    kind, rows_path, config, out = sys.argv[1:5]
    rows = json.loads(Path(rows_path).read_text())
    if kind == "serving":
        serving(rows, config.split(","), Path(out))
    else:
        tradeoff(rows, config, Path(out))


if __name__ == "__main__":
    main()
