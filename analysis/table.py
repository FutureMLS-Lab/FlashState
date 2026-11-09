#!/usr/bin/env python3
"""Results tables in Markdown from collected readings (analysis/collect.py --json).

    python3 analysis/table.py ROWS.json summary CONFIG...   FlashStates' change against vLLM: one row per configuration
                                                             and session count that both engines ran (the README's table)
    python3 analysis/table.py ROWS.json values CONFIG       both engines' readings, one row per session count and engine

Decode TPM per GPU counts output tokens per minute per GPU. Prefill TPM per GPU counts the prompt tokens (cached and
computed) of the requests whose first token arrives in the window, per minute per GPU. Uncached tokens p50 is the
median of the prompt tokens a request computes rather than reads from the cache. A throughput changes as a ratio (x),
uncached tokens and p90 TTFT as a percentage.
"""
import json
import sys
from pathlib import Path

from display import METRICS, change, title


def paired(rows: list[dict], config: str) -> dict[int, tuple[dict, dict]]:
    by = {(r["engine"], r["sessions"]): r for r in rows if r["config"] == config}
    return {s: (by["official", s], by["flashstate", s])
            for e, s in sorted(by) if e == "official" and ("flashstate", s) in by}


def summary(rows: list[dict], configs: list[str]) -> str:
    out = ["| Model and setting | Sessions | " + " | ".join(label for _, label, *_ in METRICS) + " |",
           "| --- | ---: |" + " ---: |" * len(METRICS)]
    for config in configs:
        for i, (s, (o, f)) in enumerate(paired(rows, config).items()):
            cells = [change(o[k], f[k], better) for k, *_, better in METRICS]
            out.append(f"| {title(config) if i == 0 else ''} | {s} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def values(rows: list[dict], config: str) -> str:
    heads = [f"{label} ({unit})" if unit else label for _, label, unit, *_ in METRICS]
    out = ["| Sessions | Engine | " + " | ".join(heads) + " | Cache hit rate |",
           "| ---: | --- |" + " ---: |" * (len(METRICS) + 1)]
    for s, (o, f) in paired(rows, config).items():
        for name, r in (("vLLM", o), ("FlashStates", f)):
            cells = []
            for k, _, _, scale, fmt, better in METRICS:
                cell = format(r[k] / scale, fmt)
                cells.append(cell if r is o else f"{cell} ({change(o[k], f[k], better)})")
            out.append(f"| {s} | {name} | " + " | ".join(cells) + f" | {r['hit_rate']:.3f} |")
    return "\n".join(out)


def main() -> None:
    rows_path, kind, *configs = sys.argv[1:]
    rows = json.loads(Path(rows_path).read_text())
    print(summary(rows, configs) if kind == "summary" else values(rows, configs[0]))


if __name__ == "__main__":
    main()
