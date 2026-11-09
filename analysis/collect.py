#!/usr/bin/env python3
"""Collect the readings of every run under a results directory into one table.

    python3 analysis/collect.py RESULTS_DIR [--json OUT]

Runs are found as RESULTS_DIR/<configuration>/<engine>/<sessions>/readings.json, the layout the README's commands
write. Prints one row per run, sorted by configuration, engine and sessions.
"""
import argparse
import json
from pathlib import Path

COLUMNS = ("output_tpm", "prompt_tpm_per_gpu", "ttft_p50_s", "ttft_p90_s", "decode_p50_tok_s", "hit_rate", "errors")


def collect(results: Path) -> list[dict]:
    rows = []
    for path in results.glob("*/*/*/readings.json"):
        config, engine, sessions = path.parts[-4:-1]
        if not sessions.isdigit():
            continue
        rows.append({"config": config, "engine": engine, "sessions": int(sessions), **json.loads(path.read_text())})
    return sorted(rows, key=lambda r: (r["config"], r["engine"], r["sessions"]))


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}" if value < 10 else f"{value:,.0f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--json", type=Path, help="also write the rows to this file")
    a = parser.parse_args()
    rows = collect(a.results)
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1) + "\n")
    header = ("config", "engine", "sessions", *COLUMNS)
    print("\t".join(header))
    for r in rows:
        print("\t".join(fmt(r.get(c)) for c in header))


if __name__ == "__main__":
    main()
