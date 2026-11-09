#!/usr/bin/env python3
"""Readings of one replay run from AIPerf's per-request export.

    python3 analysis/readings.py ARTIFACT_DIR [--gpus 2] [--window 900 2700]

t0 is the send time of the first profiling request; the window is [t0 + LO, t0 + HI).

- output_tpm: output tokens arriving in the window x 60 / (HI - LO); a request's completion tokens are spread evenly
  over its stream chunks (AIPerf records chunk arrivals, not token counts per chunk).
- prompt_tpm (prefill throughput): prompt tokens (cached and computed) of the requests whose first token arrives in
  the window, per minute.
- TTFT p50/p90, hit rate (cached / prompt tokens), uncached tokens p50 (the median of the prompt tokens a request
  computes rather than reads from the cache) and p50 decode speed ((completion - 1) / (end-to-end - TTFT)) of the
  profiling requests sent in the window.
- errors: profiling requests that failed and were not cancelled at the end of the run.
Per-GPU values divide by --gpus (the GPUs of one server).
"""
import argparse
import json
import statistics
from pathlib import Path

UNIT_S = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0, "sec": 1.0}


def seconds(node):
    if not isinstance(node, dict) or node.get("value") is None:
        return None
    scale, value = UNIT_S[str(node.get("unit", "ms")).lower()], node["value"]
    return [float(v) * scale for v in value] if isinstance(value, list) else float(value) * scale


def count(node):
    return int(node["value"]) if isinstance(node, dict) and node.get("value") is not None else None


def percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def profiling(record):
    return str(record["metadata"].get("benchmark_phase", "")).lower().rsplit(".", 1)[-1] == "profiling"


def readings(artifact_dir: Path, gpus: int, lo: float, hi: float) -> dict:
    path = next(artifact_dir.rglob("profile_export.jsonl"))
    records = [json.loads(line) for line in path.open() if line.strip()]
    records = [r for r in records if (r.get("metadata") or {}).get("request_start_ns") is not None]
    t0 = min(r["metadata"]["request_start_ns"] for r in records if profiling(r)) / 1e9
    tokens, prefill, window = 0.0, 0, []
    for r in records:
        meta, metrics = r["metadata"], r.get("metrics") or {}
        sent = meta["request_start_ns"] / 1e9
        ttft = seconds(metrics.get("time_to_first_token"))
        total = count(metrics.get("usage_completion_tokens")) or count(metrics.get("output_token_count"))
        if ttft is not None and lo <= sent + ttft - t0 < hi:
            prefill += count(metrics.get("usage_prompt_tokens")) or 0
        if ttft is not None and total:
            arrivals = [sent + ttft]
            for gap in seconds(metrics.get("inter_chunk_latency")) or []:
                arrivals.append(arrivals[-1] + gap)
            tokens += total / len(arrivals) * sum(lo <= int(a - t0) < hi for a in arrivals)
        if profiling(r) and lo <= sent - t0 < hi:
            e2e = seconds(metrics.get("request_latency"))
            decode = None
            if ttft is not None and e2e is not None and total and total > 1 and e2e > ttft:
                decode = (total - 1) / (e2e - ttft)
            window.append({"ttft": ttft, "decode": decode,
                           "prompt": count(metrics.get("usage_prompt_tokens")) or 0,
                           "cached": count(metrics.get("usage_prompt_cache_read_tokens")) or 0})
    ttfts = [w["ttft"] for w in window if w["ttft"] is not None]
    decodes = [w["decode"] for w in window if w["decode"] is not None]
    prompt = sum(w["prompt"] for w in window)
    minutes = (hi - lo) / 60
    errors = sum(1 for r in records if profiling(r) and r.get("error") and not r["metadata"].get("was_cancelled"))
    return {
        "output_tpm": round(tokens / minutes), "output_tpm_per_gpu": round(tokens / minutes / gpus),
        "prompt_tpm": round(prefill / minutes), "prompt_tpm_per_gpu": round(prefill / minutes / gpus),
        "ttft_p50_s": percentile(ttfts, 0.5), "ttft_p90_s": percentile(ttfts, 0.9),
        "decode_p50_tok_s": statistics.median(decodes) if decodes else None,
        "hit_rate": sum(w["cached"] for w in window) / prompt if prompt else None,
        "uncached_p50": statistics.median(w["prompt"] - w["cached"] for w in window) if window else None,
        "requests_in_window": len(window), "errors": errors, "window_s": [lo, hi], "gpus": gpus,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--window", type=float, nargs=2, default=(900.0, 2700.0), metavar=("LO", "HI"))
    a = parser.parse_args()
    print(json.dumps(readings(a.artifact_dir, a.gpus, *a.window), indent=1))


if __name__ == "__main__":
    main()
