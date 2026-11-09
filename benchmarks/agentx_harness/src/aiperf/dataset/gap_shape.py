# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reshape the idle gaps of a trace replay: the same mean, another distribution.

A cache policy that protects recently used prefixes bets that a session idle
for a short time comes back soon. The recorded agent traces make that bet
easy to win. ``weka_gap_shape`` replaces every recorded inter-turn delay
(``Turn.delay`` of turns after the first) with a draw from another
distribution, keeping the mean over all replaced delays and the delay cap, so
the offered load stays the same and only the predictability of returns
changes.

Shapes:

- ``exp``: exponential, memoryless (idle age says nothing about the return).
- ``lognorm[:SIGMA]``: lognormal, heavy-tailed (default ``SIGMA`` 2.0).
- ``mix[:SHARE[:RATIO]]``: two exponentials, a ``SHARE`` of the gaps (default
  0.1) with ``RATIO`` times the mean of the rest (default 30): mostly seconds,
  a few minutes.
- ``shuffle``: the recorded gaps, permuted across all conversations.

Draws are seeded by the dataset's ``random_seed`` and the shape. The scale is
set so that the mean after the cap equals the recorded mean.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from aiperf.common.models import Conversation

GAP_SHAPES = ("exp", "lognorm", "mix", "shuffle")


@dataclass(frozen=True, slots=True)
class GapShape:
    """A parsed ``weka_gap_shape`` value."""

    kind: str
    """One of ``GAP_SHAPES``."""
    sigma: float = 2.0
    """Lognormal shape parameter (``lognorm`` only)."""
    share: float = 0.1
    """Share of long gaps (``mix`` only)."""
    ratio: float = 30.0
    """Mean of the long gaps over the mean of the short ones (``mix`` only)."""


def parse_gap_shape(spec: str | None) -> GapShape | None:
    """Parse ``weka_gap_shape``; None keeps the recorded gaps.

    Example: ``"lognorm:1.5"`` gives ``GapShape("lognorm", sigma=1.5)``;
    ``"mix:0.2:10"`` gives ``GapShape("mix", share=0.2, ratio=10.0)``;
    ``"pareto"`` raises ValueError.
    """
    if spec is None:
        return None
    kind, *params = spec.split(":")
    limits = {"exp": 0, "shuffle": 0, "lognorm": 1, "mix": 2}
    if kind not in GAP_SHAPES or len(params) > limits[kind]:
        raise ValueError(
            "weka_gap_shape must be exp, lognorm[:SIGMA], mix[:SHARE[:RATIO]] "
            f"or shuffle, got {spec!r}"
        )
    try:
        values = [float(p) for p in params]
    except ValueError:
        raise ValueError(
            f"weka_gap_shape parameters must be numbers, got {spec!r}"
        ) from None
    if kind == "lognorm":
        shape = GapShape(kind, sigma=values[0]) if values else GapShape(kind)
        if not 0.0 < shape.sigma <= 5.0:
            raise ValueError(
                f"weka_gap_shape lognorm SIGMA must be in (0, 5], got {spec!r}"
            )
        return shape
    if kind == "mix":
        shape = GapShape(
            kind,
            share=values[0] if values else 0.1,
            ratio=values[1] if len(values) > 1 else 30.0,
        )
        if not 0.0 < shape.share < 1.0 or shape.ratio <= 1.0:
            raise ValueError(
                f"weka_gap_shape mix needs SHARE in (0, 1) and RATIO > 1, got {spec!r}"
            )
        return shape
    return GapShape(kind)


def _unit_draws(shape: GapShape, n: int, rng: random.Random) -> list[float]:
    """``n`` draws with mean 1 in expectation."""
    if shape.kind == "exp":
        return [rng.expovariate(1.0) for _ in range(n)]
    if shape.kind == "lognorm":
        mu = -shape.sigma**2 / 2.0
        return [rng.lognormvariate(mu, shape.sigma) for _ in range(n)]
    short = 1.0 / (1.0 - shape.share + shape.share * shape.ratio)
    return [
        rng.expovariate(1.0 / (short * shape.ratio))
        if rng.random() < shape.share
        else rng.expovariate(1.0 / short)
        for _ in range(n)
    ]


def _capped_scale(draws: list[float], target: float, cap: float | None) -> float:
    """The factor ``s`` with ``mean(min(s * x, cap)) == target``.

    Found by bisection: the capped mean grows with ``s``.
    """
    mean = sum(draws) / len(draws)
    if cap is None or mean <= 0.0:
        return target / mean if mean > 0.0 else 0.0

    def capped_mean(s: float) -> float:
        return sum(min(s * x, cap) for x in draws) / len(draws)

    lo, hi = 0.0, target / mean
    while capped_mean(hi) < target and hi < 1e12:
        hi *= 2.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if capped_mean(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def gap_stats(gaps: list[float]) -> dict[str, float]:
    """Count, mean, p50, p90, p99 (ms) and coefficient of variation of ``gaps``."""
    if not gaps:
        return {"n": 0}
    ordered = sorted(gaps)
    n = len(ordered)
    mean = sum(ordered) / n
    var = sum((g - mean) ** 2 for g in ordered) / n

    def pct(q: float) -> float:
        return ordered[min(n - 1, int(q * n))]

    return {
        "n": n,
        "mean": mean,
        "p50": pct(0.5),
        "p90": pct(0.9),
        "p99": pct(0.99),
        "cv": math.sqrt(var) / mean if mean > 0.0 else 0.0,
    }


def reshape_gaps(
    conversations: list[Conversation],
    *,
    shape: GapShape,
    seed: int,
    cap_ms: float | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Replace the delays of all turns after the first, in place.

    Returns the gap stats before and after.

    The mean over the replaced delays stays the same, and no new delay exceeds
    ``cap_ms``. Turns without a delay keep none.
    """
    turns = [t for c in conversations for t in c.turns[1:] if t.delay is not None]
    gaps = [float(t.delay) for t in turns]
    before = gap_stats(gaps)
    if not gaps:
        return before, before
    rng = random.Random(f"{seed}:gap-shape:{shape.kind}")
    if shape.kind == "shuffle":
        new = list(gaps)
        rng.shuffle(new)
    else:
        draws = _unit_draws(shape, len(gaps), rng)
        scale = _capped_scale(draws, before["mean"], cap_ms)
        new = [
            min(scale * x, cap_ms) if cap_ms is not None else scale * x for x in draws
        ]
    for turn, gap in zip(turns, new, strict=True):
        turn.delay = gap
    return before, gap_stats(new)
