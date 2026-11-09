# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live-assistant replay for delta-encoded weka traces.

With ``weka_live_assistant`` on, the weka reconstructor tags every synthesized
assistant message with the index of the request whose output it stands for.
The worker records each request's actual model output and, when building a
later request, swaps the recorded synthetic assistant text for that live
output. The server then sees its own generated tokens again on the next turn,
so decode-produced KV can be reused the way a real agent loop reuses it.

Prefix structure is unchanged: everything outside assistant messages is the
recorded hash-block content, and a reset turn still re-emits the kept prefix
(now carrying live outputs) followed by fresh never-seen blocks.

With ``weka_token_ids`` the messages carry ``token_ids`` instead of text and the
live output is the exact token-ID list the server streamed back
(``return_token_ids``), so the next prompt continues the server's own token
sequence with no detokenize/retokenize step.

With ``weka_user_turns`` every turn is labelled with the user turn it belongs
to, inferred from recorded timing (:func:`label_user_turns`), and the labels
ride to vLLM as ``vllm_xargs``. ``weka_clear_thinking`` adds the context edit of
chat templates that drop the reasoning of assistant messages before the last
user message (GLM's ``clear_thinking``): once a later user turn opens, an
output's reasoning prefix leaves every later prompt.
"""

from __future__ import annotations

import hashlib
import random
from bisect import bisect_left, bisect_right
from typing import Any

from aiperf.common.models.dataset_models import Conversation
from aiperf.common.models.record_models import ParsedResponse, ReasoningResponseData

TOKEN_IDS_KEY = "token_ids"
"""Message key holding a token-ID segment (token-ID replay)."""

LIVE_SOURCE_TURN_KEY = "aiperf_live_source_turn"
"""Assistant-message key: turn index whose live output replaces this message."""

LIVE_KEEP_FRACTION_KEY = "aiperf_live_keep_fraction"
"""Assistant-message key: fraction of the output kept after a context cut (absent = 1)."""

LIVE_CLEAR_AT_TURN_KEY = "aiperf_live_clear_at_turn"
"""Assistant-message key: first turn whose prompt drops this output's reasoning."""

LIVE_REASONING_SHARE_KEY = "aiperf_live_reasoning_share"
"""Assistant-message key: fraction of this output, from its start, that is reasoning."""

LIVE_CLEAR_SPANS_KEY = "aiperf_live_clear_spans"
"""Assistant-message key: ``[first, end)`` turn ranges whose prompts drop this output's reasoning."""

_LIVE_KEYS = (
    LIVE_SOURCE_TURN_KEY,
    LIVE_KEEP_FRACTION_KEY,
    LIVE_CLEAR_AT_TURN_KEY,
    LIVE_REASONING_SHARE_KEY,
    LIVE_CLEAR_SPANS_KEY,
)

CLEAR_MIX_SWITCH = "switch"
"""``weka_clear_thinking_mix`` value: every user turn picks clear or keep (seeded, even odds)."""

USER_TURN_GAP_MS = 120_000.0
"""A recorded idle gap at least this long, with no other agent of the trace
active inside it, is a wait on the user rather than on a tool."""

USER_TURN_XARGS = ("user_turn_open", "user_turn_end", "clear_thinking")
"""``vllm_xargs`` keys each labelled turn carries (0 or 1)."""

REASONING_SHARE_PERCENTILES = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.026363,
    0.084812,
    0.122239,
    0.151634,
    0.18251,
    0.205426,
    0.2336,
    0.258964,
    0.285124,
    0.30922,
    0.331891,
    0.357542,
    0.378378,
    0.404297,
    0.426387,
    0.450512,
    0.472991,
    0.495916,
    0.517705,
    0.540319,
    0.56105,
    0.581022,
    0.599613,
    0.620079,
    0.637833,
    0.656304,
    0.672337,
    0.690553,
    0.707024,
    0.723485,
    0.738658,
    0.750983,
    0.763415,
    0.775327,
    0.787879,
    0.799287,
    0.812152,
    0.823654,
    0.83366,
    0.842304,
    0.851414,
    0.860976,
    0.869501,
    0.878618,
    0.886939,
    0.895548,
    0.903159,
    0.909962,
    0.917745,
    0.925611,
    0.93338,
    0.94081,
    0.948314,
    0.955975,
    0.964359,
    0.975199,
    0.999989,
)
"""0th..100th percentiles of the reasoning share (characters) of an assistant
output in Claude Code sessions run on GLM-5.2 (17,960 main-conversation
outputs, mean 0.357, 43.8% without reasoning)."""


def has_live_markers(raw_messages: list[dict[str, Any]] | None) -> bool:
    """True when any message carries a live-assistant source marker."""
    return bool(raw_messages) and any(
        isinstance(m, dict) and LIVE_SOURCE_TURN_KEY in m for m in raw_messages
    )


def has_token_id_messages(raw_messages: list[dict[str, Any]] | None) -> bool:
    """True when the messages are token-ID segments rather than chat text."""
    return (
        bool(raw_messages)
        and isinstance(raw_messages[0], dict)
        and (TOKEN_IDS_KEY in raw_messages[0])
    )


def live_token_ids_from_responses(responses: list[Any]) -> list[int] | None:
    """Concatenate the ``token_ids`` a streamed completion returned, or None.

    Reads ``choices[0].token_ids`` from every chunk (vLLM ``return_token_ids``),
    which covers every generated token including ones whose text is empty.
    """
    ids: list[int] = []
    for response in responses:
        data = response.get_json()
        if not data:
            continue
        for choice in data.get("choices") or ():
            if choice.get("index", 0) == 0 and choice.get(TOKEN_IDS_KEY):
                ids.extend(choice[TOKEN_IDS_KEY])
    return ids or None


def live_output_from_parsed(
    parsed_responses: list[ParsedResponse],
) -> dict[str, str] | None:
    """Collect a response's content and reasoning text, or None when empty.

    Example: streamed chunks ``reasoning="a"``, ``reasoning="b"``,
    ``content="c"`` give ``{"content": "c", "reasoning_content": "ab"}``.
    """
    content: list[str] = []
    reasoning: list[str] = []
    for response in parsed_responses:
        data = response.data
        if data is None:
            continue
        if isinstance(data, ReasoningResponseData):
            if data.reasoning:
                reasoning.append(data.reasoning)
            if data.content:
                content.append(data.content)
        else:
            text = data.get_text()
            if text:
                content.append(text)
    if not content and not reasoning:
        return None
    out = {"content": "".join(content)}
    if reasoning:
        out["reasoning_content"] = "".join(reasoning)
    return out


def reasoning_share(seed: int, session_id: str, source_turn: int) -> float:
    """Seeded draw of an output's reasoning share from REASONING_SHARE_PERCENTILES.

    Deterministic per (seed, session, turn), independent of load order.
    """
    key = f"{seed}:{session_id}:{source_turn}".encode()
    u = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    x = u * 100
    i = min(int(x), 99)
    lo, hi = REASONING_SHARE_PERCENTILES[i], REASONING_SHARE_PERCENTILES[i + 1]
    return lo + (hi - lo) * (x - i)


def user_turn_openers(
    conversations: list[Conversation], gap_ms: float = USER_TURN_GAP_MS
) -> list[list[bool]]:
    """Per conversation, whether each turn opens a user turn.

    A conversation's first turn opens one. A later turn opens one when the
    recorded idle gap since the previous turn's end is at least ``gap_ms`` and
    no turn of another conversation of the same trace starts or ends inside
    it (a parent waiting on a subagent waits on a tool, not on the user).
    Turns without recorded timing never open one after the first.
    """
    events: dict[str, list[tuple[float, int]]] = {}
    for ci, conv in enumerate(conversations):
        for turn in conv.turns:
            if turn.timestamp is None:
                continue
            trace = turn.source_trace_id or conv.session_id
            end = turn.timestamp + (turn.api_time_ms or 0.0)
            events.setdefault(trace, []).extend(((turn.timestamp, ci), (end, ci)))
    for evs in events.values():
        evs.sort()
    times = {trace: [t for t, _ in evs] for trace, evs in events.items()}

    def other_active(trace: str, ci: int, lo: float, hi: float) -> bool:
        evs, ts = events.get(trace, []), times.get(trace, [])
        return any(c != ci for _, c in evs[bisect_right(ts, lo) : bisect_left(ts, hi)])

    result = []
    for ci, conv in enumerate(conversations):
        opens = []
        for k, turn in enumerate(conv.turns):
            if k == 0:
                opens.append(True)
                continue
            prev = conv.turns[k - 1]
            if turn.timestamp is None or prev.timestamp is None:
                opens.append(False)
                continue
            lo = prev.timestamp + (prev.api_time_ms or 0.0)
            hi = turn.timestamp
            trace = turn.source_trace_id or conv.session_id
            opens.append(hi - lo >= gap_ms and not other_active(trace, ci, lo, hi))
        result.append(opens)
    return result


def label_user_turns(
    conversations: list[Conversation],
    *,
    clear_thinking: bool,
    seed: int,
    gap_ms: float = USER_TURN_GAP_MS,
) -> None:
    """Label every turn with its user turn, in place.

    Each turn's ``extra_body["vllm_xargs"]`` gets ``user_turn_open`` (it opens
    a user turn), ``user_turn_end`` (the next turn opens one, or it is the
    conversation's last) and ``clear_thinking``. With ``clear_thinking`` every
    live-output marker also gets the first later turn that opens a user turn
    (from that turn on the output's reasoning is dropped) and a seeded
    reasoning share.
    """
    for conv, opens in zip(
        conversations, user_turn_openers(conversations, gap_ms), strict=True
    ):
        n = len(conv.turns)
        clear_at: list[int | None] = [None] * n
        nxt = None
        for k in range(n - 1, -1, -1):
            clear_at[k] = nxt
            if opens[k]:
                nxt = k
        for k, turn in enumerate(conv.turns):
            body = dict(turn.extra_body or {})
            xargs = dict(body.get("vllm_xargs") or {})
            xargs.update(
                user_turn_open=int(opens[k]),
                user_turn_end=int(k == n - 1 or opens[k + 1]),
                clear_thinking=int(clear_thinking),
            )
            body["vllm_xargs"] = xargs
            turn.extra_body = body
            if not clear_thinking or not has_live_markers(turn.raw_messages):
                continue
            messages = []
            for m in turn.raw_messages:
                source = m.get(LIVE_SOURCE_TURN_KEY) if isinstance(m, dict) else None
                if (
                    source is not None
                    and 0 <= source < n
                    and clear_at[source] is not None
                ):
                    m = {
                        **m,
                        LIVE_CLEAR_AT_TURN_KEY: clear_at[source],
                        LIVE_REASONING_SHARE_KEY: reasoning_share(
                            seed, conv.session_id, source
                        ),
                    }
                messages.append(m)
            turn.raw_messages = messages


def parse_clear_mix(mix: str | None) -> float | str | None:
    """``weka_clear_thinking_mix`` as a share of traces in (0, 1], ``"switch"``, or None.

    Example: ``"0.5"`` gives 0.5; ``"switch"`` gives ``"switch"``; ``"0"`` raises.
    """
    if mix is None:
        return None
    if mix == CLEAR_MIX_SWITCH:
        return mix
    try:
        share = float(mix)
    except ValueError:
        share = -1.0
    if not 0.0 < share <= 1.0:
        raise ValueError(
            f"weka_clear_thinking_mix must be a share in (0, 1] or {CLEAR_MIX_SWITCH!r}, got {mix!r}"
        )
    return share


def mix_clear_thinking(
    conversations: list[Conversation],
    *,
    mix: float | str,
    seed: int,
    gap_ms: float = USER_TURN_GAP_MS,
) -> None:
    """Apply the clear-thinking context edit to a seeded part of the traffic, in place, with no hint.

    A share ``mix`` clears exactly ``round(mix * T)`` of the ``T`` traces (seeded
    choice; every conversation of a cleared trace clears, the others keep their
    reasoning). ``"switch"`` lets every user turn of every conversation pick
    clear or keep (seeded, even odds): a request of a clearing user turn drops
    the reasoning of every output of earlier user turns, a request of a keeping
    one drops none. User turns are inferred as for ``label_user_turns``. Unlike
    ``label_user_turns`` no ``vllm_xargs`` are added: the server sees only the
    prompts.
    """
    openers = user_turn_openers(conversations, gap_ms)
    cleared_traces: set[str] = set()
    if mix != CLEAR_MIX_SWITCH:
        traces = sorted(
            {
                (c.turns[0].source_trace_id if c.turns else None) or c.session_id
                for c in conversations
            }
        )
        order = list(traces)
        random.Random(f"{seed}:clear-mix").shuffle(order)
        cleared_traces = set(order[: round(float(mix) * len(order))])
    for conv, opens in zip(conversations, openers, strict=True):
        n = len(conv.turns)
        starts = [k for k in range(n) if opens[k]]
        trace = (
            conv.turns[0].source_trace_id if conv.turns else None
        ) or conv.session_id
        spans_all: list[tuple[int, int]] = []
        for i, lo in enumerate(starts):
            hi = starts[i + 1] if i + 1 < len(starts) else n
            if mix == CLEAR_MIX_SWITCH:
                key = f"{seed}:{conv.session_id}:{lo}:switch".encode()
                clear = hashlib.sha256(key).digest()[0] < 128
            else:
                clear = trace in cleared_traces
            if clear:
                spans_all.append((lo, hi))
        if not spans_all:
            continue
        for turn in conv.turns:
            if not has_live_markers(turn.raw_messages):
                continue
            messages = []
            for m in turn.raw_messages:
                source = m.get(LIVE_SOURCE_TURN_KEY) if isinstance(m, dict) else None
                if source is not None and 0 <= source < n:
                    # Only user turns that open after the output's own turn drop it.
                    spans = [[lo, hi] for lo, hi in spans_all if lo > source]
                    if spans:
                        m = {
                            **m,
                            LIVE_CLEAR_SPANS_KEY: spans,
                            LIVE_REASONING_SHARE_KEY: reasoning_share(
                                seed, conv.session_id, source
                            ),
                        }
                messages.append(m)
            turn.raw_messages = messages


def _live_message(
    message: dict[str, Any],
    live: dict[str, str] | list[int] | None,
    current_turn: int | None = None,
) -> dict[str, Any]:
    """Return ``message`` without markers, with the live output swapped in when known.

    A context cut that kept only part of the recorded output keeps the same
    fraction of the live output: of its token IDs in token-ID replay, else of
    its characters, reasoning first (the order chat templates render it in).
    From the turn a clear-thinking marker names on, the output's reasoning is
    dropped: the leading reasoning share of its token IDs (live or recorded),
    or its reasoning text.
    """
    out = {k: v for k, v in message.items() if k not in _LIVE_KEYS}
    clear_at = message.get(LIVE_CLEAR_AT_TURN_KEY)
    cleared = (
        clear_at is not None and current_turn is not None and current_turn >= clear_at
    ) or (
        current_turn is not None
        and any(
            lo <= current_turn < hi
            for lo, hi in message.get(LIVE_CLEAR_SPANS_KEY) or ()
        )
    )
    keep = message.get(LIVE_KEEP_FRACTION_KEY)
    if TOKEN_IDS_KEY in message:
        if live is not None:
            ids = live if keep is None else live[: round(keep * len(live))]
            full = len(live)
        else:
            ids = message[TOKEN_IDS_KEY]
            full = round(len(ids) / keep) if keep else len(ids)
        if cleared:
            ids = ids[round(message[LIVE_REASONING_SHARE_KEY] * full) :]
        out[TOKEN_IDS_KEY] = ids
        return out
    if live is None:
        return out
    reasoning = "" if cleared else live.get("reasoning_content", "")
    content = live.get("content", "")
    if keep is not None:
        budget = round(keep * (len(reasoning) + len(content)))
        content = content[: max(0, budget - len(reasoning))]
        reasoning = reasoning[:budget]
    out["content"] = content
    if reasoning:
        out["reasoning_content"] = reasoning
    else:
        out.pop("reasoning_content", None)
    return out


def substitute_live_outputs(
    turns: list[Any],
    outputs: dict[int, dict[str, str] | list[int]],
    current_turn: int | None = None,
) -> list[Any]:
    """Copy ``turns`` with marked assistant messages rendered from ``outputs``.

    ``turns`` are dataset ``Turn`` objects shared across sessions, so marked
    turns are copied rather than mutated. Unmarked turns pass through as-is.
    A marker whose source turn has no recorded live output (history before a
    mid-trace start, a failed request) keeps the recorded text.
    ``current_turn`` is the turn being sent, for clear-thinking markers.
    """
    result = []
    for turn in turns:
        raw = turn.raw_messages
        if not has_live_markers(raw):
            result.append(turn)
            continue
        messages = [
            _live_message(m, outputs.get(m[LIVE_SOURCE_TURN_KEY]), current_turn)
            if isinstance(m, dict) and LIVE_SOURCE_TURN_KEY in m
            else m
            for m in raw
        ]
        result.append(turn.model_copy(update={"raw_messages": messages}))
    return result
