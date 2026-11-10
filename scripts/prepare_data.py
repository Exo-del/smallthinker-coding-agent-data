"""Prepare the SmallThinker coding-agent SFT corpus.

Reads the raw source dumps in ``raw/``, normalises every source into one
canonical ChatML dialect (inline think block + inline tool-call block), filters
on the *exact* SmallThinker token counts, deduplicates, and writes gzipped
JSONL shards plus a manifest.

Usage:
    python3 prepare_data.py --out ../smallthinker-coding-agent-data [--sources hermes,trjxter]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sftnorm as N  # noqa: E402

RAW = HERE / "raw"
CACHE = HERE / "cache"

MODEL_ID = "Tiiny/SmallThinker-4BA0.6B-Instruct"

MAX_SEQ_LEN = 3072
MAX_TARGET_TOKENS = 2560
MIN_PROMPT_TOKENS = 24
# A target can never exceed MAX_TARGET_TOKENS tokens.  No real text tokenises
# to fewer than ~3 characters per token, so anything above 16 chars/token is
# certainly over the limit.  This lets us skip tokenising huge outputs (the
# Jackrong dump is full of 100k+ character solutions) without changing which
# conversations are usable - the exact token filter still decides.
MAX_ASSISTANT_CHARS = MAX_TARGET_TOKENS * 16
TOKEN_CHUNK = 64
TOKEN_CHAR_BUDGET = 6_000_000
# Nothing past this many rendered characters can influence a turn that fits:
# 90k chars is >= 5.6k tokens even at an unusually sparse 16 chars/token, and
# 5.6k tokens (MAX_SEQ_LEN + MAX_TARGET_TOKENS) is already past the point
# where every later turn is dropped.  Open-SWE trajectories reach 1.3 MB of
# rendered text, so the prefix cut is what keeps tokenisation in memory.
MAX_RENDER_CHARS = 90_000
CUTOFF_TOKENS = MAX_SEQ_LEN + MAX_TARGET_TOKENS + 64
VALID_FRACTION = 0.02
SEED = 42

# source -> (category, max_conversations, max_target_tokens_budget)
SOURCES: Dict[str, Dict[str, Any]] = {
    "hermes":      {"category": "tool",      "max_conv": 16431, "budget": 7_500_000},
    "zake7749":    {"category": "tool",      "max_conv": 284,   "budget": 300_000},
    "nemotron":    {"category": "tool",      "max_conv": 3000,  "budget": 1_500_000},
    "openswe":     {"category": "tool",      "max_conv": 1500,  "budget": 1_200_000},
    "trjxter":     {"category": "reasoning", "max_conv": 8014,  "budget": 3_000_000},
    "jackrong":    {"category": "reasoning", "max_conv": 100000,"budget": 2_000_000},
    "blythet":     {"category": "reasoning", "max_conv": 1008,  "budget": 400_000},
    "beyoru":      {"category": "reasoning", "max_conv": 1000,  "budget": 400_000},
    "codefeedback": {"category": "coding",   "max_conv": 12000, "budget": 4_000_000},
}

LICENCES = {
    "hermes": "apache-2.0",
    "zake7749": "unspecified",
    "nemotron": "nvidia-open-model-license",
    "openswe": "nvidia-open-model-license",
    "trjxter": "apache-2.0",
    "jackrong": "CC-BY-4.0 / CC-BY-SA-4.0 (per row metadata)",
    "blythet": "cc-by-sa-4.0",
    "beyoru": "apache-2.0",
    "codefeedback": "apache-2.0",
}

UPSTREAM = {
    "hermes": "https://huggingface.co/datasets/r0b0tlab/deepseek-hermes-reasoning-traces",
    "zake7749": "https://huggingface.co/datasets/zake7749/deepseek-v4-pro-agent-tool-calling-trajectory",
    "nemotron": "https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v4",
    "openswe": "https://huggingface.co/datasets/nvidia/Open-SWE-Traces",
    "trjxter": "https://huggingface.co/datasets/trjxter/DeepSeek-V4-Pro-Reasoning-8000x",
    "jackrong": "https://huggingface.co/datasets/Jackrong/DeepSeek-V4-Pro-Distilled-200K",
    "blythet": "https://huggingface.co/datasets/blythet/deepseek-v4-pro-math-cot-1k",
    "beyoru": "https://huggingface.co/datasets/beyoru/Deepseek-v4-pro-max-distill-1500x",
    "codefeedback": "https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==========================================================================
# Source builders - each returns a list of conversation dicts
# ==========================================================================


def _conv(cid: str, source: str, messages: List[Dict[str, str]],
          tools: Optional[List[Dict[str, Any]]], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    messages = N.sanitize_conversation(messages)
    N.ensure_system(messages)
    return {
        "id": cid,
        "source": source,
        "category": SOURCES[source]["category"],
        "messages": messages,
        "tools": tools or [],
        "meta": meta or {},
    }


def build_hermes() -> List[Dict[str, Any]]:
    """DeepSeek V4 Pro Hermes traces.

    The upstream repo publishes VRAM-tiered variants; we use all of them
    (nano <=2048 tok, budget <=4096, standard <=8192) so short and long
    trajectories are both represented.  Tools schemas are synthesised from the
    arguments the traces actually use.
    """
    import pyarrow.parquet as pq

    files = [RAW / "hermes_nano.parquet", RAW / "hermes_budget.parquet",
             RAW / "hermes_standard.parquet"]
    rows: List[Dict[str, Any]] = []
    for path in files:
        if not path.exists():
            continue
        part = pq.read_table(path).to_pylist()
        log("hermes: %s -> %d rows" % (path.name, len(part)))
        rows.extend(part)
    if not rows:
        raise SystemExit("no hermes files found in %s" % RAW)

    collector = N.ToolSpecCollector()
    parsed = []
    for r in rows:
        leading, events = N.parse_hermes_events(r.get("text") or "")
        parsed.append((r, leading, events))

    # Pass 1 - build the messages, canonicalise them and collect the tool
    # names from the *canonical* assistant turns.  Scanning the raw events is
    # not enough: the source sometimes nests a <tool_call> inside a
    # <think> block, and the canonicaliser hoists it out.
    built = []
    for r, leading, events in parsed:
        user = N.hermes_user_content(leading, r.get("task") or "")
        if not user:
            continue
        msgs = N.hermes_events_to_messages(events)
        # a non-"User:" leading chunk is assistant prose, not a user turn
        lead = (leading or "").strip()
        if lead and not lead.lower().startswith("user:"):
            if msgs and msgs[0]["role"] == "assistant":
                msgs[0]["content"] = N.sanitize_assistant_content(lead + "\n" + msgs[0]["content"])
        msgs.insert(0, {"role": "user", "content": user})
        msgs = N.sanitize_conversation(msgs)
        if not any(m["role"] == "assistant" for m in msgs):
            continue
        names: List[str] = []
        for m in msgs:
            if m["role"] != "assistant":
                continue
            for block in (m.get("content") or "").split(N.TOOL_CALL_OPEN)[1:]:
                body = block.split(N.TOOL_CALL_CLOSE)[0]
                call = N.parse_call_payload(body)
                if call is None:
                    continue
                name = N.HERMES_TOOL_ALIASES.get(call["name"], call["name"]).strip()
                names.append(name)
                collector.observe(name, call["arguments"])
        built.append((r, msgs, names))

    specs = collector.build_specs()
    log("hermes: %d distinct tool names -> %s" % (len(specs), sorted(specs)))

    convs = []
    for r, msgs, names in built:
        tools = [specs[n] for n in dict.fromkeys(names) if n in specs]
        convs.append(_conv("hermes:%s" % r.get("seed_id"), "hermes", msgs, tools,
                           {"seed_id": r.get("seed_id"), "variant": r.get("variant")}))
    return convs


Zake7749_SYSTEM = (
    "You are a customer service agent for the {domain} domain. Help the user with the "
    "tools provided. Authenticate first when a policy requires it, validate every action "
    "against the retrieved data, ask for explicit confirmation before any write action, "
    "and never invent tool results. Make at most one tool call per turn and never combine "
    "a tool call with a user-facing reply. Always emit valid JSON arguments."
)


def build_zake7749() -> List[Dict[str, Any]]:
    """DeepSeek V4 Pro agentic tool-calling trajectories (ToolScale derived).

    The upstream system prompt is a ~9.4 kB domain policy and the tool list has
    21 full schemas: together they are ~5.8 k tokens, which does not fit a 2 k
    window.  We therefore (a) replace the policy with a compact agent brief and
    (b) expose only the tools the trajectory actually calls, with compressed
    schemas.  The behavioural policy is demonstrated by the trajectory itself,
    so the supervision signal is preserved while the prompt becomes affordable.
    """
    import pyarrow.parquet as pq

    rows = pq.read_table(RAW / "zake7749_agent.parquet").to_pylist()
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = r["id"]
        if key not in best or len(r["messages"]) > len(best[key]["messages"]):
            best[key] = r
    log(f"zake7749: {len(rows)} rows -> {len(best)} trajectories")

    convs = []
    for key, r in best.items():
        all_tools = N.normalize_tools(r.get("tools"))
        by_name = {t["function"]["name"]: t for t in all_tools}
        msgs: List[Dict[str, str]] = []
        used: List[str] = []
        for m in r["messages"]:
            role = m.get("role")
            content = m.get("content") or ""
            if role == "assistant":
                calls = []
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                    if fn.get("name"):
                        calls.append({"name": fn["name"], "arguments": fn.get("arguments")})
                        if fn["name"] not in used:
                            used.append(fn["name"])
                text = N.assistant_text(
                    reasoning=m.get("reasoning") or m.get("reasoning_content"),
                    content=content,
                    tool_calls=calls,
                )
                if text.strip():
                    msgs.append({"role": "assistant", "content": N.sanitize_assistant_content(text)})
            elif role == "tool":
                msgs.append({"role": "tool", "content": content})
            elif role == "user":
                msgs.append({"role": role, "content": content})
        if not any(m["role"] == "assistant" for m in msgs):
            continue
        tools = [by_name[n] for n in used if n in by_name]
        # synthesise a stub for any tool the trajectory calls but the schema
        # list does not declare
        for n in used:
            if n not in by_name:
                tools.append({"type": "function", "function": {
                    "name": n, "description": "Invoke the '%s' tool." % n,
                    "parameters": {"type": "object", "properties": {}}}})
                by_name[n] = tools[-1]
        tools = N.compress_tools(tools)
        system = Zake7749_SYSTEM.format(domain=(r.get("domain") or "general"))
        msgs.insert(0, {"role": "system", "content": system})
        convs.append(_conv("zake7749:%s" % key, "zake7749", msgs, tools,
                           {"domain": r.get("domain"), "score": r.get("score"),
                            "system_prompt_replaced": True}))
    return convs


def build_nemotron(max_conv: int) -> List[Dict[str, Any]]:
    """Stream the 10.7 GB tool_calling dump and keep a uniform sample."""
    import pyarrow.parquet as pq  # noqa: F401  (keeps pyarrow import cost in one place)

    path = RAW / "nemotron_tool_calling.jsonl"
    cache_path = CACHE / "nemotron_subset.jsonl.gz"
    if cache_path.exists():
        log(f"nemotron: loading cached subset {cache_path}")
        convs = []
        with gzip.open(cache_path, "rt", encoding="utf-8") as f:
            for line in f:
                convs.append(json.loads(line))
        return convs

    if not path.exists():
        log("nemotron: RAW FILE MISSING - skipping source")
        return []

    kept: List[Dict[str, Any]] = []
    scanned = 0
    t0 = time.time()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            scanned += 1
            try:
                r = json.loads(line)
            except Exception:
                continue
            tools = N.compress_tools(N.normalize_tools(r.get("tools")))
            msgs: List[Dict[str, str]] = []
            n_calls = 0
            for m in r.get("messages") or []:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                content = m.get("content") or ""
                if role == "assistant":
                    calls = []
                    for tc in m.get("tool_calls") or []:
                        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                        if fn.get("name"):
                            calls.append({"name": fn["name"], "arguments": fn.get("arguments")})
                    n_calls += len(calls)
                    text = N.assistant_text(
                        reasoning=m.get("reasoning_content") or m.get("reasoning"),
                        content=content,
                        tool_calls=calls,
                    )
                    if text.strip():
                        msgs.append({"role": "assistant", "content": N.sanitize_assistant_content(text)})
                elif role == "tool":
                    msgs.append({"role": "tool", "content": content})
                elif role in ("system", "user"):
                    msgs.append({"role": role, "content": content})
            if not n_calls or not any(m["role"] == "assistant" for m in msgs):
                continue
            kept.append(_conv("nemotron:%s" % (r.get("uuid") or scanned), "nemotron", msgs, tools,
                              {"parallel_tool_calls": r.get("parallel_tool_calls")}))
            if len(kept) >= max_conv:
                break
            if scanned % 20000 == 0:
                log(f"nemotron: scanned {scanned} lines, kept {len(kept)} ({time.time()-t0:.0f}s)")
    log(f"nemotron: scanned {scanned} lines -> kept {len(kept)}")

    CACHE.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt", encoding="utf-8") as f:
        for c in kept:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return kept


def build_openswe(max_conv: int) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    files = sorted(RAW.glob("openswe_*.parquet"))
    log(f"openswe: {len(files)} shards")
    pool: List[Dict[str, Any]] = []
    for path in files:
        rows = pq.read_table(path).to_pylist()
        # resolved trajectories first
        rows.sort(key=lambda r: (not bool(r.get("resolved")),))
        for r in rows:
            all_tools = N.normalize_tools(r.get("tools"))
            by_name = {t["function"]["name"]: t for t in all_tools}
            msgs: List[Dict[str, str]] = []
            used: List[str] = []
            for m in r.get("messages") or []:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                content = m.get("content") or ""
                if role == "assistant":
                    calls = []
                    for tc in m.get("tool_calls") or []:
                        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                        if fn.get("name"):
                            calls.append({"name": fn["name"], "arguments": fn.get("arguments")})
                            if fn["name"] not in used:
                                used.append(fn["name"])
                    text = N.assistant_text(
                        reasoning=m.get("reasoning_content") or m.get("reasoning"),
                        content=content,
                        tool_calls=calls,
                    )
                    if text.strip():
                        msgs.append({"role": "assistant", "content": N.sanitize_assistant_content(text)})
                elif role == "tool":
                    msgs.append({"role": "tool", "content": content})
                elif role in ("system", "user"):
                    msgs.append({"role": role, "content": content})
            if not any(m["role"] == "assistant" for m in msgs):
                continue
            # only expose the tools this trajectory actually drives: the full
            # Bash/Edit/Read schemas cost ~900 prompt tokens and would push
            # every turn of a 23-turn trajectory past the 2 k window.
            tools = N.compress_tools([by_name[n] for n in used if n in by_name],
                                      desc_chars=140, prop_desc_chars=70)
            for n in used:
                if n not in by_name:
                    tools.append({"type": "function", "function": {
                        "name": n, "description": "Invoke the '%s' tool." % n,
                        "parameters": {"type": "object", "properties": {}}}})
            pool.append(_conv("openswe:%s" % (r.get("trajectory_id") or r.get("instance_id")),
                              "openswe", msgs, tools,
                              {"repo": r.get("repo"), "resolved": bool(r.get("resolved")),
                               "instance_id": r.get("instance_id")}))
    log(f"openswe: pool={len(pool)}")
    return pool[:max_conv]


def _inline_think_split(text: str) -> Tuple[Optional[str], str]:
    """Split an inline-think assistant text into (reasoning, prose)."""
    t = text or ""
    if N.THINK_OPEN in t:
        i = t.index(N.THINK_OPEN)
        j = t.find(N.THINK_CLOSE, i)
        if j != -1:
            reasoning = t[i + len(N.THINK_OPEN) : j].strip()
            prose = (t[:i] + t[j + len(N.THINK_CLOSE) :]).strip()
            return reasoning, prose
    return None, t.strip()


def _simple_conv(cid: str, source: str, user: str, assistant: str,
                 meta: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    user = (user or "").strip()
    if not user:
        return None
    content = N.canonicalize_assistant(assistant or "")
    if not content:
        return None
    msgs = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": content},
    ]
    return _conv(cid, source, msgs, None, meta)


def build_trjxter(max_conv: int) -> List[Dict[str, Any]]:
    convs: List[Dict[str, Any]] = []
    path = RAW / "trjxter_8000x.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(convs) >= max_conv:
                break
            r = json.loads(line)
            c = _simple_conv("trjxter:%s" % (r.get("id") or i), "trjxter",
                             r.get("input") or "", r.get("output") or "",
                             {"domain": r.get("domain")})
            if c:
                convs.append(c)
    return convs


def build_jackrong(max_conv: int) -> List[Dict[str, Any]]:
    import gzip as _gzip

    convs: List[Dict[str, Any]] = []
    for path in sorted(RAW.glob("jackrong_stem_*.jsonl.gz")):
        with _gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if len(convs) >= max_conv:
                    break
                r = json.loads(line)
                c = _simple_conv("jackrong:%s" % (r.get("id") or i), "jackrong",
                                 r.get("input") or "", r.get("output") or "",
                                 {"domain": r.get("domain"), "license": (r.get("meta") or {}).get("license")})
                if c:
                    convs.append(c)
        if len(convs) >= max_conv:
            break
    return convs


def build_blythet(max_conv: int) -> List[Dict[str, Any]]:
    convs: List[Dict[str, Any]] = []
    path = RAW / "blythet_train_core.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(convs) >= max_conv:
                break
            r = json.loads(line)
            assistant = N.assistant_text(reasoning=r.get("reasoning"), content=r.get("answer"))
            c = _simple_conv("blythet:%s" % (r.get("id") or i), "blythet",
                             r.get("problem") or "", assistant)
            if c:
                convs.append(c)
    return convs


def build_beyoru(max_conv: int) -> List[Dict[str, Any]]:
    convs: List[Dict[str, Any]] = []
    path = RAW / "beyoru_1500x.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(convs) >= max_conv:
                break
            r = json.loads(line)
            assistant = N.assistant_text(reasoning=r.get("reasoning"), content=r.get("response"))
            c = _simple_conv("beyoru:%s" % (r.get("id") or i), "beyoru",
                             r.get("prompt") or "", assistant,
                             {"domain": r.get("domain")})
            if c:
                convs.append(c)
    return convs


def build_codefeedback(max_conv: int) -> List[Dict[str, Any]]:
    convs: List[Dict[str, Any]] = []
    path = RAW / "codefeedback.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(convs) >= max_conv:
                break
            r = json.loads(line)
            c = _simple_conv("codefeedback:%d" % i, "codefeedback",
                             r.get("query") or "", r.get("answer") or "",
                             {"lang": r.get("lang"), "resource": r.get("resource")})
            if c:
                convs.append(c)
    return convs


BUILDERS = {
    "hermes": lambda: build_hermes(),
    "zake7749": lambda: build_zake7749(),
    "nemotron": lambda: build_nemotron(SOURCES["nemotron"]["max_conv"]),
    "openswe": lambda: build_openswe(SOURCES["openswe"]["max_conv"]),
    "trjxter": lambda: build_trjxter(SOURCES["trjxter"]["max_conv"]),
    "jackrong": lambda: build_jackrong(SOURCES["jackrong"]["max_conv"]),
    "blythet": lambda: build_blythet(SOURCES["blythet"]["max_conv"]),
    "beyoru": lambda: build_beyoru(SOURCES["beyoru"]["max_conv"]),
    "codefeedback": lambda: build_codefeedback(SOURCES["codefeedback"]["max_conv"]),
}


# ==========================================================================
# Tokenisation / selection
# ==========================================================================


def _truncate_oversized(conv: Dict[str, Any]) -> None:
    """Drop everything from the first over-long assistant turn onwards.

    An assistant turn of more than ``MAX_ASSISTANT_CHARS`` characters cannot
    fit into ``MAX_TARGET_TOKENS`` tokens, so it is useless as a target.  We
    cut the conversation at that point instead of tokenising hundreds of
    kilobytes of text we are going to throw away anyway.
    """
    messages = conv["messages"]
    for idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        if len(message.get("content") or "") > MAX_ASSISTANT_CHARS:
            conv["messages"] = messages[:idx]
            conv["meta"]["truncated_oversized"] = True
            return not any(m.get("role") == "assistant" for m in conv["messages"])
    return True


def _block_starts(messages: Sequence[Dict[str, str]]) -> List[int]:
    """Message index where each rendered block starts.

    Consecutive ``tool`` messages render as a single block, everything else is
    one block per message.
    """
    starts: List[int] = []
    i = 1 if (messages and messages[0].get("role") == "system") else 0
    n = len(messages)
    while i < n:
        starts.append(i)
        if messages[i].get("role") == "tool":
            while i < n and messages[i].get("role") == "tool":
                i += 1
        else:
            i += 1
    return starts


def _render_prefix(conv: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Render the conversation, cutting everything past ``MAX_RENDER_CHARS``.

    Blocks past the cut belong to turns whose prompt is already far beyond
    ``MAX_SEQ_LEN``, so they can never be supervised; dropping them keeps a
    single 1 MB Open-SWE trajectory from dominating memory.
    """
    messages = conv["messages"]
    sys_block, blocks = N.render_blocks(messages, conv.get("tools") or None)
    total = len(sys_block)
    keep = 0
    for block in blocks:
        total += len(block)
        if total > MAX_RENDER_CHARS:
            break
        keep += 1
    if keep < len(blocks):
        starts = _block_starts(messages)
        cut = starts[keep] if keep < len(starts) else len(messages)
        conv["messages"] = messages[:cut]
        conv["meta"]["truncated_context"] = True
        blocks = blocks[:keep]
    return sys_block, blocks


def _select_from_blocks(conv: Dict[str, Any], t_sys: int, t_gen: int,
                        t_blocks: Sequence[int]) -> Optional[Dict[str, Any]]:
    """Pick the assistant turns that fit the token budget.

    ``t_blocks[i]`` is the token count of the i-th rendered message block.
    """
    messages = conv["messages"]

    # map every message index to its block index and the token count before it
    entries: List[Tuple[int, int, int]] = []
    bi = 0
    i = 1 if (messages and messages[0].get("role") == "system") else 0
    cum = 0
    n = len(messages)
    while i < n:
        if messages[i].get("role") == "tool":
            while i < n and messages[i].get("role") == "tool":
                entries.append((i, bi, cum))
                i += 1
            cum += t_blocks[bi]
            bi += 1
        else:
            entries.append((i, bi, cum))
            cum += t_blocks[bi]
            bi += 1
            i += 1

    kept: List[Dict[str, Any]] = []
    last_idx = -1
    for msg_idx, block_idx, before in entries:
        m = messages[msg_idx]
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        if not content.strip():
            continue
        p_len = t_sys + before + t_gen
        t_len = t_blocks[block_idx] - t_gen
        if t_len <= 0:
            continue
        if t_len > MAX_TARGET_TOKENS:
            continue
        if p_len + t_len > MAX_SEQ_LEN:
            continue
        if p_len < MIN_PROMPT_TOKENS:
            continue
        kept.append({
            "i": msg_idx,
            "prompt_tokens": p_len,
            "target_tokens": t_len,
            "think": N.THINK_OPEN in content,
            "tool_call": N.TOOL_CALL_OPEN in content,
        })
        last_idx = msg_idx

    if not kept:
        return None
    conv["messages"] = messages[: last_idx + 1]
    conv["meta"]["turns"] = kept
    conv["meta"]["dedup_key"] = N.dedup_key(
        "\x00".join((m.get("content") or "") for m in conv["messages"]),
        conv["messages"][last_idx].get("content") or "",
    )
    return conv


def select_turns(tok, conv: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Keep the assistant turns that fit the token budget.

    Uses the verified fast renderer (see ``sftnorm.render_blocks``): every
    conversation is rendered and tokenised once, block by block, instead of
    once per assistant turn.  Returns the conversation with a ``meta.turns``
    list, or ``None`` when no turn is usable.
    """
    _truncate_oversized(conv)
    if not conv["messages"]:
        return None
    sys_block, blocks = _render_prefix(conv)
    t_sys = len(tok(sys_block, add_special_tokens=False)["input_ids"])
    t_gen = len(tok(N.GEN_PROMPT, add_special_tokens=False)["input_ids"])
    t_blocks = [len(tok(b, add_special_tokens=False)["input_ids"]) for b in blocks]
    return _select_from_blocks(conv, t_sys, t_gen, t_blocks)


def select_turns_batch(tok, convs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Same as ``select_turns`` but tokenises many conversations at once.

    Conversations are grouped into batches bounded by *characters* as well as
    count, because a single Open-SWE trajectory can render to over a megabyte
    and a naive 256-conversation batch runs the box out of memory.
    """
    out: List[Dict[str, Any]] = []
    t_gen = len(tok([N.GEN_PROMPT], add_special_tokens=False)["input_ids"][0])
    i = 0
    n = len(convs)
    while i < n:
        batch: List[Tuple[Dict[str, Any], List[str]]] = []
        chars = 0
        while i < n and len(batch) < TOKEN_CHUNK and chars < TOKEN_CHAR_BUDGET:
            conv = convs[i]
            i += 1
            if not _truncate_oversized(conv) or not conv["messages"]:
                continue
            sys_block, blocks = _render_prefix(conv)
            texts = [sys_block] + blocks
            batch.append((conv, texts))
            chars += sum(len(t) for t in texts)
        if not batch:
            continue
        flat = [t for _, texts in batch for t in texts]
        encoded = tok(flat, add_special_tokens=False)["input_ids"]
        pos = 0
        for conv, texts in batch:
            t_sys = len(encoded[pos])
            pos += 1
            t_blocks = [len(x) for x in encoded[pos : pos + len(texts) - 1]]
            pos += len(texts) - 1
            selected = _select_from_blocks(conv, t_sys, t_gen, t_blocks)
            if selected is not None:
                out.append(selected)
        del flat, encoded
    return out


def apply_budget(convs: List[Dict[str, Any]], source: str, rng: random.Random) -> List[Dict[str, Any]]:
    """Shuffle, then keep conversations until the source budget is exhausted."""
    cfg = SOURCES[source]
    order = list(range(len(convs)))
    rng.shuffle(order)
    out: List[Dict[str, Any]] = []
    spent = 0
    for idx in order:
        if len(out) >= cfg["max_conv"] or spent >= cfg["budget"]:
            break
        c = convs[idx]
        cost = sum(t["target_tokens"] for t in c["meta"]["turns"])
        out.append(c)
        spent += cost
    return out


# ==========================================================================
# Writing
# ==========================================================================


def split_of(conv_id: str) -> str:
    h = hashlib.sha1(conv_id.encode()).hexdigest()
    return "valid" if int(h[:8], 16) / 0xFFFFFFFF < VALID_FRACTION else "train"


def write_shards(convs: List[Dict[str, Any]], out_dir: Path, prefix: str,
                 per_shard: int = 2000) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for shard_idx, start in enumerate(range(0, len(convs), per_shard)):
        chunk = convs[start : start + per_shard]
        name = "%s/part-%03d.jsonl.gz" % (prefix, shard_idx)
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as f:
            for c in chunk:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        written.append("data/" + name)
    return written


def pct(values: Sequence[int], q: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dataset repo directory")
    ap.add_argument("--sources", default="", help="comma separated subset of sources")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    log("loading tokenizer %s" % MODEL_ID)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    wanted = [s.strip() for s in args.sources.split(",") if s.strip()] or list(SOURCES)
    rng = random.Random(SEED)
    CACHE.mkdir(parents=True, exist_ok=True)

    all_convs: List[Dict[str, Any]] = []
    per_source: Dict[str, Any] = {}

    for source in wanted:
        cache_path = CACHE / ("%s.selected.jsonl.gz" % source)
        if cache_path.exists() and not args.no_cache:
            log("%s: loading cached selection" % source)
            convs = []
            with gzip.open(cache_path, "rt", encoding="utf-8") as f:
                for line in f:
                    convs.append(json.loads(line))
        else:
            t0 = time.time()
            convs = BUILDERS[source]()
            log("%s: built %d conversations in %.0fs" % (source, len(convs), time.time() - t0))
            before = len(convs)
            convs = select_turns_batch(tok, convs)
            log("%s: %d/%d conversations survive token filtering (%.0fs)"
                % (source, len(convs), before, time.time() - t0))
            with gzip.open(cache_path, "wt", encoding="utf-8") as f:
                for c in convs:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
        convs = apply_budget(convs, source, rng)

        # dedup
        seen = set()
        unique = []
        dup = 0
        for c in convs:
            sig = c["meta"].get("dedup_key") or c["id"]
            if sig in seen:
                dup += 1
                continue
            seen.add(sig)
            unique.append(c)
        convs = unique

        target_tokens = [t["target_tokens"] for c in convs for t in c["meta"]["turns"]]
        prompt_tokens = [t["prompt_tokens"] for c in convs for t in c["meta"]["turns"]]
        n_turns = sum(len(c["meta"]["turns"]) for c in convs)
        n_think = sum(1 for c in convs for t in c["meta"]["turns"] if t["think"])
        n_call = sum(1 for c in convs for t in c["meta"]["turns"] if t["tool_call"])
        per_source[source] = {
            "category": SOURCES[source]["category"],
            "conversations": len(convs),
            "assistant_turns": n_turns,
            "duplicates_dropped": dup,
            "think_turns": n_think,
            "tool_call_turns": n_call,
            "think_frac": round(n_think / max(1, n_turns), 4),
            "tool_call_frac": round(n_call / max(1, n_turns), 4),
            "target_tokens": sum(target_tokens),
            "prompt_tokens": sum(prompt_tokens),
            "target_tokens_p50": pct(target_tokens, 0.50),
            "target_tokens_p90": pct(target_tokens, 0.90),
            "target_tokens_max": max(target_tokens) if target_tokens else 0,
            "prompt_tokens_p50": pct(prompt_tokens, 0.50),
            "prompt_tokens_p90": pct(prompt_tokens, 0.90),
            "licence": LICENCES.get(source, "unspecified"),
            "upstream": UPSTREAM.get(source, ""),
        }
        log("%s: %s" % (source, json.dumps(per_source[source])))
        all_convs.extend(convs)

    # global dedup across sources
    seen = set()
    final: List[Dict[str, Any]] = []
    cross_dup = 0
    for c in all_convs:
        sig = c["meta"].get("dedup_key") or c["id"]
        if sig in seen:
            cross_dup += 1
            continue
        seen.add(sig)
        final.append(c)
    log("cross-source duplicates dropped: %d" % cross_dup)

    train = [c for c in final if split_of(c["id"]) == "train"]
    valid = [c for c in final if split_of(c["id"]) == "valid"]
    rng.shuffle(train)
    rng.shuffle(valid)
    log("train=%d valid=%d" % (len(train), len(valid)))

    out_root = Path(args.out)
    data_dir = out_root / "data"
    if data_dir.exists():
        for p in sorted(data_dir.rglob("*.jsonl.gz")):
            p.unlink()
    shards = write_shards(train, data_dir, "train")
    shards += write_shards(valid, data_dir, "valid")
    log("wrote %d shards" % len(shards))

    # ---- manifest ------------------------------------------------------
    cat_tokens = defaultdict(int)
    cat_turns = defaultdict(int)
    cat_convs = defaultdict(int)
    total_target = 0
    total_prompt = 0
    total_turns = 0
    for src, info in per_source.items():
        cat = info["category"]
        cat_tokens[cat] += info["target_tokens"]
        cat_turns[cat] += info["assistant_turns"]
        cat_convs[cat] += info["conversations"]
        total_target += info["target_tokens"]
        total_prompt += info["prompt_tokens"]
        total_turns += info["assistant_turns"]

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": MODEL_ID,
        "generator": "prepare_data.py",
        "config": {
            "max_seq_len": MAX_SEQ_LEN,
            "max_target_tokens": MAX_TARGET_TOKENS,
            "min_prompt_tokens": MIN_PROMPT_TOKENS,
            "valid_fraction": VALID_FRACTION,
            "seed": SEED,
        },
        "totals": {
            "conversations": len(final),
            "train_conversations": len(train),
            "valid_conversations": len(valid),
            "assistant_turns": total_turns,
            "target_tokens": total_target,
            "prompt_tokens": total_prompt,
            "target_tokens_p50": pct([t["target_tokens"] for c in final for t in c["meta"]["turns"]], 0.5),
            "target_tokens_p90": pct([t["target_tokens"] for c in final for t in c["meta"]["turns"]], 0.9),
            "target_tokens_max": max([t["target_tokens"] for c in final for t in c["meta"]["turns"]] or [0]),
        },
        "mixture_by_category": {
            cat: {
                "conversations": cat_convs[cat],
                "assistant_turns": cat_turns[cat],
                "target_tokens": cat_tokens[cat],
                "target_token_share": round(cat_tokens[cat] / max(1, total_target), 4),
                "turn_share": round(cat_turns[cat] / max(1, total_turns), 4),
            }
            for cat in sorted(cat_tokens)
        },
        "sources": per_source,
        "shards": shards,
        "licences": LICENCES,
        "upstream": UPSTREAM,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log("manifest written to %s" % (out_root / "manifest.json"))
    print(json.dumps(manifest["mixture_by_category"], indent=2))
    print(json.dumps(manifest["totals"], indent=2))


if __name__ == "__main__":
    main()
