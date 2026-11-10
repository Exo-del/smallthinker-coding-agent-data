"""Verify that the fast block renderer matches the real chat template exactly.

Checks, on real conversations from every source:
  1. render_conversation(...) == tok.apply_chat_template(..., tokenize=False)
  2. the incremental token arithmetic (prompt/target) equals the naive
     "render the full prefix per turn" arithmetic
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sftnorm as N
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Tiiny/SmallThinker-4BA0.6B-Instruct", trust_remote_code=True)


def check(messages, tools, label):
    tools = tools or None
    fast = N.render_conversation(messages, tools)
    ref = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
    if fast != ref:
        print("!! STRING MISMATCH", label)
        for i, (a, b) in enumerate(zip(fast, ref)):
            if a != b:
                print("   first diff at", i)
                print("   fast:", repr(fast[max(0, i - 120):i + 120]))
                print("   ref :", repr(ref[max(0, i - 120):i + 120]))
                break
        else:
            print("   length differs", len(fast), len(ref))
            print("   fast tail:", repr(fast[-160:]))
            print("   ref  tail:", repr(ref[-160:]))
        return False

    # incremental token arithmetic
    sys_block, blocks = N.render_blocks(messages, tools)
    t_sys = len(tok(sys_block, add_special_tokens=False)["input_ids"])
    t_gen = len(tok(N.GEN_PROMPT, add_special_tokens=False)["input_ids"])
    t_blocks = [len(tok(b, add_special_tokens=False)["input_ids"]) for b in blocks]

    # map blocks back to messages (tool runs collapse several messages)
    idx = 1 if (messages and messages[0].get("role") == "system") else 0
    block_for_message: Dict[int, int] = {}
    bi = 0
    i = idx
    while i < len(messages):
        if messages[i].get("role") == "tool":
            while i < len(messages) and messages[i].get("role") == "tool":
                block_for_message[i] = bi
                i += 1
            bi += 1
        else:
            block_for_message[i] = bi
            bi += 1
            i += 1

    prefix = t_sys
    ok = True
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        b = block_for_message[i]
        fast_prompt = t_sys + sum(t_blocks[:b]) + t_gen
        fast_target = t_blocks[b] - t_gen
        # naive
        prompt_s = tok.apply_chat_template(messages[:i], tools=tools, tokenize=False,
                                          add_generation_prompt=True)
        full_s = tok.apply_chat_template(messages[:i + 1], tools=tools, tokenize=False,
                                         add_generation_prompt=False)
        pids = tok(prompt_s, add_special_tokens=False)["input_ids"]
        fids = tok(full_s, add_special_tokens=False)["input_ids"]
        if fids[: len(pids)] != pids:
            print("!! prefix property broken at msg", i, label)
            ok = False
            continue
        if fast_prompt != len(pids) or fast_target != len(fids) - len(pids):
            print("!! TOKEN MATH MISMATCH msg", i, label,
                  "fast", fast_prompt, fast_target, "naive", len(pids), len(fids) - len(pids))
            ok = False
        pass
    return ok


from typing import Dict  # noqa: E402

results = {}
samples = []

# --- build a few conversations straight from the raw sources -------------
import pyarrow.parquet as pq  # noqa: E402

hermes = pq.read_table(HERE / "raw/hermes_nano.parquet").to_pylist()[:6]
for r in hermes:
    leading, events = N.parse_hermes_events(r.get("text") or "")
    msgs = N.hermes_events_to_messages(events)
    user = N.hermes_user_content(leading, r.get("task") or "")
    msgs.insert(0, {"role": "user", "content": user})
    N.ensure_system(msgs)
    samples.append(("hermes", msgs, [{"type": "function", "function": {"name": "terminal",
        "description": "run", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]))
    samples.append(("hermes-notools", msgs, []))

zrows = pq.read_table(HERE / "raw/zake7749_agent.parquet").to_pylist()
best = {}
for r in zrows:
    if r["id"] not in best or len(r["messages"]) > len(best[r["id"]]["messages"]):
        best[r["id"]] = r
for r in list(best.values())[:4]:
    msgs = []
    for m in r["messages"]:
        role = m.get("role")
        if role == "assistant":
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or tc
                calls.append({"name": fn["name"], "arguments": fn.get("arguments")})
            text = N.assistant_text(reasoning=m.get("reasoning"), content=m.get("content"), tool_calls=calls)
            msgs.append({"role": "assistant", "content": N.sanitize_assistant_content(text)})
        elif role == "tool":
            msgs.append({"role": "tool", "content": m.get("content") or ""})
        else:
            msgs.append({"role": role, "content": m.get("content") or ""})
    N.ensure_system(msgs)
    samples.append(("zake7749", msgs, N.normalize_tools(r.get("tools"))))

orows = pq.read_table(HERE / "raw/openswe_cc_swe_00.parquet").to_pylist()[:3]
for r in orows:
    msgs = []
    for m in r["messages"]:
        role = m.get("role")
        if role == "assistant":
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or tc
                calls.append({"name": fn["name"], "arguments": fn.get("arguments")})
            text = N.assistant_text(reasoning=m.get("reasoning_content"), content=m.get("content"), tool_calls=calls)
            msgs.append({"role": "assistant", "content": N.sanitize_assistant_content(text)})
        elif role == "tool":
            msgs.append({"role": "tool", "content": m.get("content") or ""})
        else:
            msgs.append({"role": role, "content": m.get("content") or ""})
    N.ensure_system(msgs)
    samples.append(("openswe", msgs, N.compress_tools(N.normalize_tools(r.get("tools")))))

nem = json.load(open(HERE / "probe_raw/nemotron.json"))
for r in nem[:2]:
    msgs = []
    for m in r["messages"]:
        role = m.get("role")
        if role == "assistant":
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or tc
                calls.append({"name": fn["name"], "arguments": fn.get("arguments")})
            text = N.assistant_text(reasoning=m.get("reasoning_content"), content=m.get("content"), tool_calls=calls)
            msgs.append({"role": "assistant", "content": N.sanitize_assistant_content(text)})
        elif role == "tool":
            msgs.append({"role": "tool", "content": m.get("content") or ""})
        else:
            msgs.append({"role": role, "content": m.get("content") or ""})
    N.ensure_system(msgs)
    samples.append(("nemotron", msgs, N.compress_tools(N.normalize_tools(r.get("tools")))))

# simple sources
import gzip  # noqa: E402

with open(HERE / "raw/trjxter_8000x.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 2:
            break
        r = json.loads(line)
        msgs = [{"role": "system", "content": N.DEFAULT_SYSTEM},
                {"role": "user", "content": r["input"]},
                {"role": "assistant", "content": r["output"]}]
        samples.append(("trjxter", msgs, []))

fails = 0
for label, msgs, tools in samples:
    if not check(msgs, tools, label):
        fails += 1
    else:
        print("ok", label, len(msgs), "msgs")

print()
print("FAILURES:", fails, "of", len(samples))
sys.exit(1 if fails else 0)
