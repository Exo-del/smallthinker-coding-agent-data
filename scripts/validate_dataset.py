"""Validate the prepared corpus before it is shipped.

Checks every conversation in the emitted shards for:
  * the fast renderer still matches ``apply_chat_template`` byte for byte
  * the recorded target/prompt token counts still hold
  * think blocks are balanced and tool-call blocks contain valid JSON
  * tool results follow an assistant turn that actually called a tool
  * every called tool is declared in the tool list (when the source had one)

Run:  python3 validate_dataset.py --data-dir <repo>/data
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sftnorm as N  # noqa: E402


def iter_shards(data_dir: Path, split: str):
    for shard in sorted((data_dir / split).glob("part-*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield shard.name, json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", default="Tiiny/SmallThinker-4BA0.6B-Instruct")
    ap.add_argument("--splits", default="train,valid")
    ap.add_argument("--render-check", default="all", choices=["all", "sample"])
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    data_dir = Path(args.data_dir)

    errors = Counter()
    stats = Counter()
    length_hist = Counter()
    samples = []
    checked_render = 0

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        for shard_name, conv in iter_shards(data_dir, split):
            stats["conversations"] += 1
            stats["source_" + conv.get("source", "?")] += 1
            messages = conv["messages"]
            tools = conv.get("tools") or None
            declared = {t["function"]["name"] for t in (tools or [])}

            # --- renderer parity -----------------------------------------
            if args.render_check == "all" or len(samples) < 50:
                fast = N.render_conversation(messages, tools)
                ref = tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                              add_generation_prompt=False)
                checked_render += 1
                if fast != ref:
                    errors["render_mismatch"] += 1
                    if len(samples) < 3:
                        samples.append({"id": conv.get("id"), "why": "render_mismatch"})

            # --- structure ------------------------------------------------
            if not messages or messages[0].get("role") != "system":
                errors["missing_system"] += 1
            last_assistant_had_call = False
            calls = []
            for m in messages:
                role = m.get("role")
                content = m.get("content") or ""
                if role == "assistant":
                    # Structural parse: match the tool-call blocks with the
                    # same regex the canonicaliser uses, then count tags only
                    # outside matched blocks.  A literal " TInt" style tag
                    # inside a JSON argument is data, not structure.
                    blocks = list(N._CALL_BLOCK_RE.finditer(content))
                    rest = N._CALL_BLOCK_RE.sub(" ", content)
                    if rest.count(N.THINK_OPEN) != rest.count(N.THINK_CLOSE):
                        errors["unbalanced_think"] += 1
                    if rest.count(N.TOOL_CALL_OPEN) or rest.count(N.TOOL_CALL_CLOSE):
                        errors["unbalanced_tool_call"] += 1
                    for mm in blocks:
                        body = mm.group(1).strip()
                        if not body:
                            errors["empty_tool_call"] += 1
                            continue
                        try:
                            obj = json.loads(body)
                        except Exception:
                            errors["bad_tool_call_json"] += 1
                            continue
                        if not isinstance(obj, dict) or "name" not in obj or "arguments" not in obj:
                            errors["bad_tool_call_shape"] += 1
                            continue
                        if N.THINK_OPEN in json.dumps(obj, ensure_ascii=False):
                            stats["think_tag_in_payload"] += 1
                        calls.append(obj["name"])
                        if declared and obj["name"] not in declared:
                            errors["undeclared_tool"] += 1
                    last_assistant_had_call = bool(calls)
                    stats["assistant_messages"] += 1
                    stats["tool_call_blocks"] += len(calls)
                    if calls:
                        stats["assistant_with_calls"] += 1
                elif role == "tool":
                    stats["tool_messages"] += 1
                    if not last_assistant_had_call:
                        errors["tool_without_call"] += 1
                else:
                    calls = []

            # --- token accounting -----------------------------------------
            turns = conv.get("meta", {}).get("turns", [])
            if not turns:
                errors["no_turns"] += 1
            stats["kept_turns"] += len(turns)
            for turn in turns:
                i = turn["i"]
                if i >= len(messages):
                    errors["turn_index_out_of_range"] += 1
                    continue
                sys_block, blocks = N.render_blocks(messages, tools)
                t_sys = len(tok(sys_block, add_special_tokens=False)["input_ids"])
                t_gen = len(tok(N.GEN_PROMPT, add_special_tokens=False)["input_ids"])
                t_blocks = [len(tok(b, add_special_tokens=False)["input_ids"]) for b in blocks]
                # locate the block that holds message i
                idx = 1 if (messages and messages[0].get("role") == "system") else 0
                bi = 0
                before = 0
                found = None
                while idx < len(messages):
                    if messages[idx].get("role") == "tool":
                        run_start = idx
                        while idx < len(messages) and messages[idx].get("role") == "tool":
                            idx += 1
                        if run_start <= i < idx:
                            found = (bi, before)
                            break
                        before += t_blocks[bi]
                        bi += 1
                    else:
                        if idx == i:
                            found = (bi, before)
                            break
                        before += t_blocks[bi]
                        bi += 1
                        idx += 1
                if found is None:
                    errors["block_not_found"] += 1
                    continue
                b, before = found
                p_len = t_sys + before + t_gen
                t_len = t_blocks[b] - t_gen
                if p_len != turn["prompt_tokens"] or t_len != turn["target_tokens"]:
                    errors["token_count_mismatch"] += 1
                length_hist[min(16, (p_len + t_len) // 256)] += 1

    print("=== errors ===")
    if not errors:
        print("  none")
    for k, v in errors.most_common():
        print(f"  {k}: {v}")
    print()
    print("=== stats ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print()
    print("=== sequence length histogram (256-token buckets) ===")
    for bucket in sorted(length_hist):
        print(f"  {bucket * 256:5d}-{(bucket + 1) * 256:5d}: {length_hist[bucket]}")
    print()
    print("renderer parity checks:", checked_render)
    print("sample problems:", json.dumps(samples[:5], indent=2))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
