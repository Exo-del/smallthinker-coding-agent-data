"""Measure whether an adapter emits the structural tokens it was trained on.

The v1 run fitted the corpus well (teacher-forced CE on real examples: tool
1.039 -> 0.638, coding 0.587 -> 0.460, reasoning 1.100 -> 0.927) but still could
not emit `<think>`: p(`<think>`) was ~0 and its rank ~10,000 at every prompt we
tried, canonical system prompt and real corpus prompt included.  The tags are
one token per turn, ~0.4 % of the supervised tokens, so a plain mean loss barely
moves them - while `<tool_call>` was already the base model's top choice.

This probe is the acceptance test for that fix.  It reports, per prompt, the
probability and *rank* of `<think>`, `</think>`, `<tool_call>`, plus a short
greedy generation, and writes everything to JSON.

Usage:
  python format_probe.py --model /kaggle/working/model --adapter <dir> \
      --out /kaggle/working/probe.json [--corpus <dir>]
"""

import argparse
import glob
import gzip
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default="")
    p.add_argument("--out", default="/kaggle/working/format_probe.json")
    p.add_argument("--corpus", default="")
    p.add_argument("--max-new", type=int, default=140)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


TOOLS = [{"type": "function", "function": {
    "name": "read_file",
    "description": "Read the contents of a file from the working directory.",
    "parameters": {"type": "object",
                   "properties": {"path": {"type": "string",
                                           "description": "Path of the file to read."}},
                   "required": ["path"]}}}]

CANON = ("You are a precise software engineering assistant. Understand the task, "
         "inspect relevant context, use available tools when they materially help, "
         "interpret tool results, make minimal correct changes, and verify important "
         "claims. Do not invent tool results. If no tool is needed, answer directly.")

CASES = [
    ("canonical+tool", CANON, "Read the file config.yaml and tell me the database host.", True),
    ("short+tool", "You are a precise software engineering assistant.",
     "Read the file config.yaml and tell me the database host.", True),
    ("noprompt+tool", None,
     "Read the file app.py and summarise what it does.", True),
    ("canonical+math", CANON, "A shop sells 17 boxes of 23 apples. How many apples?", False),
    ("canonical+code", CANON, "Write a Python function that returns the nth Fibonacci number.", False),
]


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = {name: tok(t, add_special_tokens=False)["input_ids"][0]
           for name, t in [("think", "<" + "think>"), ("think_end", "</" + "think>"),
                           ("tool_call", "<tool_call>"), ("tool_end", "</" + "tool_call>"),
                           ("im_end", "<|im_end|>")]}
    print("token ids:", ids)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16,
        attn_implementation="sdpa").to(args.device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        print("adapter:", args.adapter)
    model.eval()

    def build(system, user, with_tools):
        msgs = []
        if system is not None:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        return tok.apply_chat_template(msgs, tools=TOOLS if with_tools else None,
                                       tokenize=False, add_generation_prompt=True)

    def probe(label, prompt, max_new):
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
        with torch.no_grad():
            out = model(**enc, use_cache=False, return_dict=True)
            logits = out.logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
            stats = {}
            for name, tid in ids.items():
                stats[name] = {"p": round(float(probs[tid]), 6),
                               "rank": int((logits > logits[tid]).sum()) + 1}
            top = torch.topk(probs, 5)
            top5 = [(tok.decode([int(i)]), round(float(p), 4))
                    for p, i in zip(top.values.tolist(), top.indices.tolist())]
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 eos_token_id=[151643, 151645])
        text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
        print("=" * 74)
        print(f"{label}: p(think)={stats['think']['p']:.5f} rank={stats['think']['rank']} "
              f"p(tool_call)={stats['tool_call']['p']:.5f} rank={stats['tool_call']['rank']}")
        print("top5:", top5)
        print(text[:500])
        return {"label": label, "stats": stats, "top5": top5, "text": text}

    results = []
    for label, system, user, with_tools in CASES:
        results.append(probe(label, build(system, user, with_tools), args.max_new))

    if args.corpus:
        found = 0
        for shard in sorted(glob.glob(os.path.join(args.corpus, "data/train/*.jsonl.gz")))[:4]:
            with gzip.open(shard, "rt") as f:
                for line in f:
                    conv = json.loads(line)
                    if conv.get("category") != "tool" or not conv.get("meta", {}).get("turns"):
                        continue
                    i = conv["meta"]["turns"][0]["i"]
                    prompt = tok.apply_chat_template(
                        conv["messages"][:i], tools=conv.get("tools") or None,
                        tokenize=False, add_generation_prompt=True)
                    if len(tok(prompt, add_special_tokens=False)["input_ids"]) > 3000:
                        continue
                    results.append(probe(f"corpus #{found}", prompt, args.max_new))
                    found += 1
                    if found >= 2:
                        break
            if found >= 2:
                break

    summary = {r["label"]: {"p_think": r["stats"]["think"]["p"],
                            "rank_think": r["stats"]["think"]["rank"],
                            "p_tool_call": r["stats"]["tool_call"]["p"],
                            "starts_with_think": r["text"].startswith("<" + "think>")}
               for r in results}
    with open(args.out, "w") as f:
        json.dump({"token_ids": ids, "summary": summary, "results": results}, f, indent=1)
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:>18}: p(think)={v['p_think']:.5f} rank={v['rank_think']:>6} "
              f"starts_with_think={v['starts_with_think']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
