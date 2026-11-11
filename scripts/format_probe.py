"""Measure the *format* behaviour of an adapter: does it open and close a think
block correctly?

Why this exists: v1 fitted the corpus (teacher-forced CE on real examples: tool
1.039 -> 0.638) yet never emitted `<think>` - p was ~0 and rank ~10,000 at every
prompt, corpus prompts included.  The tags are one token per turn, ~0.4 % of the
supervised tokens, so a plain mean loss leaves them untrained while
`<tool_call>` was already the base model's argmax.

A weighted loss plus LoRA on `lm_head` fixed the logits, but the two think tags
are then in a near-tie, and which one wins depends on the loss mass rather than
on the context (run #1 opened with `</think>`, run #2 opened with `<think>` and
*closed* with `<think>`).  So the probe measures both decisions:

  OPEN   at the assistant-start position: p / rank of `<think>` and `</think>`,
         plus a greedy generation that should start with `<think>`.
  CLOSE  after a seeded reasoning span: p / rank of `</think>`, plus a short
         generation that should start with `</think>`.

Usage:
  python format_probe.py --model /kaggle/working/model --adapter <dir> \\
      --out /kaggle/working/probe.json [--corpus <dir>] [--max-new 140]
"""

import argparse
import glob
import gzip
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

THINK = "<" + "think>"
THINK_END = "</" + "think>"
TOOL_CALL = "<tool_call>"
TOOL_END = "</" + "tool_call>"


def extract_reasoning(text, max_chars=420):
    """Pull the reasoning out of a corpus turn so the close probe is seeded with
    in-distribution text instead of a synthetic sentence.

    Returns "" when the turn has no think block, which makes the caller skip the
    close probe rather than measure a bogus position.
    """
    i = text.find(THINK)
    if i < 0:
        return ""
    j = text.find(THINK_END, i + len(THINK))
    body = text[i + len(THINK): j if j >= 0 else None].strip()
    if len(body) > max_chars:
        cut = body[:max_chars]
        body = cut.rsplit(" ", 1)[0] if " " in cut else cut
    return body


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default="")
    p.add_argument("--out", default="/kaggle/working/format_probe.json")
    p.add_argument("--corpus", default="")
    p.add_argument("--max-new", type=int, default=140)
    p.add_argument("--close-new", type=int, default=80)
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
    ("canonical+tool", CANON, "Read the file config.yaml and tell me the database host.", True,
     "We need the database host from config.yaml. Reading the file will show it."),
    ("short+tool", "You are a precise software engineering assistant.",
     "Read the file config.yaml and tell me the database host.", True,
     "The user wants the database host. I should read config.yaml first."),
    ("noprompt+tool", None, "Read the file app.py and summarise what it does.", True,
     "I need to see app.py before summarising it."),
    ("canonical+math", CANON, "A shop sells 17 boxes of 23 apples. How many apples?", False,
     "17 boxes of 23 apples: 17 * 23 = 391."),
    ("canonical+code", CANON, "Write a Python function that returns the nth Fibonacci number.", False,
     "An iterative loop over two accumulators gives the nth Fibonacci number."),
]


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = {name: tok(t, add_special_tokens=False)["input_ids"][0]
           for name, t in [("think", THINK), ("think_end", THINK_END),
                           ("tool_call", TOOL_CALL), ("tool_end", TOOL_END),
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

    def distribute(text_in):
        enc = tok(text_in, return_tensors="pt", add_special_tokens=False).to(args.device)
        with torch.no_grad():
            out = model(**enc, use_cache=False, return_dict=True)
        logits = out.logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        stats = {name: {"p": round(float(probs[tid]), 6),
                        "rank": int((logits > logits[tid]).sum()) + 1}
                 for name, tid in ids.items()}
        top = torch.topk(probs, 4)
        top4 = [(tok.decode([int(i)]), round(float(p), 4))
                for p, i in zip(top.values.tolist(), top.indices.tolist())]
        return enc, stats, top4

    def generate(enc, n):
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=n, do_sample=False,
                                 eos_token_id=[151643, 151645])
        return tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)

    def open_probe(label, prompt):
        enc, stats, top4 = distribute(prompt)
        text = generate(enc, args.max_new)
        print("=" * 74)
        print(f"OPEN  {label}: p(think)={stats['think']['p']:.4f} rank={stats['think']['rank']} "
              f"| p(</think>)={stats['think_end']['p']:.4f} rank={stats['think_end']['rank']} "
              f"| p(tool_call)={stats['tool_call']['p']:.5f}")
        print("top4:", top4)
        print(text[:420])
        return {"label": label, "kind": "open", "stats": stats, "top4": top4, "text": text}

    def close_probe(label, prompt, seed_reasoning):
        seeded = prompt + THINK + "\n" + seed_reasoning + "\n"
        enc, stats, top4 = distribute(seeded)
        text = generate(enc, args.close_new)
        print("=" * 74)
        print(f"CLOSE {label}: p(</think>)={stats['think_end']['p']:.4f} "
              f"rank={stats['think_end']['rank']} | p(think)={stats['think']['p']:.4f} "
              f"rank={stats['think']['rank']} | p(tool_call)={stats['tool_call']['p']:.4f}")
        print("top4:", top4)
        print(text[:300])
        return {"label": label, "kind": "close", "stats": stats, "top4": top4,
                "text": text, "seeded_prompt_tail": seeded[-160:]}

    results = []
    for label, system, user, with_tools, seed in CASES:
        prompt = build(system, user, with_tools)
        results.append(open_probe(label, prompt))
        results.append(close_probe(label, prompt, seed))

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
                    label = f"corpus #{found}"
                    results.append(open_probe(label, prompt))
                    # Seed the close probe with the turn's *real* reasoning so the
                    # in-distribution control measures both halves of the format.
                    seed = extract_reasoning(conv["messages"][i].get("content", "")
                                             if i < len(conv["messages"]) else "")
                    if seed:
                        results.append(close_probe(label, prompt, seed))
                    found += 1
                    if found >= 2:
                        break
            if found >= 2:
                break

    opens = {r["label"]: {"p_think": r["stats"]["think"]["p"],
                          "rank_think": r["stats"]["think"]["rank"],
                          "p_think_end": r["stats"]["think_end"]["p"],
                          "p_tool_call": r["stats"]["tool_call"]["p"],
                          "starts_with_think": r["text"].startswith(THINK)}
             for r in results if r["kind"] == "open"}
    closes = {r["label"]: {"p_think_end": r["stats"]["think_end"]["p"],
                           "rank_think_end": r["stats"]["think_end"]["rank"],
                           "p_think": r["stats"]["think"]["p"],
                           "starts_with_think_end": r["text"].startswith(THINK_END),
                           "starts_with_think": r["text"].startswith(THINK)}
              for r in results if r["kind"] == "close"}
    # a turn is well formed only if the opener is right AND the closer follows it
    verdict = {k: {"opens_ok": opens[k]["starts_with_think"],
                   "closes_ok": closes.get(k, {}).get("starts_with_think_end")}
               for k in opens}
    # only prompts that got *both* probes count; `closes_ok is None` means the
    # close probe never ran (a corpus turn with no reasoning to seed from), which
    # must not be reported as a failure.
    measured = [k for k, v in verdict.items() if v["closes_ok"] is not None]

    with open(args.out, "w") as f:
        json.dump({"token_ids": ids, "summary": opens, "closes": closes,
                   "verdict": verdict, "measured": measured,
                   "results": results}, f, indent=1)
    print("\n=== SCORECARD ===")
    for k, v in verdict.items():
        state = "-" if v["closes_ok"] is None else v["closes_ok"]
        print(f"{k:>18}: opens={v['opens_ok']} closes={state}")
    n_open = sum(1 for v in verdict.values() if v["opens_ok"])
    n_ok = sum(1 for k in measured if verdict[k]["opens_ok"] and verdict[k]["closes_ok"])
    print(f"opens ok:          {n_open}/{len(verdict)}")
    print(f"fully well formed: {n_ok}/{len(measured)} (measured prompts)")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
