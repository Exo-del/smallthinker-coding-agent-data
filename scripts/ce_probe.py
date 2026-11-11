"""Teacher-forced cross-entropy on real corpus examples, base vs adapters.

Why: the format runs trade a massive loss weight on a handful of structural
tokens. That is the right trade for format, but it is not free - v1's tool CE on
real examples was 1.039 (base) -> 0.638 (adapter), and a 200x tag weight could
push the model to over-fit the tags at the cost of ordinary tokens.

This probe answers one question: did the format surgery make the model *worse*
at the corpus it was trained on? It reports mean CE over target tokens per
category, so the comparison is like-for-like with the diag2 numbers
(tool 1.0389 -> 0.6376, coding 0.5873 -> 0.4603, reasoning 1.1000 -> 0.9271).

Usage:
  python ce_probe.py --model /kaggle/working/model --corpus /kaggle/working/corpus \
      --adapters "base=" "v1=/kaggle/working/lora_v1" "v2=/kaggle/working/lora_v2" \
      --out /kaggle/working/ce_probe.json
"""

import argparse
import glob
import gzip
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", default="/kaggle/working/ce_probe.json")
    p.add_argument("--adapters", nargs="+", default=["base="],
                   help='entries of the form name=path (empty path means the base model)')
    p.add_argument("--per-cat", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=3072)
    p.add_argument("--categories", default="tool,reasoning,coding")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def collect(tok, corpus, categories, per_cat, seq_len):
    """First turn of the first `per_cat` conversations per category.

    Deterministic (files are read in sorted order), so every adapter is scored on
    exactly the same examples.
    """
    wanted = {c: [] for c in categories}
    for shard in sorted(glob.glob(os.path.join(corpus, "data/train/*.jsonl.gz"))):
        if all(len(v) >= per_cat for v in wanted.values()):
            break
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                conv = json.loads(line)
                cat = conv.get("category")
                if cat not in wanted or len(wanted[cat]) >= per_cat:
                    continue
                turns = conv.get("meta", {}).get("turns") or []
                if not turns:
                    continue
                i = turns[0]["i"]
                msgs = conv["messages"]
                if i >= len(msgs):
                    continue
                tools = conv.get("tools") or None
                prompt = tok.apply_chat_template(msgs[:i], tools=tools, tokenize=False,
                                                 add_generation_prompt=True)
                full = tok.apply_chat_template(msgs[: i + 1], tools=tools, tokenize=False,
                                               add_generation_prompt=False)
                pids = tok(prompt, add_special_tokens=False)["input_ids"]
                fids = tok(full, add_special_tokens=False)["input_ids"]
                if fids[: len(pids)] != pids or len(fids) - len(pids) <= 0:
                    continue
                if len(fids) > seq_len:
                    continue
                labels = [-100] * len(pids) + fids[len(pids):]
                wanted[cat].append({"input_ids": fids, "labels": labels,
                                    "source": conv.get("source", "?")})
    return wanted


@torch.no_grad()
def score(model, tok, examples, args):
    model.eval()
    out = {}
    for cat, items in examples.items():
        total, count, per = 0.0, 0, []
        for ex in items:
            ids = torch.tensor([ex["input_ids"]], device=args.device)
            labels = torch.tensor([ex["labels"]], device=args.device)
            logits = model(input_ids=ids, use_cache=False, return_dict=True).logits
            ce = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]).float(),
                                 labels[:, 1:].reshape(-1), ignore_index=-100,
                                 reduction="sum")
            n = int((labels[:, 1:] != -100).sum())
            total += float(ce)
            count += n
            per.append(round(float(ce) / max(n, 1), 4))
            del logits, ce
        out[cat] = {"ce": round(total / max(count, 1), 4), "target_tokens": count,
                    "per_example": per, "n": len(items)}
        print(f"    {cat:>10}: CE {out[cat]['ce']:.4f} over {count} tokens "
              f"({len(items)} examples)", flush=True)
    return out


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    categories = args.categories.split(",")

    print("collecting examples...", flush=True)
    examples = collect(tok, args.corpus, categories, args.per_cat, args.seq_len)
    for cat, items in examples.items():
        print(f"  {cat}: {len(items)} examples, "
              f"{sum(sum(1 for x in it['labels'] if x != -100) for it in items)} target tokens")

    print("loading base model...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16,
        attn_implementation="sdpa").to(args.device)

    results = {}
    for entry in args.adapters:
        name, _, path = entry.partition("=")
        print(f"\n=== {name} ({path or 'base model'}) ===", flush=True)
        model = base
        if path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, path)
            print("  adapter loaded", flush=True)
        results[name] = score(model, tok, examples, args)
        if path:
            base = model.unload()
            del model
            torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump({"per_cat": args.per_cat, "seq_len": args.seq_len,
                   "results": results}, f, indent=1)

    print("\n=== CE (lower is better) ===")
    names = list(results)
    print(f"{'category':>10} " + " ".join(f"{n:>10}" for n in names))
    for cat in categories:
        row = " ".join(f"{results[n][cat]['ce']:>10.4f}" for n in names)
        print(f"{cat:>10} {row}")
    if names[0] == "base":
        print("\n=== delta vs base ===")
        for cat in categories:
            row = " ".join(f"{results[n][cat]['ce'] - results['base'][cat]['ce']:>+10.4f}"
                           for n in names[1:])
            print(f"{cat:>10} {row}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
