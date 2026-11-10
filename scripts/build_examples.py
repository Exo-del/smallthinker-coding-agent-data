"""Expand the shipped corpus into tokenised training arrays.

Runs inside Kaggle after cloning the dataset repository.  Every assistant turn
that survived the preparation-time token filter becomes one training sequence:

    input_ids = tokens( render(messages[:i]) + generation prompt + target )
    labels    = -100 everywhere except the target span

The target span is the *tail* of the sequence, which lets the trainer compute
the LM head over the target span only (``logits_to_keep``) instead of the full
151,936-entry vocabulary over the whole window.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sftnorm as N  # noqa: E402


def iter_conversations(data_dir: Path, split: str):
    for shard in sorted((data_dir / split).glob("part-*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def build_split(tok, data_dir: Path, split: str, out_dir: Path, seq_len: int,
                prefix: str, log_every: int = 2000) -> dict:
    convs = list(iter_conversations(data_dir, split))
    n_turns = sum(len(c.get("meta", {}).get("turns", [])) for c in convs)
    print(f"[{split}] {len(convs)} conversations / {n_turns} turns", flush=True)

    ids_out = out_dir / f"{prefix}_input_ids.npy"
    lab_out = out_dir / f"{prefix}_labels.npy"
    len_out = out_dir / f"{prefix}_lengths.npy"
    out_dir.mkdir(parents=True, exist_ok=True)

    # exact upper bound: one row per assistant turn (plus slack for re-runs)
    n_max = n_turns + 64
    x = np.lib.format.open_memmap(ids_out, mode="w+", dtype=np.int32, shape=(n_max, seq_len))
    y = np.lib.format.open_memmap(lab_out, mode="w+", dtype=np.int32, shape=(n_max, seq_len))
    lengths = np.zeros(n_max, dtype=np.int32)

    n = 0
    skipped_long = 0
    skipped_bad = 0
    target_tokens = 0
    total_tokens = 0
    per_source = {}
    t_gen = len(tok(N.GEN_PROMPT, add_special_tokens=False)["input_ids"])

    for conv in convs:
        messages = conv["messages"]
        tools = conv.get("tools") or None
        source = conv.get("source", "?")
        for turn in conv.get("meta", {}).get("turns", []):
            i = turn["i"]
            if i >= len(messages):
                skipped_bad += 1
                continue
            prompt = tok.apply_chat_template(messages[:i], tools=tools, tokenize=False,
                                             add_generation_prompt=True)
            full = tok.apply_chat_template(messages[: i + 1], tools=tools, tokenize=False,
                                           add_generation_prompt=False)
            pids = tok(prompt, add_special_tokens=False)["input_ids"]
            fids = tok(full, add_special_tokens=False)["input_ids"]
            if fids[: len(pids)] != pids:
                skipped_bad += 1
                continue
            if len(fids) > seq_len:
                skipped_long += 1
                continue
            if len(fids) - len(pids) <= 0:
                skipped_bad += 1
                continue
            if n >= n_max:
                break
            labels = [-100] * len(fids)
            for j in range(len(pids), len(fids)):
                labels[j] = fids[j]
            x[n, : len(fids)] = fids
            y[n, : len(fids)] = labels
            lengths[n] = len(fids)
            target_tokens += len(fids) - len(pids)
            total_tokens += len(fids)
            st = per_source.setdefault(source, {"examples": 0, "target_tokens": 0, "total_tokens": 0})
            st["examples"] += 1
            st["target_tokens"] += len(fids) - len(pids)
            st["total_tokens"] += len(fids)
            n += 1
        if n and n % log_every < 32:
            print(f"[{split}] {n} examples", flush=True)

    x.flush()
    y.flush()
    del x, y
    # Rows past n were never written; the trainer filters them out by
    # dropping every zero-length entry, and because writes are sequential the
    # surviving indices still line up with the memmaps.
    np.save(len_out, lengths)
    print(f"[{split}] wrote {n} examples "
          f"(skipped long={skipped_long}, bad={skipped_bad})", flush=True)
    return {
        "split": split,
        "examples": n,
        "skipped_too_long": skipped_long,
        "skipped_invalid": skipped_bad,
        "target_tokens": target_tokens,
        "total_tokens": total_tokens,
        "per_source": per_source,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="repo directory containing data/")
    ap.add_argument("--out", required=True, help="directory for the .npy arrays")
    ap.add_argument("--seq-len", type=int, default=3072)
    ap.add_argument("--model", default="Tiiny/SmallThinker-4BA0.6B-Instruct")
    ap.add_argument("--splits", default="train,valid")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    data_dir = Path(args.data_dir)
    # accept either the repo root (which contains data/) or data/ itself
    if (data_dir / "data").is_dir():
        data_dir = data_dir / "data"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"seq_len": args.seq_len, "model": args.model, "splits": {}}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        prefix = "train" if split == "train" else "valid"
        summary["splits"][split] = build_split(tok, data_dir, split, out_dir,
                                               args.seq_len, prefix)
    with open(out_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
