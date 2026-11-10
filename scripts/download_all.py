"""Download all raw source files needed for dataset preparation.

Runs in background; writes a JSON index of local paths at the end.
"""
import json, os, sys, time
from pathlib import Path

from huggingface_hub import hf_hub_download

OUT = Path("/media/louhmiwaslost/DATA/AGENT_WORKSPACE/smallthinker_experiment/raw")
OUT.mkdir(parents=True, exist_ok=True)

JOBS = [
    # (repo_id, filename, repo_type, local_name)
    ("r0b0tlab/deepseek-hermes-reasoning-traces", "train.parquet", "dataset", "hermes_train.parquet"),
    ("zake7749/deepseek-v4-pro-agent-tool-calling-trajectory", "data/train-00000-of-00001.parquet", "dataset", "zake7749_agent.parquet"),
    ("trjxter/DeepSeek-V4-Pro-Reasoning-8000x", "train.jsonl", "dataset", "trjxter_8000x.jsonl"),
    ("blythet/deepseek-v4-pro-math-cot-1k", "train_core.jsonl", "dataset", "blythet_train_core.jsonl"),
    ("beyoru/Deepseek-v4-pro-max-distill-1500x", "output.jsonl", "dataset", "beyoru_1500x.jsonl"),
    ("m-a-p/CodeFeedback-Filtered-Instruction", "CodeFeedback-Filtered-Instruction.jsonl", "dataset", "codefeedback.jsonl"),
    ("nvidia/Open-SWE-Traces", "data/claudecode/qwen38_flash_next/scale-swe/train-00000-of-00007.parquet", "dataset", "openswe_cc_swe_00.parquet"),
    ("nvidia/Open-SWE-Traces", "data/claudecode/qwen38_flash_next/scale-swe/train-00001-of-00007.parquet", "dataset", "openswe_cc_swe_01.parquet"),
    ("nvidia/Open-SWE-Traces", "data/claudecode/qwen38_flash_next/swe-rebench-v2/train-00000-of-00008.parquet", "dataset", "openswe_cc_rebench_00.parquet"),
    # Jackrong STEM shards (all 6) - Math is unusable but grab one shard to confirm
    ("Jackrong/DeepSeek-V4-Pro-Distilled-200K", "data/STEM/train-00000.jsonl.gz", "dataset", "jackrong_stem_00.jsonl.gz"),
    ("Jackrong/DeepSeek-V4-Pro-Distilled-200K", "data/STEM/train-00001.jsonl.gz", "dataset", "jackrong_stem_01.jsonl.gz"),
    ("Jackrong/DeepSeek-V4-Pro-Distilled-200K", "data/STEM/train-00002.jsonl.gz", "dataset", "jackrong_stem_02.jsonl.gz"),
    ("Jackrong/DeepSeek-V4-Pro-Distilled-200K", "data/STEM/train-00003.jsonl.gz", "dataset", "jackrong_stem_03.jsonl.gz"),
    ("Jackrong/DeepSeek-V4-Pro-Distilled-200K", "data/STEM/train-00004.jsonl.gz", "dataset", "jackrong_stem_04.jsonl.gz"),
    ("Jackrong/DeepSeek-V4-Pro-Distilled-200K", "data/STEM/train-00005.jsonl.gz", "dataset", "jackrong_stem_05.jsonl.gz"),
]

index = {}
for repo, fname, rtype, local in JOBS:
    dest = OUT / local
    if dest.exists() and dest.stat().st_size > 0:
        print(f"SKIP {local} (exists {dest.stat().st_size/1e6:.1f} MB)", flush=True)
        index[local] = str(dest)
        continue
    t0 = time.time()
    for attempt in range(4):
        try:
            p = hf_hub_download(repo, fname, repo_type=rtype)
            break
        except Exception as e:
            print(f"  retry {attempt+1} {local}: {e!r}", flush=True)
            if attempt == 3:
                p = None
            time.sleep(5)
    if p is None:
        print(f"FAIL {local}", flush=True)
        continue
    # move/symlink into raw dir
    os.symlink(os.path.realpath(p), dest) if not dest.exists() else None
    sz = dest.stat().st_size / 1e6
    index[local] = str(dest)
    print(f"OK {local} {sz:.1f} MB in {time.time()-t0:.0f}s", flush=True)

with open(OUT / "index.json", "w") as f:
    json.dump(index, f, indent=2)
print("DONE", len(index), "files", flush=True)
