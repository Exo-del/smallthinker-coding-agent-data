# SmallThinker coding-agent SFT corpus

A prepared, pre-filtered, byte-stable training corpus for teaching
`Tiiny/SmallThinker-4BA0.6B-Instruct` three things it does not have out of the
box:

1. **thinking** — inline `<think> … </think>` reasoning traces,
2. **tool calling** — the model's native `<tool_call>{json}</tool_call>` /
   `<tool_response>…</tool_response>` protocol,
3. **coding** — DeepSeek V4 Pro style code answers.

SFT only: every row is a ready-to-train conversation. The mixture is balanced
by **assistant-target tokens**, not by row count, so very long reasoning traces
do not drown the short agentic turns.

| Category | Target-token share | Sources |
|---|---|---|
| Tool calling (agentic) | **49.2 %** | Hermes, zake7749, Nemotron, Open-SWE |
| Reasoning / STEM / math | **30.1 %** | trjxter, Jackrong-STEM, blythet, beyoru |
| Coding, no-think | **20.7 %** | CodeFeedback |

**31,881 conversations · 39,754 assistant turns · 19.31 M target tokens ·
29.43 M prompt tokens · 37 MB gzipped (16 train shards + 1 valid shard).**

---

## Layout

```
data/train/part-000.jsonl.gz … part-015.jsonl.gz   31,242 conversations
data/valid/part-000.jsonl.gz                          639 conversations
manifest.json      exact per-source counts, token stats, licences, config
DESIGN.md          why the mixture looks the way it does (measured findings)
scripts/           the pipeline that produced this corpus + the trainer
```

Each line of a shard is one JSON object:

```json
{
  "id": "hermes:trace_052687",
  "source": "hermes",
  "category": "tool",
  "messages": [
    {"role": "system",    "content": "…source-specific system prompt…"},
    {"role": "user",      "content": "…"},
    {"role": "assistant", "content": "<think>…</think>\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"api.py\"}}\n</tool_call>"},
    {"role": "tool",      "content": "…tool output…"}
  ],
  "tools": [{"type": "function", "function": {"name": "read_file", "description": "…", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}],
  "meta": {
    "turns": [{"i": 2, "prompt_tokens": 731, "target_tokens": 254, "think": true, "tool_call": true}],
    "dedup_key": "…"
  }
}
```

`meta.turns` lists the assistant turns that fit the window. A conversation is
truncated at the last usable turn, so the trainer only ever sees prompt+target
pairs that fit `max_seq_len`.

## Format contract

* **Thinking is inline** inside `content`, always first:
  `<think>\n…\n</think>`, then prose, then tool calls.
* **Tool calls are inline text**, exactly as the chat template emits them:
  `<tool_call>\n{"name": "…", "arguments": {…}}\n</tool_call>` — default
  `json.dumps` spacing, insertion-ordered keys.
* **Tool results** are `{"role": "tool"}` messages; the template merges runs of
  them into one user block with one `<tool_response>…</tool_response>` section
  each.
* `<tool_call>` and `</tool_call>` are single special tokens (151657/151658);
  `<think>` is plain text. Assistant content never starts with whitespace, so
  `tokens(prompt)` is always a strict prefix of `tokens(prompt + target)`.

## Limits

* `max_seq_len = 3072`, `max_target_tokens = 2560`, `min_prompt_tokens = 24`.
* Target tokens p50 = 304, p90 = 1193, max 2559.
* Validation split is 2 % (hash-based, stable).

## Sources and licences

| Source | Rows used | Licence |
|---|---|---|
| [`r0b0tlab/deepseek-hermes-reasoning-traces`](https://huggingface.co/datasets/r0b0tlab/deepseek-hermes-reasoning-traces) | 12,959 | apache-2.0 |
| [`zake7749/deepseek-v4-pro-agent-tool-calling-trajectory`](https://huggingface.co/datasets/zake7749/deepseek-v4-pro-agent-tool-calling-trajectory) | 284 | not listed upstream |
| [`nvidia/Nemotron-SFT-Agentic-v4`](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v4) | 2,040 | NVIDIA Open Model License |
| [`nvidia/Open-SWE-Traces`](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) | 1,359 | NVIDIA Open Model License |
| [`trjxter/DeepSeek-V4-Pro-Reasoning-8000x`](https://huggingface.co/datasets/trjxter/DeepSeek-V4-Pro-Reasoning-8000x) | 2,170 | apache-2.0 |
| [`Jackrong/DeepSeek-V4-Pro-Distilled-200K`](https://huggingface.co/datasets/Jackrong/DeepSeek-V4-Pro-Distilled-200K) (STEM only) | 2,283 | CC-BY-4.0 / CC-BY-SA-4.0 per row |
| [`blythet/deepseek-v4-pro-math-cot-1k`](https://huggingface.co/datasets/blythet/deepseek-v4-pro-math-cot-1k) | 281 | cc-by-sa-4.0 |
| [`beyoru/Deepseek-v4-pro-max-distill-1500x`](https://huggingface.co/datasets/beyoru/Deepseek-v4-pro-max-distill-1500x) | 336 | apache-2.0 |
| [`m-a-p/CodeFeedback-Filtered-Instruction`](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) | 10,169 | apache-2.0 |

Derived-work obligations from the share-alike sources carry over to this
corpus. Jackrong's Math split was dropped entirely (p50 target 42 k tokens —
it cannot fit any reasonable window).

## Validation

`scripts/validate_dataset.py` re-checks every emitted row:

* fast renderer vs. `tokenizer.apply_chat_template` — **31,881/31,881 byte-identical**
* recorded prompt/target token counts — **0 mismatches**
* tag balance (structural parse), tool-call JSON, declared-tool coverage,
  orphan tool results — **0 errors**

## Regeneration

```bash
# 1. fetch the raw source dumps into raw/ (see scripts/download_all.py)
# 2. build
python3 scripts/prepare_data.py --out . --no-cache
# 3. prove it
python3 scripts/validate_dataset.py --data-dir data --render-check all
```

`prepare_data.py` needs the SmallThinker tokenizer (downloaded automatically)
and ~7 GB of RAM. It takes ~10 minutes for the full corpus.

## Training

**Environment pin (important).** The model's remote code is transformers 4.x
code: it imports `HybridCache`, `LossKwargs` and `ROPE_INIT_FUNCTIONS["default"]`,
all removed in transformers 5.x. Kaggle ships 5.20.2, so the notebooks first
install:

```
transformers==4.53.2  huggingface_hub==0.36.2  tokenizers==0.21.4  peft==0.17.1
```

and load the weights into a **dot-free local directory**
(`snapshot_download(..., local_dir="/kaggle/working/model")`) — the repo id
`4BA0.6B` breaks the dynamic-module path in 4.x. Rendering was verified
byte-identical between 4.53.2 and 5.20.2 across 150 conversations.

See `DESIGN.md` §4. Short version: LoRA r=8 / α=16 on q,k,v,o, fp16, batch 1
per GPU × grad-accum 8, SDPA attention, `logits_to_keep` restricted to the
target span. The trainer lives in `scripts/train_smallthinker.py` and the
Kaggle notebooks are generated by `scripts/make_notebooks.py`.
