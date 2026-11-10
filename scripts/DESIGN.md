# SmallThinker Coding-Agent LoRA SFT — Dataset Design

**Model:** `Tiiny/SmallThinker-4BA0.6B-Instruct` (4B total / 0.6B active MoE, 32 layers, hidden 1536, 12 heads / 2 KV, vocab 151,936, ctx 32K, `sliding_window_size=4096` with a mixed full/SWA layout).
**Hardware:** 2 × NVIDIA T4 (15 GiB each, Kaggle `NvidiaTeslaT4`).
**Goal:** teach the base model (a) *thinking* (`<think>…</think>` reasoning traces), (b) *tool calling* in the model's native `<tool_call>` / `<tool_response>` protocol, and (c) *coding* ability — while retaining DeepSeek V4 Pro STEM/reasoning strength.

---

## 1. Findings from the source survey (measured, not assumed)

### 1.1 Think / no-think mix

Measured with correctly-constructed tag literals (a bare `<` + `think>` concatenation — writing the literal tags directly into source files corrupts them).

| Source | think-any | think-at-start | tool-call | tool-response | Note |
|---|---|---|---|---|---|
| `r0b0tlab/deepseek-hermes-reasoning-traces` | **1.000** | 0.960 | **1.000** | 0.143 | **Native SmallThinker format already** |
| `trjxter/DeepSeek-V4-Pro-Reasoning-8000x` | **1.000** | 1.000 | 0 | 0 | Think is inline in `output` |
| `Jackrong/DeepSeek-V4-Pro-Distilled-200K` (Math + STEM) | **1.000** | 1.000 | 0 | 0 | Think inline; **no no-think rows at all** |
| `blythet/deepseek-v4-pro-math-cot-1k` | 0.000 | 0.000 | 0 | 0 | Reasoning is a *separate* `reasoning` column |
| `beyoru/Deepseek-v4-pro-max-distill-1500x` | 0.000 | 0.000 | 0 | 0 | Reasoning is a separate `reasoning` column |
| `nvidia/Nemotron-SFT-Agentic-v4` [tool_calling] | 0.000 | 0.000 | 0 | 0 | 100 % of assistant turns have `reasoning_content` |
| `nvidia/Open-SWE-Traces` v2.0/claudecode | 0.000 | 0.000 | 0 | 0 | 99.86 % of assistant turns have `reasoning_content` |
| `m-a-p/CodeFeedback-Filtered-Instruction` | 0.000 | 0.000 | 0 | 0 | No reasoning at all — pure coding answers |

**Conclusion — the answer to "is there mixed no-think + think?":**
The DeepSeek V4 Pro flagship corpus (`Jackrong/…Distilled-200K`) is **100 % think, 0 % no-think** — it is a pure reasoning corpus. The *only* no-think mass in the whole candidate set is CodeFeedback (plain coding answers, no reasoning) and the Open-SWE/Nemotron assistant turns whose `reasoning_content` is empty.

The practical consequence: **thinking must be synthesized for the tool-calling and coding sources** by wrapping their `reasoning_content` / `reasoning` field into `<think>…</think>`, and the no-think signal must be preserved deliberately (a slice of CodeFeedback stays no-think) so the model can answer directly instead of always burning tokens on a think block.

### 1.2 Length profile (SmallThinker tokenizer, target = assistant turn tokens)

| Source | p50 | p90 | p99 | max | usable at seq-2048? |
|---|---|---|---|---|---|
| DeepSeek-200K **Math** | 42,086 | 92,087 | 131,311 | 131,311 | ❌ **no** — even p10 is >11K |
| DeepSeek-200K **STEM** | 4,347 | 45,982 | 105,956 | 105,956 | ⚠️ only the short tail |
| `trjxter/…Reasoning-8000x` | 2,838 | 6,127 | 11,184 | 13,813 | ⚠️ after filter |
| `blythet/…math-cot-1k` | 2,091 | 7,510 | 32,134 | 32,134 | ⚠️ after filter |
| `beyoru/…max-distill-1500x` | 2,015 | 11,898 | 48,482 | 48,482 | ⚠️ after filter |
| Hermes (assistant turns) | **63** | 245 | 1,743 | 6,134 | ✅ yes |
| Open-SWE (assistant turns) | 339 | 1,456 | 5,145 | 12,481 | ✅ mostly |
| Nemotron (assistant turns) | 44 | 745 | 1,450 | 2,409 | ✅ yes |
| CodeFeedback (answers) | 354 | 669 | 1,223 | 1,491 | ✅ yes |

**Conclusion:** the flagship Math split is **unusable** — its median answer is 42 K tokens, 20× our sequence length. The flagship STEM split is usable only through a hard length filter (it has a short tail: min 498, p50 4.3 K, and ~10 % of rows are under ~1.5 K tokens). The *right* DeepSeek V4 Pro reasoning source for this model is **`trjxter/DeepSeek-V4-Pro-Reasoning-8000x`** (8,014 rows, 100 % think, p50 2.8 K) plus the filtered short tail of the 200 K STEM set.

### 1.3 Protocol compatibility (the make-or-break detail)

`<tool_call>` and `</tool_call>` are **special tokens** in the SmallThinker vocab (ids 151657 / 151658). `<think>` is *not* — it tokenizes as ordinary text (`<th`,`ink`,`>`). So:

- Hermes is already in the model's **native** format: `<think>…</think>\n<tool_call>\n{json}\n</tool_call>\n<tool_response>…</tool_response>`. It needs almost no surgery.
- The chat template renders assistant tool calls as `<tool_call>\n{"name": "X", "arguments": {…}}\n</tool_call>` — note the **spaces after `:` and `,`** (Jinja `tojson`, insertion order preserved). The previous notebook's `assistant_target_text` emitted compact JSON (`{"name":"X","arguments":{…}}`), i.e. a **train/inference format mismatch**. This must be fixed: the target text must be produced by the *same* renderer the chat template uses, or better, by rendering the assistant turn through `apply_chat_template` itself.
- Nemotron/Open-SWE keep reasoning in `reasoning_content` and tool calls in structured `tool_calls`; the model needs them re-expressed in the native text protocol.

### 1.4 Open-SWE split choice

`nvidia/Open-SWE-Traces` v2.0 has 5 framework configs × ~100 K rows each (~7 GB per framework). The `claudecode` config with `qwen38_flash_next` teacher is used here — it has the richest tool set (`Bash`, `Read`, `Edit`, …) and `resolved` flags for quality filtering. Only the `scale-swe` + `swe-rebench-v2` shards for `claudecode` are downloaded (not all 98 K rows).

---

## 2. Mixture design

**Principle: tool calling first.** The base model has neither thinking nor tool calling, so the dominant axis of the mixture is *agentic tool use*, with reasoning/STEM as the second axis and coding as the third. Every source is normalised into the same native ChatML dialect, and each source keeps its own **system prompt** so the model learns to condition tool use on the available tool list.

| # | Source | Role | Cap (rows) | Why |
|---|---|---|---|---|
| 1 | `r0b0tlab/deepseek-hermes-reasoning-traces` | **tool-call backbone + thinking** | 16,431 (all) | Native `<tool_call>`/`<tool_response>` format, 100 % think, multi-turn, p50 turn 63 tok. Zero conversion risk. |
| 2 | `zake7749/deepseek-v4-pro-agent-tool-calling-trajectory` | **DeepSeek V4 Pro agentic tool use** | 2,510 (all) | Actual DeepSeek V4 Pro: 100 % reasoning, parallel tool calls (39 %), multi-step (98.6 %), score-filtered. |
| 3 | `nvidia/Nemotron-SFT-Agentic-v4` [tool_calling] | **tool-call breadth** | 2,500 conversations | 10.7 GB of diverse API schemas; every row has real tool calls + tool results + reasoning. |
| 4 | `nvidia/Open-SWE-Traces` v2.0/claudecode | **coding agent, real repo trajectories** | 3,000 trajectories (resolved-first) | Genuine software-engineering loops with `Bash`/`Read`/`Edit`, `reasoning_content`, `resolved` label. |
| 5 | `trjxter/DeepSeek-V4-Pro-Reasoning-8000x` | **DeepSeek V4 Pro reasoning/STEM** | 8,014 (all, length-filtered) | 100 % think, p50 2.8 K tokens — the only DeepSeek V4 Pro set that fits a 2 K window at scale. |
| 6 | `Jackrong/DeepSeek-V4-Pro-Distilled-200K` [STEM] | **DeepSeek V4 Pro STEM** | length-filtered tail (~3–5 K tok) | Flagship STEM corpus; only rows under the token cap survive. Math split dropped entirely. |
| 7 | `blythet/deepseek-v4-pro-math-cot-1k` | **math CoT** | 1,008 (all) | Clean DeepSeek V4 Pro math reasoning; `reasoning` + `answer` columns. |
| 8 | `beyoru/Deepseek-v4-pro-max-distill-1500x` | **max-effort reasoning** | 1,499 (all) | `reasoning_effort=max` traces incl. a math split. |
| 9 | `m-a-p/CodeFeedback-Filtered-Instruction` | **coding, no-think** | 8,000 | Keeps a no-think slice alive so the model can answer coding questions directly. |

### 2.1 Target token mixture (by assistant-target tokens, after filtering)

| Category | Target share | Mechanism |
|---|---|---|
| Tool-calling (agentic) | **~50 %** | Hermes + zake7749 + Nemotron + Open-SWE |
| Reasoning / STEM / math | **~30 %** | trjxter + Jackrong STEM + blythet + beyoru |
| Coding, no-think | **~20 %** | CodeFeedback (+ Open-SWE code turns) |

---

## 3. Normalisation contract (the output schema)

Every example is emitted as one JSON object per line (zstd-compressed JSONL):

```json
{
  "id": "hermes:seed_id:turn_no",
  "source": "hermes",
  "messages": [
    {"role": "system",    "content": "…source-specific system prompt…"},
    {"role": "user",      "content": "…"},
    {"role": "assistant", "content": "<think>…</think>\n<tool_call>…</tool_call>"},
    {"role": "tool",      "content": "…"}
  ],
  "tools": [{"type": "function", "function": {"name": "…", "description": "…", "parameters": {…}}}],
  "meta": {"think": true, "has_tool_call": true, "target_tokens": 123}
}
```

Rules:
1. **Assistant thinking is always inline `<think>…</think>`** inside `content`, followed by the answer and/or `<tool_call>` blocks — exactly the SmallThinker convention.
2. **Tool calls are inline text** `<tool_call>\n{"name": …, "arguments": …}\n</tool_call>` (matching the template's spacing) **and** mirrored into a structured `tool_calls` field is *not* used — we keep one canonical representation (inline text) to avoid train/inference drift. Tool *results* are `{"role":"tool","content":…}` messages, which the template renders as `<tool_response>…</tool_response>`.
3. **Tools schema** is attached only where the source actually provides it (Nemotron, Open-SWE, zake7749). Hermes has no schemas → it gets a synthesised schema from the tool names observed in its own `<tool_call>` blocks, so the model still learns "look at the tool list, call a tool".
4. **Assistant-only loss** is applied by the Kaggle trainer, not by the prep script; the prep script only guarantees clean role ordering.
5. **Dedup** by SHA-1 of the normalised user prompt + first 256 chars of the target.

---

## 4. Tokenisation & training plan (Kaggle)

- **Sequence length 2048** (smoke-test 2048 → 1536 → 1024 if OOM). Rationale: p90 of the tool-call sources is ≤ 1.5 K, and the reasoning sources are capped to fit. `sliding_window_size=4096` means 2048 sits entirely inside the SWA window for most layers — no position-embedding extrapolation.
- **Batch 1 per GPU, grad-accum 8** → effective batch 16 sequences. Micro-batch 2 at 2048 OOMs (known constraint).
- **LoRA r=8, α=16, dropout 0.05** on `q_proj,k_proj,v_proj,o_proj`, fp16.
- **`logits_to_keep`**: the model's `forward` supports it, so the trainer computes logits only over the target span instead of the full 2048×151,936 — a large memory and compute saving on a 15 GiB T4.
- **DDP over 2 GPUs**, one full replica each; checkpoint every 100 optimizer steps; resumable across Kaggle sessions.
- **Manual activation checkpointing.** The model advertises
  `supports_gradient_checkpointing = False`, but only because
  `SmallThinkerModel.forward` never threads `self._gradient_checkpointing_func`
  into the decoder layers. `train_smallthinker.py` wraps each layer in
  `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` instead, which
  is equivalent and drops activation memory from O(n_layers × seq) to
  O(seq) + one layer's recompute. Measured on a T4: without it, 2048 tokens
  with a 512-token target peaks at 13.49 GiB and 2048/1024 OOMs; with it,
  3072 tokens fits.

---

## 5. Ship format & pipeline

```
local (this box)                      git                      Kaggle
─────────────────────────────    ─────────────────────    ───────────────────────
prepare_data.py
  ├ normalise 9 sources
  ├ filter by target tokens
  ├ dedup
  └ write data/*.jsonl.zst
                              ──►  Exo-del/smallthinker-coding-agent-data
                                      (data + manifest.json + README)
                                                             clone_data.py
                                                             ├ load jsonl.zst
                                                             ├ build assistant-turn examples
                                                             ├ tokenize → npy arrays
                                                             └ train_smallthinker.py (DDP LoRA)
```

**Why git and not a Kaggle dataset:** the previous run died with zero logs and zero outputs — the failure point was the notebook's own dataset streaming. Shipping a *prepared, pre-filtered, byte-stable* artefact removes every network/HF dependency from the training notebook except the base model itself.

**Repo layout**

```
smallthinker-coding-agent-data/
  README.md                 provenance, licences, how to regenerate
  manifest.json             per-source counts, token stats, think/tool ratios
  data/
    train.jsonl.zst
    valid.jsonl.zst
  scripts/
    prepare_data.py         the generator (committed for reproducibility)
    build_examples.py       turn-expansion + tokenisation, run inside Kaggle
```

---

## 6. Licences & attribution

| Source | Licence |
|---|---|
| `r0b0tlab/deepseek-hermes-reasoning-traces` | check repo card |
| `zake7749/deepseek-v4-pro-agent-tool-calling-trajectory` | check repo card |
| `nvidia/Nemotron-SFT-Agentic-v4` | NVIDIA Open Model License |
| `nvidia/Open-SWE-Traces` | NVIDIA Open Model License |
| `trjxter/DeepSeek-V4-Pro-Reasoning-8000x` | check repo card |
| `Jackrong/DeepSeek-V4-Pro-Distilled-200K` | mixed CC-BY-4.0 / CC-BY-SA-4.0 (per release manifest) |
| `blythet/deepseek-v4-pro-math-cot-1k` | check repo card |
| `beyoru/Deepseek-v4-pro-max-distill-1500x` | check repo card |
| `m-a-p/CodeFeedback-Filtered-Instruction` | check repo card |

Every row is attributed in `manifest.json` with its source id, and the repo README lists the upstream URLs. Sources with non-commercial or share-alike terms are flagged so the derived dataset carries the same terms.
