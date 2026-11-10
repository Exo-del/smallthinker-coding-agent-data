"""Generate the Kaggle notebooks (probe + full training) as .ipynb files."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "notebooks"
OUT.mkdir(exist_ok=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


NOTIFY_CELL = '''# push progress to ntfy so the run can be followed without opening Kaggle
import urllib.request

NTFY_URL = "https://ntfy.sh/amine-opencode-d86756a5"

def notify(msg, prio="default"):
    try:
        req = urllib.request.Request(
            NTFY_URL, data=msg.encode("utf-8"),
            headers={"Title": "SmallThinker Kaggle", "Priority": prio})
        urllib.request.urlopen(req, timeout=8).read()
        print("[ntfy]", msg)
    except Exception as exc:
        print("[ntfy failed]", exc)

notify("notebook started")
'''


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13.16"},
            "accelerator": "GPU T4 x2",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ==========================================================================
# 1. memory / environment probe
# ==========================================================================
probe = notebook([
    md("""# SmallThinker-4BA0.6B LoRA - environment and memory probe

Purpose: the previous Kaggle run of this experiment died with **zero logs and
zero outputs** (a Kaggle platform fault, not a code fault - see `LAUNCH_RECORD.md`).
Before spending any GPU quota on the real run we verify, on the actual
machine, that:

1. the notebook produces logs at all,
2. `Tiiny/SmallThinker-4BA0.6B-Instruct` loads and runs,
3. a LoRA forward/backward at our target sequence length fits in 15 GiB,
4. SDPA + `logits_to_keep` (the memory optimisations) actually work.

Nothing here is saved as a dataset; this notebook only prints measurements.
"""),
    code("""import os, sys, time, json, subprocess, shutil

print("python:", sys.version)
print("cwd:", os.getcwd())
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu{i}: {p.name}  mem={p.total_memory/1024**3:.2f} GiB  cc={p.major}.{p.minor}")
except Exception as e:
    print("torch import failed:", repr(e))

import subprocess
print("nvidia-smi:")
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
"""),
    code(NOTIFY_CELL),
    code("""# Tiiny/SmallThinker-4BA0.6B-Instruct is a transformers 4.x model: its remote
# code imports HybridCache, LossKwargs and ROPE_INIT_FUNCTIONS["default"], all of
# which transformers 5.x removed.  Pin the 4.x stack, and load the weights from a
# dot-free local directory because the repo id (4BA0.6B) breaks the dynamic
# module path in 4.x.
!pip install -q "transformers==4.53.2" "huggingface_hub==0.36.2" "tokenizers==0.21.4" "peft==0.17.1" 2>&1 | tail -3
import peft, transformers, huggingface_hub
print("peft:", peft.__version__, "| transformers:", transformers.__version__, "| hub:", huggingface_hub.__version__)

from huggingface_hub import snapshot_download
import pathlib
MODEL_ID = "Tiiny/SmallThinker-4BA0.6B-Instruct"
MODEL = snapshot_download(MODEL_ID, local_dir="/kaggle/working/model")
print("model files:", MODEL)

# The MoE router normalises its sigmoid weights in place:
#     routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
# which autograd rejects ("modified by an inplace operation") because the
# division's backward needs the pre-division tensor.  Rewrite it out of place.
_mp = pathlib.Path(MODEL) / "modeling_smallthinker.py"
_src = _mp.read_text()
_bad = "routing_weights /= routing_weights.sum(dim=-1, keepdim=True)"
_good = "routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)"
if _bad in _src:
    _mp.write_text(_src.replace(_bad, _good))
    print("patched in-place router normalisation -> out of place")
else:
    print("router patch not needed (upstream changed)")
"""),
    code("""import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

print("loading base model ...", flush=True)
t0 = time.time()
base = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True, torch_dtype=torch.float16,
    low_cpu_mem_usage=True, attn_implementation="sdpa")
base.to("cuda:0")
base.config.use_cache = False
print(f"loaded in {time.time()-t0:.0f}s", flush=True)
n = sum(p.numel() for p in base.parameters())
print(f"params: {n/1e9:.3f} B")
print("after load, cuda mem: %.2f GiB" % (torch.cuda.memory_allocated(0)/1024**3))

cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                 bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg)
model.print_trainable_parameters()
"""),
    code("""import torch, torch.nn.functional as F, time

model.train()

# Manual activation checkpointing.  The model reports
# supports_gradient_checkpointing=False only because it never wires up
# self._gradient_checkpointing_func; wrapping each decoder layer is equivalent
# and cuts activation memory from O(n_layers x seq) to O(seq).
import torch.utils.checkpoint as _ck

def find_layers(m):
    node = m
    for _ in range(6):
        if hasattr(node, "layers"):
            return node.layers
        node = getattr(node, "model", None)
        if node is None:
            break
    raise RuntimeError("decoder layers not found")

def enable_layer_checkpointing(m):
    layers = find_layers(m)
    for layer in layers:
        if getattr(layer, "_ckpt_patched", False):
            continue
        orig = layer.forward
        def make(forward_fn):
            def checkpointed(*args, **kwargs):
                kwargs.pop("past_key_value", None)
                return _ck.checkpoint(forward_fn, *args, use_reentrant=False, **kwargs)
            return checkpointed
        layer.forward = make(orig)
        layer._ckpt_patched = True
    return len(layers)

print("checkpointed layers:", enable_layer_checkpointing(model))

def bench(seq_len, target_len, steps=6):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0)
    dev = "cuda:0"
    ids = torch.randint(1000, 140000, (1, seq_len), device=dev)
    labels = torch.full_like(ids, -100)
    labels[:, seq_len-target_len:] = ids[:, seq_len-target_len:]
    times = []
    for i in range(steps):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                        use_cache=False, return_dict=True, logits_to_keep=target_len+1)
            logits = out.logits[:, :-1, :].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]),
                                   labels[:, -target_len:].reshape(-1), ignore_index=-100)
        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); times.append(time.time()-t0)
    peak = torch.cuda.max_memory_allocated(0)/1024**3
    return {"seq": seq_len, "target": target_len, "peak_gib": round(peak,2),
            "step_s": round(sum(times[2:])/max(1,len(times[2:])),3), "loss": round(float(loss),3)}

results = []
for seq, tgt in [(2048, 512), (3072, 512), (3072, 1536), (3072, 2560), (4096, 1536)]:
    try:
        r = bench(seq, tgt)
        results.append(r); print(r, flush=True)
        notify(f"probe seq={seq} target={tgt}: peak {r['peak_gib']} GiB, {r['step_s']}s/step")
    except RuntimeError as e:
        print(f"seq={seq} target={tgt} FAILED: {e}", flush=True)
        results.append({"seq": seq, "target": tgt, "error": str(e)[:200]})
        torch.cuda.empty_cache(); model.zero_grad(set_to_none=True)
        break

print(json.dumps(results, indent=2))
with open("/kaggle/working/probe_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("PROBE COMPLETE")
notify("probe complete: " + ", ".join(f"{r.get('seq')}/{r.get('target')}={r.get('peak_gib','FAIL')}GiB" for r in results), prio="high")
"""),
    md("""Read `probe_results.json`. Pick the largest `seq` whose `peak_gib` stays
below ~13.5 GiB and use it as `SEQ_LEN` in the training notebook.  If every
length failed, fall back to 1024 and shrink `MAX_TARGET_TOKENS` in the
dataset builder.
"""),
])


# ==========================================================================
# 2. full training notebook
# ==========================================================================
train = notebook([
    md("""# SmallThinker-4BA0.6B-Instruct - coding-agent LoRA SFT

Trains a LoRA adapter on `Tiiny/SmallThinker-4BA0.6B-Instruct` so the model
learns three things it does not have out of the box:

1. **thinking** - inline `<think> ...</think>` reasoning traces,
2. **tool calling** - the model's native `<tool_call> {json} </tool_call>` protocol,
3. **coding** - reinforced DeepSeek V4 Pro style code answers.

The dataset is **not** streamed from Hugging Face.  It is cloned from
`https://github.com/Exo-del/smallthinker-coding-agent-data` - a prepared,
pre-filtered, byte-stable corpus with an explicit manifest.  The previous run
of this experiment burned ~2.8 h of GPU quota and died with zero logs because
it depended on live dataset streaming; shipping the data removes every moving
part except the base model download.

Pipeline: clone -> tokenise -> LoRA DDP over 2x T4 -> evaluate -> zip adapter.
"""),
    code("""import os, sys, time, json, subprocess, shutil, glob

WORK = "/kaggle/working"
os.chdir(WORK)
print("python:", sys.version.replace("\\n", " "))
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "gpus:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name} {p.total_memory/1024**3:.2f} GiB")
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
"""),
    code(NOTIFY_CELL),
    code("""# Tiiny/SmallThinker-4BA0.6B-Instruct is a transformers 4.x model: its remote
# code imports HybridCache, LossKwargs and ROPE_INIT_FUNCTIONS["default"], all of
# which transformers 5.x removed.  Pin the 4.x stack, and load the weights from a
# dot-free local directory because the repo id (4BA0.6B) breaks the dynamic
# module path in 4.x.
!pip install -q "transformers==4.53.2" "huggingface_hub==0.36.2" "tokenizers==0.21.4" "peft==0.17.1" 2>&1 | tail -3
import peft, transformers, huggingface_hub
print("peft:", peft.__version__, "| transformers:", transformers.__version__, "| hub:", huggingface_hub.__version__)

from huggingface_hub import snapshot_download
import pathlib
MODEL_ID = "Tiiny/SmallThinker-4BA0.6B-Instruct"
MODEL = snapshot_download(MODEL_ID, local_dir="/kaggle/working/model")
print("model files:", MODEL)

# The MoE router normalises its sigmoid weights in place:
#     routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
# which autograd rejects ("modified by an inplace operation") because the
# division's backward needs the pre-division tensor.  Rewrite it out of place.
_mp = pathlib.Path(MODEL) / "modeling_smallthinker.py"
_src = _mp.read_text()
_bad = "routing_weights /= routing_weights.sum(dim=-1, keepdim=True)"
_good = "routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)"
if _bad in _src:
    _mp.write_text(_src.replace(_bad, _good))
    print("patched in-place router normalisation -> out of place")
else:
    print("router patch not needed (upstream changed)")
"""),
    code("""REPO = "https://github.com/Exo-del/smallthinker-coding-agent-data.git"
DATA_DIR = "/kaggle/working/smallthinker-coding-agent-data"

if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR)
r = subprocess.run(["git", "clone", "--depth", "1", REPO, DATA_DIR],
                   capture_output=True, text=True)
print(r.stdout[-2000:]); print(r.stderr[-2000:])
assert r.returncode == 0, "git clone failed"
print(sorted(os.listdir(DATA_DIR)))
manifest = json.load(open(f"{DATA_DIR}/manifest.json"))
print(json.dumps(manifest["totals"], indent=2))
print(json.dumps(manifest["mixture_by_category"], indent=2))
notify("data cloned: %d convs / %.1fM target tokens" % (
    manifest["totals"]["conversations"], manifest["totals"]["target_tokens"]/1e6))
"""),
    code("""# configuration - SEQ_LEN must match what the probe showed fits in memory
SEQ_LEN = int(os.environ.get("SEQ_LEN", "3072"))
REPO_DIR = "/kaggle/working/smallthinker-coding-agent-data"
ARR_DIR = "/kaggle/working/arrays"

LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.05
TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj"
BATCH_PER_GPU = 1
GRAD_ACCUM = 8
LR = 1e-4
WEIGHT_DECAY = 0.01
WARMUP = 60
MAX_STEPS = 4000
SAVE_EVERY = 100
RUN_HOURS = 8.0
PAD_ID = 151643

print(f"SEQ_LEN={SEQ_LEN} LR={LR} steps={MAX_STEPS} accum={GRAD_ACCUM}")
"""),
    code("""# build the tokenised arrays from the cloned corpus
t0 = time.time()
r = subprocess.run(
    [sys.executable, f"{REPO_DIR}/scripts/build_examples.py",
     "--data-dir", REPO_DIR, "--out", ARR_DIR, "--seq-len", str(SEQ_LEN), "--model", MODEL],
    capture_output=True, text=True)
print(r.stdout[-6000:]); print(r.stderr[-4000:])
assert r.returncode == 0, "build_examples failed"
print("built in %.0fs" % (time.time()-t0))
notify("tokenised arrays built in %.0f min" % ((time.time()-t0)/60))
print(json.load(open(f"{ARR_DIR}/dataset_summary.json"))["totals"] if os.path.exists(f"{ARR_DIR}/dataset_summary.json") else "")
"""),
    code("""# smoke test: 12 optimizer steps on a single GPU, watching memory and loss
smoke = subprocess.run(
    [sys.executable, f"{REPO_DIR}/scripts/train_smallthinker.py",
     "--model", MODEL, "--data", ARR_DIR, "--out", "/kaggle/working/smoke_out",
     "--seq-len", str(SEQ_LEN), "--pad-id", str(PAD_ID), "--batch", "1",
     "--grad-accum", "4", "--max-steps", "3", "--save-every", "1000",
     "--run-hours", "0.4", "--attn", "sdpa", "--rank", str(LORA_R),
     "--alpha", str(LORA_ALPHA), "--dropout", str(LORA_DROPOUT), "--lr", str(LR)],
    capture_output=True, text=True)
print(smoke.stdout[-8000:]); print(smoke.stderr[-4000:])
assert smoke.returncode == 0, "smoke test failed"
print("SMOKE TEST OK")
notify("smoke test OK - starting full DDP run", prio="high")
"""),
    code("""# full DDP run over both T4s
os.makedirs("/kaggle/working/train_out", exist_ok=True)
cmd = [
    sys.executable, "-m", "torch.distributed.run",
    "--nproc_per_node", "2", "--master_port", "29517",
    f"{REPO_DIR}/scripts/train_smallthinker.py",
    "--model", MODEL, "--data", ARR_DIR, "--out", "/kaggle/working/train_out",
    "--seq-len", str(SEQ_LEN), "--pad-id", str(PAD_ID),
    "--batch", str(BATCH_PER_GPU), "--grad-accum", str(GRAD_ACCUM),
    "--max-steps", str(MAX_STEPS), "--save-every", str(SAVE_EVERY),
    "--run-hours", str(RUN_HOURS), "--attn", "sdpa",
    "--rank", str(LORA_R), "--alpha", str(LORA_ALPHA), "--dropout", str(LORA_DROPOUT),
    "--lr", str(LR), "--weight-decay", str(WEIGHT_DECAY), "--warmup", str(WARMUP),
]
print(" ".join(cmd))
t0 = time.time()
with open("/kaggle/working/train.log", "w") as log:
    proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
print("exit code:", proc.returncode, "elapsed: %.0f s" % (time.time()-t0))
notify("training finished (exit %d) after %.1f h" % (proc.returncode, (time.time()-t0)/3600),
       prio="high" if proc.returncode == 0 else "urgent")
print(open("/kaggle/working/train.log").read()[-8000:])
"""),
    code("""# package the adapter as the notebook output
import zipfile, shutil, os, json
adapter_dir = "/kaggle/working/train_out/adapter_final"
if not os.path.isdir(adapter_dir):
    cands = sorted(glob.glob("/kaggle/working/train_out/checkpoints/step_*"))
    adapter_dir = cands[-1] if cands else adapter_dir
print("packaging", adapter_dir)
out_zip = "/kaggle/working/smallthinker-coding-agent-lora.zip"
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(adapter_dir):
        for fn in files:
            if fn == "training_state.pt":
                continue
            p = os.path.join(root, fn)
            z.write(p, os.path.relpath(p, os.path.dirname(adapter_dir)))
print("zip size: %.1f MB" % (os.path.getsize(out_zip)/1e6))
notify("adapter zipped: %.1f MB" % (os.path.getsize(out_zip)/1e6))
print(open("/kaggle/working/train_out/run_summary.json").read() if os.path.exists("/kaggle/working/train_out/run_summary.json") else "no run summary")
"""),
    code("""# quick qualitative check: does the adapter now think and call tools?
import torch, json, os, glob
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

if not os.path.isdir(adapter_dir):
    cands = sorted(glob.glob("/kaggle/working/train_out/checkpoints/step_*"))
    adapter_dir = cands[-1] if cands else ""
assert os.path.isdir(adapter_dir), "no adapter found - did training finish?"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
                                             torch_dtype=torch.float16,
                                             attn_implementation="sdpa").to("cuda:0")
model = PeftModel.from_pretrained(model, adapter_dir)
print("adapter loaded from", adapter_dir)
model.eval()

TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file from the project.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
prompts = [
    "List the functions defined in app.py and explain what each one does.",
    "Read the file config.yaml and tell me the database host.",
]
for p in prompts:
    msgs = [{"role": "system", "content": "You are a precise software engineering assistant."},
            {"role": "user", "content": p}]
    text = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=220, do_sample=False)
    print("="*70); print("PROMPT:", p); print("-"*70)
    print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False))
"""),
])


for name, nb in [("SmallThinker_Memory_Probe.kaggle.ipynb", probe),
                 ("SmallThinker_Coding_Agent_LoRA.kaggle.ipynb", train)]:
    path = OUT / name
    path.write_text(json.dumps(nb, indent=1))
    print("wrote", path, path.stat().st_size, "bytes")
