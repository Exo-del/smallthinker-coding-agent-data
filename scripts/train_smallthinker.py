import os, json, time, math, argparse, random
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

# torch>=2.4 moved the autocast / GradScaler API; support both spellings.
if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
    def autocast(**kw):
        return torch.amp.autocast("cuda", **kw)

    def make_scaler(enabled=True):
        return torch.amp.GradScaler("cuda", enabled=enabled)
else:  # pragma: no cover - older torch
    def autocast(**kw):
        return torch.cuda.amp.autocast(**kw)

    def make_scaler(enabled=True):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", default="")
    p.add_argument("--run-hours", type=float, default=8.0)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=768)
    p.add_argument("--pad-id", type=int, required=True)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attn", default="sdpa")
    return p.parse_args()


class MemmapDataset(Dataset):
    def __init__(self, data_dir, prefix, pad_id):
        data_dir = Path(data_dir)
        self.x = np.load(data_dir / f"{prefix}_input_ids.npy", mmap_mode="r")
        self.y = np.load(data_dir / f"{prefix}_labels.npy", mmap_mode="r")
        lengths_path = data_dir / f"{prefix}_lengths.npy"
        if lengths_path.exists():
            self.lengths = np.load(lengths_path)
            # drop rows that were never filled
            self.lengths = self.lengths[self.lengths > 0]
        else:
            self.lengths = None
        self.pad_id = int(pad_id)
    def __len__(self):
        if self.lengths is not None:
            return len(self.lengths)
        return self.x.shape[0]
    def __getitem__(self, i):
        x = np.asarray(self.x[i], dtype=np.int64)
        y = np.asarray(self.y[i], dtype=np.int64)
        if self.lengths is not None:
            n = int(self.lengths[i])
        else:
            nz = np.flatnonzero(x != self.pad_id)
            n = int(nz[-1] + 1) if len(nz) else 1
        return torch.from_numpy(x[:n].copy()), torch.from_numpy(y[:n].copy())


def collate(batch, pad_id):
    max_len = max(x.shape[0] for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        pad = max_len - x.shape[0]
        if pad:
            x = F.pad(x, (0, pad), value=pad_id)
            y = F.pad(y, (0, pad), value=-100)
        xs.append(x); ys.append(y)
    input_ids = torch.stack(xs)
    labels = torch.stack(ys)
    attention_mask = (input_ids != pad_id).long()
    return input_ids, attention_mask, labels


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    torch.set_num_threads(1)
    random.seed(args.seed + local_rank); np.random.seed(args.seed + local_rank); torch.manual_seed(args.seed + local_rank); torch.cuda.manual_seed_all(args.seed + local_rank)
    rank = dist.get_rank() if distributed else 0
    is_main = rank == 0
    out_dir = Path(args.out)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"world_size={world_size}; local_rank={local_rank}; GPU={torch.cuda.get_device_name(local_rank)}", flush=True)
        print("Loading base model in FP16; gradient checkpointing intentionally disabled (unsupported by this model implementation).", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn,
    )
    base.to(device)
    base.config.use_cache = False
    if hasattr(base, "generation_config"):
        base.generation_config.use_cache = False

    if args.resume:
        if is_main: print("Loading trainable adapter:", args.resume, flush=True)
        model = PeftModel.from_pretrained(base, args.resume, is_trainable=True)
    else:
        cfg = LoraConfig(
            r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, cfg)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters found. Check PEFT target module names.")
    if is_main:
        n_train = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {n_train:,} / {n_total:,} ({100*n_train/n_total:.4f}%)", flush=True)
        print("Trainable module examples:", [n for n,p in model.named_parameters() if p.requires_grad][:12], flush=True)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False, broadcast_buffers=False)
    model.train()
    raw_model = model.module if distributed else model

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    def lr_lambda(step):
        if step < args.warmup:
            return max(1e-3, step / max(1, args.warmup))
        progress = min(1.0, (step - args.warmup) / max(1, args.max_steps - args.warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = make_scaler(enabled=True)

    global_step, start_epoch, start_batch = 0, 0, 0
    if args.resume:
        state_path = Path(args.resume) / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state.get("scaler", {}))
            global_step = int(state.get("global_step", 0))
            start_epoch = int(state.get("epoch", 0))
            start_batch = int(state.get("next_batch", 0))
            if is_main: print(f"Resumed optimizer at step={global_step}, epoch={start_epoch}, next_batch={start_batch}", flush=True)
        elif is_main:
            print("No training_state.pt found; adapter weights loaded but optimizer restarts.", flush=True)

    ds = MemmapDataset(args.data, "train", args.pad_id)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True,
                                 seed=args.seed, drop_last=True) if distributed else None
    loader = DataLoader(ds, batch_size=args.batch, sampler=sampler, shuffle=(sampler is None),
                        num_workers=0, pin_memory=True, drop_last=True,
                        collate_fn=lambda b: collate(b, args.pad_id))
    batches_per_epoch = (len(loader) // args.grad_accum) * args.grad_accum
    if batches_per_epoch < args.grad_accum:
        raise RuntimeError(f"Not enough batches ({len(loader)}) for grad_accum={args.grad_accum}.")

    optimizer.zero_grad(set_to_none=True)
    run_start = time.monotonic()
    deadline = run_start + max(0.05, args.run_hours) * 3600
    last_loss = 0.0
    stop = False

    def save_checkpoint(epoch, next_batch, label="step"):
        if distributed: dist.barrier()
        if is_main:
            name = f"step_{global_step:06d}"
            path = ckpt_dir / name
            path.mkdir(parents=True, exist_ok=True)
            raw_model.save_pretrained(path, safe_serialization=True)
            torch.save({
                "global_step": global_step, "epoch": epoch, "next_batch": next_batch,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            }, path / "training_state.pt")
            (out_dir / "latest_checkpoint.json").write_text(json.dumps({"relative_path": f"checkpoints/{name}", "global_step": global_step}, indent=2))
            print(f"Saved {label}: {path}", flush=True)
        if distributed: dist.barrier()

    epoch = start_epoch
    while global_step < args.max_steps and not stop:
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= batches_per_epoch:
                break
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            input_ids, attention_mask, labels = [t.to(device, non_blocking=True) for t in batch]
            do_sync = ((batch_idx + 1) % args.grad_accum == 0)
            sync_ctx = nullcontext() if (not distributed or do_sync) else model.no_sync()
            with sync_ctx:
                with autocast(dtype=torch.float16):
                    # The supervised span is always the tail of the sequence, so
                    # the LM head only needs to run over the target tokens.
                    # With a 151,936-entry vocabulary this is the difference
                    # between ~1.2 GiB and ~0.2 GiB of activation memory.
                    counts = (labels != -100).sum(dim=1)
                    n_target = int(counts.min().item())
                    active = labels != -100
                    tail_ok = False
                    if n_target > 0 and bool((counts == counts.max()).item()):
                        tail_ok = (bool(active[:, -n_target:].all().item())
                                   and not bool(active[:, :-n_target].any().item()))
                    if tail_ok:
                        keep = n_target + 1
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                        use_cache=False, return_dict=True,
                                        logits_to_keep=keep)
                        logits = outputs.logits[:, :-1, :].contiguous()
                        shift_labels = labels[:, -n_target:].contiguous()
                    else:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                        use_cache=False, return_dict=True)
                        logits = outputs.logits[:, :-1, :].contiguous()
                        shift_labels = labels[:, 1:].contiguous()
                    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]),
                                           shift_labels.view(-1), ignore_index=-100)
                    loss = loss / args.grad_accum
                scaler.scale(loss).backward()

            if not do_sync:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            last_loss = float(loss.detach().cpu()) * args.grad_accum
            next_batch = batch_idx + 1

            if is_main and global_step % 10 == 0:
                elapsed = (time.monotonic() - run_start) / 3600
                lr = scheduler.get_last_lr()[0]
                mem = torch.cuda.max_memory_allocated(local_rank) / 1024**3
                print(f"step={global_step} loss~={last_loss:.4f} lr={lr:.2e} elapsed_h={elapsed:.2f} peak_gib={mem:.2f}", flush=True)

            need_save = (global_step % args.save_every == 0) or (time.monotonic() >= deadline) or (global_step >= args.max_steps)
            if need_save:
                save_checkpoint(epoch, next_batch, label="checkpoint")
            stop_tensor = torch.zeros(1, device=device)
            if is_main and (time.monotonic() >= deadline or global_step >= args.max_steps):
                stop_tensor.fill_(1)
            if distributed: dist.broadcast(stop_tensor, src=0)
            elif is_main and (time.monotonic() >= deadline or global_step >= args.max_steps):
                stop_tensor.fill_(1)
            if int(stop_tensor.item()) == 1:
                stop = True
                break

        if not stop:
            epoch += 1
            start_epoch, start_batch = epoch, 0

    # Always save a final adapter snapshot and its optimizer state.
    if distributed: dist.barrier()
    if is_main:
        final_dir = out_dir / "adapter_final"
        final_dir.mkdir(parents=True, exist_ok=True)
        raw_model.save_pretrained(final_dir, safe_serialization=True)
        torch.save({
            "global_step": global_step, "epoch": epoch, "next_batch": 0,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }, final_dir / "training_state.pt")
        (out_dir / "run_summary.json").write_text(json.dumps({
            "global_step": global_step, "last_loss": last_loss,
            "elapsed_hours": (time.monotonic() - run_start) / 3600,
            "train_examples": len(ds), "world_size": world_size,
            "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "max_seq_len": args.seq_len,
        }, indent=2))
        print("Training segment finished. Adapter:", final_dir, flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
