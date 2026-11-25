"""Stage 2: distil ESM2 features into a Qwen encoder (LoRA + adapter).

    python scripts/train_distill.py --qwen models/Qwen3.5-0.8B-Base --targets data/esm_targets \
        --mode echo --run runs/qwen0.8b_echo --steps 3000

Optional --fold-eval N: at every evaluation, fold N validation proteins with
the frozen ESMFold trunk from (a) Qwen features and (b) teacher ESM2 features,
and report TM-score / lDDT between the two (agreement with the teacher).

Resuming (for batch systems with a wall-clock limit):
    --resume            continue from <run>/state.pt if it exists (otherwise start fresh), so the same
                        command can be submitted repeatedly, e.g. as a chain of `qsub -hold_jid` jobs.
    --ckpt-every-min M  save <run>/state.pt every M minutes of wall time and at every evaluation.
    --max-hours H       save and exit cleanly after H hours (keep it below the queue's 96 h limit).
    SIGTERM / SIGUSR1 / SIGUSR2 also trigger a save-and-exit (e.g. `qsub -l s_rt=...` or `-notify`).
state.pt holds the trainable weights, AdamW state, step, best val loss, RNG states and the exact
position in the data stream, so a resumed run continues with the same batches as an uninterrupted one.
When training finishes, <run>/DONE is written; a later --resume then exits immediately.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import queue
import random
import signal
import sys
import threading
import time

import torch

from qwenfold.qwen_encoder import QwenProteinEncoder, distill_loss
from qwenfold.profiling import default_device


class ShardStream:
    """Token-budget batches from shard files, shuffled at shard and item level.

    Iterating yields (batch, pos).  `pos` describes the stream position right after that batch;
    ShardStream(..., resume=pos) continues from there with exactly the same batches.
    """

    def __init__(self, files, token_budget, max_batch, seed=0, resume: dict | None = None):
        self.files, self.token_budget, self.max_batch = files, token_budget, max_batch
        self.seed, self.resume = seed, resume

    def _batches(self, f, rng):
        d = torch.load(f)
        items = list(zip(d["seqs"], d["feats"]))
        rng.shuffle(items)
        items.sort(key=lambda x: len(x[0]))  # bucket by length ...
        batches, cur = [], []
        for it in items:
            if cur and ((len(cur) + 1) * len(it[0]) > self.token_budget or len(cur) >= self.max_batch):
                batches.append(cur)
                cur = []
            cur.append(it)
        if cur:
            batches.append(cur)
        rng.shuffle(batches)  # ... but random batch order
        return batches

    def __iter__(self):
        rng = random.Random(self.seed)
        order, fi, skip = None, 0, 0
        if self.resume:  # rewind the RNG to the start of the shard we were in, then skip consumed batches
            order, fi, skip = list(self.resume["order"]), self.resume["fi"], self.resume["bi"] + 1
            rng.setstate(self.resume["rng"])
        while True:
            if order is None:
                order = self.files[:]
                rng.shuffle(order)
                fi = 0
            while fi < len(order):
                rng_at_shard = rng.getstate()
                for bi, b in enumerate(self._batches(order[fi], rng)):
                    if bi >= skip:
                        yield b, {"order": order, "fi": fi, "bi": bi, "rng": rng_at_shard}
                skip = 0
                fi += 1
            order = None


def collate(batch, device=None, pin=False):
    seqs = [s for s, _ in batch]
    L = max(len(s) for s in seqs)
    tgt = torch.zeros(len(batch), L, batch[0][1].shape[-1], pin_memory=pin)
    for i, (_, f) in enumerate(batch):
        tgt[i, : len(f)] = f.float()
    return seqs, (tgt if device is None else tgt.to(device, non_blocking=pin))


class Prefetcher:
    """Collate the next batches on a background thread (pinned memory) while the GPU trains."""

    def __init__(self, batches, device, depth=4):
        self.device = device
        self.pin = str(device).startswith("cuda")
        self.q = queue.Queue(maxsize=depth)
        threading.Thread(target=self._work, args=(iter(batches),), daemon=True).start()

    def _work(self, it):
        try:
            for b, pos in it:
                self.q.put((*collate(b, None, self.pin), pos))
        except BaseException as e:  # surface errors in the main thread
            self.q.put(e)

    def __iter__(self):
        return self

    def __next__(self):
        x = self.q.get()
        if isinstance(x, BaseException):
            raise x
        seqs, tgt, pos = x
        return seqs, tgt.to(self.device, non_blocking=self.pin), pos


@torch.no_grad()
def evaluate(enc, files, std, device, token_budget, max_items=400):
    enc.eval()
    tot, n, mse, cos = 0.0, 0, 0.0, 0.0
    seen = 0
    for f in files:
        d = torch.load(f)
        items = sorted(zip(d["seqs"], d["feats"]), key=lambda x: len(x[0]))[: max_items - seen]
        cur = []
        for it in items + [None]:
            if it is None or (cur and (len(cur) + 1) * len(it[0]) > token_budget):
                if cur:
                    seqs, tgt = collate(cur, device)
                    pred, mask = enc(seqs)
                    loss, parts = distill_loss(pred, tgt, mask, std)
                    k = len(cur)
                    tot, mse, cos, n = tot + loss.item() * k, mse + parts["mse"] * k, cos + parts["cos"] * k, n + k
                cur = []
            if it is not None:
                cur.append(it)
        seen += len(items)
        if seen >= max_items:
            break
    enc.train()
    return {"val_loss": tot / n, "val_mse": mse / n, "val_cos": cos / n, "val_n": n}


@torch.no_grad()
def fold_eval(enc, trunk_model, files, device, n_items):
    """Agreement between structures folded from Qwen features and from teacher features."""
    from qwenfold.esmfold_utils import fold_from_features, output_to_pdb_strings, sequences_to_batch
    from qwenfold.metrics import score_model
    from qwenfold.structure_io import read_pdb
    import tempfile

    enc.eval()
    d = torch.load(files[0])
    items = [x for x in zip(d["seqs"], d["feats"]) if len(x[0]) <= 300][:n_items]
    res = []
    for seq, feat in items:
        aa, mask, pos = sequences_to_batch([seq], device)
        pdbs = []
        for s_lm in (enc([seq])[0], feat.float().unsqueeze(0).to(device)):
            out = fold_from_features(trunk_model, s_lm, aa, mask, pos)
            pdbs.append(output_to_pdb_strings(out)[0])
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i, p in enumerate(pdbs):
                paths.append(os.path.join(td, f"{i}.pdb"))
                open(paths[-1], "w").write(p)
            s = score_model(read_pdb(paths[1]), read_pdb(paths[0]))
        res.append(s)
    enc.train()
    return {"fold_tm_vs_teacher": sum(r["tm_score"] for r in res) / len(res),
            "fold_lddt_vs_teacher": sum(r["lddt"] for r in res) / len(res)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", default="models/Qwen3.5-0.8B-Base")
    ap.add_argument("--encoder", choices=["qwen", "esm2"], default="qwen",
                    help="qwen: general-purpose LLM (--qwen, --mode); esm2: small protein LM (--esm), bidirectional")
    ap.add_argument("--esm", default="models/esm2_t33_650M_UR50D", help="ESM2 checkpoint for --encoder esm2")
    ap.add_argument("--esm-inject", default=None,
                    help="qwen encoder: add frozen ESM2 features (this checkpoint) to the residue token embeddings "
                         "through a trained zero-initialised projector (Calpha-style)")
    ap.add_argument("--qwen-random-init", type=int, default=None, metavar="SEED",
                    help="qwen encoder: randomly initialised Qwen weights (control for language pre-training)")
    ap.add_argument("--targets", default="data/esm_targets")
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", choices=["causal", "echo", "bidir"], default="echo")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--adapter-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--token-budget", type=int, default=2048)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--fold-eval", type=int, default=0, help="#valid proteins to fold at each eval (0=off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--lora-compute", choices=["fp32", "bf16", "autocast"], default=None,
                    help="default: autocast on CUDA (LoRA matmuls in bf16, fp32 master weights; ~1.4x faster, "
                         "same val loss as fp32 in a 1000-step check), fp32 elsewhere (PEFT default)")
    ap.add_argument("--grad-ckpt", choices=["auto", "on", "off"], default="auto",
                    help="gradient checkpointing: less memory, ~1.3x slower, numerically equivalent. "
                         "auto = off on CUDA (since 2026-11-18), on elsewhere (Mac MPS memory is tight)")
    ap.add_argument("--prefetch", action="store_true", help="collate batches on a background thread")
    ap.add_argument("--resume", action="store_true", help="continue from <run>/state.pt if present")
    ap.add_argument("--ckpt-every-min", type=float, default=30.0, help="save state.pt every N minutes (0=only at eval)")
    ap.add_argument("--max-hours", type=float, default=0.0, help="save and exit after N hours (0=no limit)")
    ap.add_argument("--stop-at-step", type=int, default=0, help=argparse.SUPPRESS)  # testing: simulate interruption
    args = ap.parse_args()
    if args.lora_compute is None:
        args.lora_compute = "autocast" if args.device.startswith("cuda") else "fp32"
    if args.grad_ckpt == "auto":
        args.grad_ckpt = "off" if args.device.startswith("cuda") else "on"

    os.makedirs(args.run, exist_ok=True)
    state_path = os.path.join(args.run, "state.pt")
    if args.resume and os.path.exists(os.path.join(args.run, "DONE")):
        print(f"{args.run}: already finished (DONE exists); nothing to do", flush=True)
        return
    state = torch.load(state_path, weights_only=False) if args.resume and os.path.exists(state_path) else None
    if state is not None:
        check_resume_args(state["args"], vars(args))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    json.dump(vars(args), open(os.path.join(args.run, "args.json"), "w"), indent=2)
    dev = args.device

    stats = torch.load(os.path.join(args.targets, "stats.pt"))
    std = stats["std"].to(dev)
    train_files = sorted(glob.glob(os.path.join(args.targets, "train_*.pt")))
    valid_files = sorted(glob.glob(os.path.join(args.targets, "valid_*.pt")))
    assert train_files and valid_files, "run scripts/cache_esm_targets.py first"

    if args.encoder == "esm2":
        from qwenfold.esm_encoder import ESMProteinEncoder
        enc = ESMProteinEncoder(args.esm, lora_r=args.lora_r, lora_compute=args.lora_compute,
                                gradient_checkpointing=(args.grad_ckpt == "on")).to(dev)
    else:
        enc = QwenProteinEncoder(args.qwen, mode=args.mode, lora_r=args.lora_r, lora_compute=args.lora_compute,
                                 gradient_checkpointing=(args.grad_ckpt == "on"), esm_inject=args.esm_inject,
                                 random_init_seed=args.qwen_random_init).to(dev)
    enc.train()
    lora_params = [p for n, p in enc.named_parameters() if p.requires_grad and "lora_" in n]
    head_params = [p for n, p in enc.named_parameters() if p.requires_grad and "lora_" not in n]
    print(f"trainable: LoRA {sum(p.numel() for p in lora_params)/1e6:.2f}M, "
          f"adapter {sum(p.numel() for p in head_params)/1e6:.2f}M")
    groups = [{"params": lora_params, "lr": args.lr}, {"params": head_params, "lr": args.adapter_lr}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.01)  # --lora-r 0: no LoRA group
    base_lrs = [g["lr"] for g in opt.param_groups]

    start_step, best, elapsed_before, data_pos = 1, math.inf, 0.0, None
    if state is not None:
        missing = enc.load_state_dict(state["model"], strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
        opt.load_state_dict(state["opt"])
        start_step, best, elapsed_before, data_pos = state["step"] + 1, state["best"], state["elapsed_s"], state["data_pos"]
        random.setstate(state["rng_python"])
        torch.set_rng_state(state["rng_torch"])
        if dev.startswith("cuda") and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(state["rng_cuda"])
        trim_log(os.path.join(args.run, "log.jsonl"), state["step"])
        print(f"resumed from {state_path} at step {state['step']} (best val_loss {best:.4f})", flush=True)

    trunk_model = None
    if args.fold_eval:
        from qwenfold.esmfold_utils import load_esmfold
        trunk_model = load_esmfold(dev, load_lm=False)

    def save_state(step):
        tmp = state_path + ".tmp"
        torch.save({"step": step, "model": enc.trainable_state_dict(), "opt": opt.state_dict(), "best": best,
                    "elapsed_s": elapsed_before + time.time() - t0, "data_pos": data_pos,
                    "rng_python": random.getstate(), "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state() if dev.startswith("cuda") else None,
                    "args": vars(args)}, tmp)
        os.replace(tmp, state_path)  # atomic: a crash mid-save never leaves a broken state.pt
        log.write(json.dumps({"checkpoint": step}) + "\n")
        log.flush()
        print(f"saved state at step {step}", flush=True)

    stop = {"signal": None}
    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2):
        signal.signal(sig, lambda signum, frame: stop.__setitem__("signal", signum))

    log = open(os.path.join(args.run, "log.jsonl"), "a")
    shard_stream = iter(ShardStream(train_files, args.token_budget, args.max_batch, args.seed, resume=data_pos))
    stream = Prefetcher(shard_stream, dev) if args.prefetch else ((*collate(b, dev), pos) for b, pos in shard_stream)
    t0 = time.time()
    last_save = time.time()
    for step in range(start_step, args.steps + 1):
        scale = min(1.0, step / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * scale
        for _ in range(args.grad_accum):
            seqs, tgt, data_pos = next(stream)
            pred, mask = enc(seqs)
            loss, parts = distill_loss(pred, tgt, mask, std)
            (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_([p for p in enc.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            rec = {"step": step, "loss": loss.item(), **parts, "lr_scale": scale,
                   "elapsed_min": (elapsed_before + time.time() - t0) / 60}
            if dev == "mps":
                rec["gpu_gb"] = torch.mps.driver_allocated_memory() / 1024**3
            elif dev.startswith("cuda"):
                rec["gpu_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        evaluated = False
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(enc, valid_files, std, dev, args.token_budget)
            if trunk_model is not None:
                ev |= fold_eval(enc, trunk_model, valid_files, dev, args.fold_eval)
            ev["step"] = step
            print("EVAL", json.dumps(ev), flush=True)
            log.write(json.dumps({"eval": ev}) + "\n")
            log.flush()
            enc.save(os.path.join(args.run, "last.pt"), extra=ev)
            if ev["val_loss"] < best:
                best = ev["val_loss"]
                enc.save(os.path.join(args.run, "best.pt"), extra=ev)
            evaluated = True
        if dev == "mps" and step % 10 == 0:
            torch.mps.empty_cache()

        out_of_time = args.max_hours and (time.time() - t0) / 3600 >= args.max_hours
        interrupted = stop["signal"] is not None or out_of_time or step == args.stop_at_step
        if step < args.steps and (evaluated or interrupted or
                                  (args.ckpt_every_min and time.time() - last_save >= 60 * args.ckpt_every_min)):
            save_state(step)
            last_save = time.time()
        if interrupted and step < args.steps:
            why = f"signal {stop['signal']}" if stop["signal"] is not None else (
                f"--max-hours {args.max_hours}" if out_of_time else f"--stop-at-step {args.stop_at_step}")
            print(f"stopping at step {step} ({why}); resume with --resume", flush=True)
            sys.exit(0)
    open(os.path.join(args.run, "DONE"), "w").write(f"finished step {args.steps}\n")
    print("training finished", flush=True)


RESUME_MUST_MATCH = ["qwen", "targets", "mode", "lora_r", "steps", "lr", "adapter_lr", "warmup", "token_budget",
                     "max_batch", "grad_accum", "seed", "lora_compute"]
RESUME_MUST_MATCH_IF_SAVED = ["encoder", "esm", "esm_inject", "qwen_random_init"]  # added 2026-11-19; older runs do not record them


def check_resume_args(saved: dict, now: dict):
    """Refuse to resume with settings that would change the training trajectory."""
    diff = {k: (saved.get(k), now.get(k)) for k in RESUME_MUST_MATCH if saved.get(k) != now.get(k)}
    diff |= {k: (saved[k], now.get(k)) for k in RESUME_MUST_MATCH_IF_SAVED if k in saved and saved[k] != now.get(k)}
    if diff:
        raise SystemExit(f"--resume: settings differ from the saved run (saved, now): {diff}")


def trim_log(path: str, step: int):
    """Drop log lines written after the checkpoint we resume from (they will be recomputed)."""
    if not os.path.exists(path):
        return
    keep = []
    for line in open(path):
        rec = json.loads(line)
        s = rec.get("step", rec.get("eval", {}).get("step", rec.get("checkpoint", 0)))
        if s <= step:
            keep.append(line)
    keep.append(json.dumps({"resume_from": step}) + "\n")
    open(path, "w").writelines(keep)

if __name__ == "__main__":
    main()
