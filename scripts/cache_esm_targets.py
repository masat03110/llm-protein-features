"""Compute ESMFold's LM features s_lm (L x 1024) for the training set = distillation targets.

    python scripts/cache_esm_targets.py --tsv data/trainset/pdb_2020-05-01_id30.tsv \
        --out data/esm_targets --limit 4000

Shards: <out>/shard_XXXX.pt = {"ids": [...], "seqs": [...], "feats": [fp16 (L,1024) tensors]}
Stats:  <out>/stats.pt = per-dimension mean / std (from the train split)
Resumable: existing shards are skipped.
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
import torch
from tqdm import tqdm

from qwenfold.esmfold_utils import lm_features, load_esmfold, sequences_to_batch
from qwenfold.profiling import default_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--out", default="data/esm_targets")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N train + matching valid")
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--token-budget", type=int, default=4096, help="residues per ESM batch")
    ap.add_argument("--device", default=default_device())
    args = ap.parse_args()

    df = pd.read_csv(args.tsv, sep="\t")
    df = df.sample(frac=1.0, random_state=0)  # deterministic shuffle
    if args.limit:
        n_valid = max(50, int(args.limit * (df["split"] == "valid").mean()))
        df = pd.concat([df[df["split"] == "train"].head(args.limit), df[df["split"] == "valid"].head(n_valid)])
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "index.tsv"), sep="\t", index=False)

    model = load_esmfold(args.device, load_lm=True)
    model.trunk = None  # only ESM2 + esm_s_mlp are needed
    if args.device == "mps":
        torch.mps.empty_cache()

    for split in ["train", "valid"]:
        sub = df[df["split"] == split]
        for si, start in enumerate(range(0, len(sub), args.shard_size)):
            path = os.path.join(args.out, f"{split}_{si:04d}.pt")
            if os.path.exists(path):
                continue
            chunk = sub.iloc[start:start + args.shard_size]
            order = chunk.sort_values("length")  # length-sorted batches = little padding
            ids, seqs, feats = [], [], []
            batch = []
            rows = list(order.itertuples())

            def flush(batch):
                bseqs = [r.sequence for r in batch]
                aa, mask, _ = sequences_to_batch(bseqs, args.device)
                f = lm_features(model, aa, mask).to(torch.float16).cpu()
                for r, x in zip(batch, f):
                    ids.append(r.entity_id)
                    seqs.append(r.sequence)
                    feats.append(x[: len(r.sequence)].clone())
                del aa, mask, f
                # Variable batch shapes fragment the MPS caching allocator; without this the
                # footprint grows until the machine swaps (observed: stall after ~12k proteins).
                if args.device == "mps":
                    torch.mps.empty_cache()

            for r in tqdm(rows, desc=f"{split} shard {si}"):
                if batch and (len(batch) + 1) * r.length > args.token_budget:
                    flush(batch)
                    batch = []
                batch.append(r)
            if batch:
                flush(batch)
            torch.save({"ids": ids, "seqs": seqs, "feats": feats}, path + ".tmp")
            os.replace(path + ".tmp", path)

    # per-dimension statistics over train residues (for a standardised loss)
    s, s2, n = 0, 0, 0
    for p in sorted(glob.glob(os.path.join(args.out, "train_*.pt"))):
        for x in torch.load(p)["feats"]:
            x = x.double()
            s, s2, n = s + x.sum(0), s2 + (x * x).sum(0), n + len(x)
    mean = s / n
    std = (s2 / n - mean ** 2).clamp(min=1e-8).sqrt()
    torch.save({"mean": mean.float(), "std": std.float(), "n_residues": n}, os.path.join(args.out, "stats.pt"))
    print(f"done; {n} train residues")


if __name__ == "__main__":
    main()
