"""Predict structures for CASP targets, recording time and peak memory per target.

    python scripts/predict.py --method esmfold --fasta data/casp14/sequences.fasta --outdir outputs/casp14/esmfold
    python scripts/predict.py --method qwenfold --ckpt runs/xxx/best.pt --fasta ... --outdir outputs/casp14/qwenfold

Writes <outdir>/<target>.pdb (residues numbered 1..L as in CASP) and
<outdir>/timings.jsonl.  Already-predicted targets are skipped (resumable).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import time

import torch

from qwenfold.predictors import ESMFoldPredictor, QwenFoldPredictor, TrunkOnlyPredictor, count_parameters
from qwenfold.profiling import default_device, measure


def read_fasta(path: str) -> dict[str, str]:
    seqs, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = ""
        elif name:
            seqs[name] += line.replace(" ", "")
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["esmfold", "qwenfold", "trunkonly"], required=True)
    ap.add_argument("--ckpt", help="QwenFold encoder checkpoint")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--target-regex", default=r"^T\d{4}$", help="default: CASP monomer targets only")
    ap.add_argument("--max-len", type=int, default=1000, help="skip longer targets (memory)")
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--num-recycles", type=int, default=None, help="default: model's max (4)")
    ap.add_argument("--trunk-dtype", choices=["fp32", "bf16"], default="fp32", help="ESMFold trunk precision")
    ap.add_argument("--quant-lm", choices=["none", "int8"], default="none", help="weight-only int8 for the language model")
    ap.add_argument("--quant-trunk", choices=["none", "int8"], default="none", help="weight-only int8 for the folding trunk")
    ap.add_argument("--device", default=default_device())
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    seqs = read_fasta(args.fasta)
    names = [n for n in seqs if re.match(args.target_regex, n)]
    if args.targets:
        names = [n for n in names if n in set(args.targets)]

    with measure(args.device) as load_m:
        if args.method == "esmfold":
            pred = ESMFoldPredictor(args.device, args.chunk_size, args.num_recycles, args.trunk_dtype,
                                    args.quant_lm, args.quant_trunk)
        elif args.method == "trunkonly":
            pred = TrunkOnlyPredictor(args.device, args.chunk_size, args.num_recycles, args.trunk_dtype,
                                      args.quant_lm, args.quant_trunk)
        else:
            pred = QwenFoldPredictor(args.ckpt, args.device, args.chunk_size, args.num_recycles, args.trunk_dtype,
                                     args.quant_lm, args.quant_trunk)
    run_info = {"method": args.method, "ckpt": args.ckpt, "device": args.device,
                "machine": platform.platform(), "torch": torch.__version__,
                "load_seconds": load_m.seconds, "load_peak_rss_gb": load_m.peak_rss_gb,
                "load_peak_gpu_gb": load_m.peak_gpu_gb, "load_peak_gpu_alloc_gb": load_m.peak_gpu_alloc_gb} | count_parameters(pred) | vars(args)
    json.dump(run_info, open(os.path.join(args.outdir, "run_info.json"), "w"), indent=2)
    print(json.dumps(run_info, indent=2))

    timing_path = os.path.join(args.outdir, "timings.jsonl")
    for name in names:
        seq = seqs[name]
        pdb_path = os.path.join(args.outdir, f"{name}.pdb")
        if os.path.exists(pdb_path):
            continue
        rec = {"target": name, "length": len(seq)}
        if len(seq) > args.max_len:
            rec["status"] = f"skipped (> {args.max_len})"
        else:
            try:
                with measure(args.device) as m:
                    pdb, info = pred.predict(seq)
                open(pdb_path, "w").write(pdb)
                rec |= {"status": "ok", "seconds": m.seconds, "peak_rss_gb": m.peak_rss_gb,
                        "peak_gpu_gb": m.peak_gpu_gb, "peak_gpu_alloc_gb": m.peak_gpu_alloc_gb,
                        "peak_gpu_reserved_gb": m.peak_gpu_reserved_gb} | info
            except RuntimeError as e:  # typically out-of-memory on long targets
                rec["status"] = f"error: {str(e)[:200]}"
            if args.device == "mps":
                torch.mps.empty_cache()
        rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        print(json.dumps(rec), flush=True)
        with open(timing_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
