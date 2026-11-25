"""Score one or more prediction sets against CASP natives (same code for all).

    python scripts/score.py --natives data/casp14/natives \
        --pred AlphaFold2=data/casp14/predictions/group427 \
        --pred ESMFold=outputs/casp14/esmfold \
        --pred QwenFold-0.8B=outputs/casp14/qwenfold_0.8b \
        --out outputs/casp14/scores.csv

Each prediction dir holds <target>.pdb (whole-target model, CASP numbering) and
optionally timings.jsonl written by scripts/predict.py.  Natives named
T1024-D1.pdb are evaluation domains (CASP's official unit for GDT_TS); the
model of T1024 is scored on each of its domains.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import pandas as pd
from tqdm import tqdm

from qwenfold.metrics import score_model
from qwenfold.structure_io import read_pdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--natives", required=True)
    ap.add_argument("--pred", action="append", required=True, help="NAME=DIR")
    ap.add_argument("--level", choices=["casp", "domain", "target"], default="casp",
                    help="casp: single evaluation domains (T1024-D1, -D2 ...) for multi-domain targets, "
                         "whole target otherwise; domain: only -D files; target: whole targets")
    ap.add_argument("--targets", nargs="*", help="restrict to these target ids (e.g. T1024)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.natives, "*.pdb")))
    base = lambda f: os.path.basename(f)
    # target ids: T1024, and since CASP15/16 also subunits (T1106s1) and conformational variants (T1214v1)
    TID = r"T\d{4}(?:[sv]\d+)?"
    tid = lambda f: re.match(TID, base(f)).group(0)
    whole = [f for f in files if re.match(rf"^{TID}\.pdb$", base(f))]
    single_dom = [f for f in files if re.match(rf"^{TID}-D\d\.pdb$", base(f))]  # D1, D2 (not D12 / D23)
    if args.level == "target":
        natives = whole
    elif args.level == "domain":
        natives = single_dom
    else:
        split = {tid(f) for f in single_dom}
        natives = sorted(single_dom + [f for f in whole if tid(f) not in split])
    if args.targets:
        natives = [f for f in natives if tid(f) in set(args.targets)]
    preds = dict(p.split("=", 1) for p in args.pred)

    rows = []
    for nat_path in tqdm(natives, desc="natives"):
        unit = os.path.basename(nat_path)[:-4]
        target = unit.split("-")[0]
        native = read_pdb(nat_path)
        if len(native.keys) < 3:
            continue
        for name, d in preds.items():
            mp = os.path.join(d, f"{target}.pdb")
            if not os.path.exists(mp):  # variant natives (T1214v1) share the sequence of T1214
                mp = os.path.join(d, re.sub(r"v\d+$", "", target) + ".pdb")
            if not os.path.exists(mp):
                continue
            s = score_model(native, read_pdb(mp))
            rows.append({"method": name, "unit": unit, "target": os.path.basename(mp)[:-4]} | s)
    df = pd.DataFrame(rows)

    # attach per-target cost (time / memory) when available
    cost = []
    for name, d in preds.items():
        tp = os.path.join(d, "timings.jsonl")
        if os.path.exists(tp):
            for line in open(tp):
                r = json.loads(line)
                cost.append({"method": name, "target": r["target"], "seconds": r.get("seconds"),
                             "peak_mem_gb": r.get("peak_rss_gb", 0) + r.get("peak_gpu_gb", 0),
                             "peak_gpu_alloc_gb": r.get("peak_gpu_alloc_gb")})
    if cost and not df.empty:
        df = df.merge(pd.DataFrame(cost), on=["method", "target"], how="left")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)

    if df.empty:
        print("no scored pairs")
        return
    # Fair summary: only units predicted by every method
    common = set.intersection(*(set(g["unit"]) for _, g in df.groupby("method")))
    sub = df[df["unit"].isin(common)]
    cols = [c for c in ["gdt_ts", "gdt_ha", "tm_score", "lddt", "lddt_ca", "rmsd_ca", "seconds", "peak_mem_gb", "peak_gpu_alloc_gb"]
            if c in sub]
    agg = sub.groupby("method")[cols].mean().round(3)
    agg.insert(0, "n_units", sub.groupby("method").size())
    print(f"\nUnits common to all methods: {len(common)} ({args.level} level)")
    print(agg.to_string())
    agg.to_csv(args.out.replace(".csv", "_summary.csv"))


if __name__ == "__main__":
    main()
