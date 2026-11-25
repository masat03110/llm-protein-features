"""Statistics for the poster: per-method mean TM-score with a 95% bootstrap CI, and the paired difference
from the reference (ESMFold) with a 95% bootstrap CI and a Wilcoxon signed-rank test.

    python analysis/poster_stats.py --out outputs/stats.csv \
        --csv 16=outputs/casp16/scores.csv [--csv 16=another.csv ...]

Input: CSV files written by scripts/score.py (columns: method, unit, tm_score, gdt_ts, lddt, peak_gpu_alloc_gb,
seconds).  Several CSVs of the same CASP round are merged by method name (the first one wins on duplicates).
Differences use only the units that both the method and the reference (--ref, default ESMFold) have.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def boot_ci(x: np.ndarray, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, help="ROUND=path")
    ap.add_argument("--ref", default="ESMFold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--external", default="AlphaFold2,CASP_,AF3,ColabFold",
                    help="prefixes of CASP server predictions (reference only); excluded when choosing the common units")
    args = ap.parse_args()
    frames = {}
    for spec in args.csv:
        rnd, path = spec.split("=", 1)
        d = pd.read_csv(path)
        old = frames.get(rnd)
        if old is not None:
            d = d[~d.method.isin(old.method.unique())]
            d = pd.concat([old, d])
        frames[rnd] = d
    rows = []
    ext = tuple(args.external.split(","))
    for rnd, d in frames.items():
        # use only units scored for all of our methods (same units within a round)
        ours = [m for m in d.method.unique() if not m.startswith(ext)]
        common = set.intersection(*(set(d[(d.method == m) & d.tm_score.notna()].unit) for m in ours))
        d = d[d.unit.isin(common)]
        ref = d[d.method == args.ref].set_index("unit")
        for m, g in d.groupby("method"):
            g = g.set_index("unit")
            tm = g.tm_score.dropna().values
            lo, hi = boot_ci(tm)
            r = {"round": f"CASP{rnd}", "method": m, "n_units": len(tm), "tm": tm.mean(), "tm_lo": lo, "tm_hi": hi,
                 "gdt_ts": g.gdt_ts.mean(), "lddt": g.lddt.mean(),
                 "gpu_mean_gb": g.peak_gpu_alloc_gb.groupby(g.target).first().mean() if "peak_gpu_alloc_gb" in g else np.nan,
                 "gpu_max_gb": g.peak_gpu_alloc_gb.max() if "peak_gpu_alloc_gb" in g else np.nan,
                 "sec_mean": g.seconds.groupby(g.target).first().mean() if "seconds" in g else np.nan}
            if len(ref) and m != args.ref:
                u = g.index.intersection(ref.index)
                diff = (g.loc[u, "tm_score"] - ref.loc[u, "tm_score"]).dropna().values
                if len(diff) >= 5:
                    dlo, dhi = boot_ci(diff)
                    p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                    r |= {"n_paired": len(diff), "diff_vs_ref": diff.mean(), "diff_lo": dlo, "diff_hi": dhi,
                          "ratio_vs_ref": g.loc[u, "tm_score"].mean() / ref.loc[u, "tm_score"].mean(),
                          "wilcoxon_p": p}
            rows.append(r)
    out = pd.DataFrame(rows).sort_values(["round", "tm"], ascending=[True, False])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
