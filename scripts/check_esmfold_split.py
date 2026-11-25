"""Sanity checks for the pipeline (run once after downloading ESMFold).

1. Our split forward (lm_features -> fold_from_features) == HF EsmForProteinFolding.infer
2. ESMFold on small proteins in data/test_pdb scores high against the crystal structure,
   with time / memory measured.
"""
from __future__ import annotations

import glob
import os
import tempfile

import torch

from qwenfold.esmfold_utils import fold_from_features, lm_features, sequences_to_batch
from qwenfold.metrics import score_model
from qwenfold.predictors import ESMFoldPredictor
from qwenfold.profiling import default_device, measure
from qwenfold.structure_io import read_pdb


def main():
    dev = default_device()
    with measure(dev) as m:
        pred = ESMFoldPredictor(dev)
    print(f"load: {m.seconds:.1f}s, RSS {m.peak_rss_gb:.1f} GB, GPU {m.peak_gpu_gb:.1f} GB")
    model = pred.model

    seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    with torch.no_grad():
        ref = model.infer(seq)
        aa, mask, pos = sequences_to_batch([seq], dev)
        ours = fold_from_features(model, lm_features(model, aa, mask), aa, mask, pos)
    diff = (ref["positions"][-1] - ours["positions"][-1]).abs().max().item()
    print(f"[1] split vs HF forward: max |dx| = {diff:.2e} A  -> {'OK' if diff < 1e-2 else 'MISMATCH'}")

    for path in sorted(glob.glob("data/test_pdb/*.pdb")):
        native = read_pdb(path)
        seq = native.sequence()
        nums = [k[0] for k in native.keys]
        if nums != list(range(nums[0], nums[0] + len(nums))):
            print(f"  skip {path}: residue numbering has gaps")
            continue
        with measure(dev) as m:
            pdb, info = pred.predict(seq, residue_numbers=nums)
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as fh:
            fh.write(pdb)
        s = score_model(native, read_pdb(fh.name))
        os.unlink(fh.name)
        print(f"[2] {os.path.basename(path)} L={len(seq)}: TM={s['tm_score']:.3f} GDT_TS={s['gdt_ts']:.1f} "
              f"lDDT={s['lddt']:.3f} pLDDT={info['plddt']:.2f} | {m.seconds:.1f}s, "
              f"RSS {m.peak_rss_gb:.1f} GB, GPU {m.peak_gpu_gb:.1f} GB")


if __name__ == "__main__":
    main()
