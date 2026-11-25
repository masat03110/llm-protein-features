"""CASP-style, sequence-dependent structure-comparison metrics (pure NumPy).

Implemented metrics
-------------------
* TM-score  (Zhang & Skolnick 2004; normalised by the native/target length)
* GDT_TS / GDT_HA (Zemla 2003; max fraction of CA within cutoffs under any
  superposition, normalised by target length)
* RMSD over all matched CA after optimal superposition
* lDDT (Mariani et al. 2013): all-atom, superposition-free, inclusion radius
  15 A, thresholds 0.5/1/2/4 A, inter-residue pairs only; plus CA-only lDDT.

Notes for the paper
-------------------
The official CASP numbers come from LGA (GDT) and OpenStructure (lDDT).  The
superposition search here follows the TM-score program strategy (fragment
seeds of length L, L/2, L/4, ... followed by iterative re-selection), which is
how TMscore itself reports GDT.  Because *all* methods (AlphaFold2 CASP14
models, ESMFold, ours) are scored by this same code, comparisons are fair;
absolute values can differ from the CASP website by ~1 GDT point.
Use ``scripts/validate_metrics.py`` to compare against official CASP tables.
"""
from __future__ import annotations

import numpy as np

from .structure_io import Structure

GDT_TS_CUTOFFS = (1.0, 2.0, 4.0, 8.0)
GDT_HA_CUTOFFS = (0.5, 1.0, 2.0, 4.0)
LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)

# Atoms whose names are interchangeable by symmetry (handled in lDDT).
SYMMETRIC_ATOMS = {
    "ASP": [("OD1", "OD2")],
    "GLU": [("OE1", "OE2")],
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
    "ARG": [("NH1", "NH2")],
}


# ----------------------------------------------------------------------------
# Superposition
# ----------------------------------------------------------------------------
def kabsch(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, t) minimising ||p @ R.T + t - q||."""
    pc, qc = p.mean(0), q.mean(0)
    h = (p - pc).T @ (q - qc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, qc - pc @ r.T


def _tm_d0(l_target: int) -> float:
    if l_target <= 21:
        return 0.5
    return max(0.5, 1.24 * (l_target - 15) ** (1.0 / 3.0) - 1.8)


def superposition_scores(x: np.ndarray, y: np.ndarray, l_target: int,
                         max_seeds_per_len: int = 20, n_iter: int = 20) -> dict:
    """Maximise TM-score and GDT counts over superpositions.

    x: model CA (N,3), y: native CA (N,3), matched pairs only.
    l_target: number of residues in the native (normaliser, as in CASP).
    """
    n = len(x)
    if n < 3:
        return {"tm_score": 0.0, "gdt_ts": 0.0, "gdt_ha": 0.0}
    d0 = _tm_d0(l_target)
    d0_search = min(max(d0, 4.5), 8.0)
    cutoffs = sorted(set(GDT_TS_CUTOFFS) | set(GDT_HA_CUTOFFS))
    best_tm = 0.0
    best_cnt = {c: 0 for c in cutoffs}

    def evaluate(idx):
        nonlocal best_tm
        r, t = kabsch(x[idx], y[idx])
        d = np.linalg.norm(x @ r.T + t - y, axis=1)
        best_tm = max(best_tm, float(np.sum(1.0 / (1.0 + (d / d0) ** 2)) / l_target))
        for c in cutoffs:
            best_cnt[c] = max(best_cnt[c], int(np.sum(d <= c)))
        return d

    # Seed fragment lengths: L, L/2, L/4, ... (>= 4)
    lengths, l_init = [], n
    while l_init >= min(4, n):
        lengths.append(l_init)
        l_init //= 2
    selection_radii = [d0_search] + cutoffs
    seen = set()
    for L in lengths:
        n_start = n - L + 1
        step = max(1, n_start // max_seeds_per_len)
        for start in range(0, n_start, step):
            seed = np.arange(start, start + L)
            d_seed = evaluate(seed)
            for radius in selection_radii:
                d = d_seed
                idx = seed
                for _ in range(n_iter):
                    r_sel = radius
                    new_idx = np.where(d < r_sel)[0]
                    while len(new_idx) < 3 and r_sel < 50:
                        r_sel += 0.5
                        new_idx = np.where(d < r_sel)[0]
                    key = new_idx.tobytes()
                    if np.array_equal(new_idx, idx) or key in seen:
                        break
                    seen.add(key)
                    idx = new_idx
                    d = evaluate(idx)
    frac = {c: best_cnt[c] / l_target for c in cutoffs}
    return {
        "tm_score": best_tm,
        "gdt_ts": 100.0 * float(np.mean([frac[c] for c in GDT_TS_CUTOFFS])),
        "gdt_ha": 100.0 * float(np.mean([frac[c] for c in GDT_HA_CUTOFFS])),
    }


# ----------------------------------------------------------------------------
# lDDT
# ----------------------------------------------------------------------------
def _lddt_from_atoms(ref: np.ndarray, mod: np.ndarray, res_id: np.ndarray,
                     present: np.ndarray, radius: float = 15.0) -> tuple[float, np.ndarray]:
    """Global lDDT and per-residue lDDT.

    ref/mod: (A,3) atom coordinates (mod rows undefined where ~present)
    res_id:  (A,) residue index of each atom (pairs within a residue ignored)
    present: (A,) bool, atom exists in the model.
    """
    a = len(ref)
    n_res = int(res_id.max()) + 1
    preserved = np.zeros(n_res)
    total = np.zeros(n_res)
    chunk = 2048
    for s in range(0, a, chunk):
        e = min(a, s + chunk)
        dref = np.linalg.norm(ref[s:e, None] - ref[None], axis=-1)
        mask = (dref < radius) & (res_id[s:e, None] != res_id[None])
        dmod = np.linalg.norm(mod[s:e, None] - mod[None], axis=-1)
        both = present[s:e, None] & present[None]
        diff = np.abs(dmod - dref)
        score = np.zeros_like(dref)
        for th in LDDT_THRESHOLDS:
            score += (diff < th) & both
        score /= len(LDDT_THRESHOLDS)
        np.add.at(preserved, res_id[s:e], (score * mask).sum(1))
        np.add.at(total, res_id[s:e], mask.sum(1))
    glob = float(preserved.sum() / max(total.sum(), 1))
    per_res = np.divide(preserved, total, out=np.zeros_like(preserved), where=total > 0)
    return glob, per_res


def _resolve_symmetry(native: Structure, model: Structure, keys) -> dict:
    """Swap symmetric side-chain atom names in the model when that better
    matches the native's local distances (standard lDDT practice)."""
    mod_atoms = {k: dict(model.residues[k].atoms) for k in keys}
    nat_ca = native.ca_coords(keys)
    for i, k in enumerate(keys):
        resn = native.residues[k].resname
        if resn not in SYMMETRIC_ATOMS:
            continue
        nbr = [keys[j] for j in np.where(np.linalg.norm(nat_ca - nat_ca[i], axis=1) < 15.0)[0] if j != i]
        ref_env, mod_env = [], []
        for kk in nbr:
            for an, xyz in native.residues[kk].atoms.items():
                if an in mod_atoms[kk]:
                    ref_env.append(xyz)
                    mod_env.append(mod_atoms[kk][an])
        if not ref_env:
            continue
        ref_env, mod_env = np.array(ref_env), np.array(mod_env)
        atoms = mod_atoms[k]
        for a1, a2 in SYMMETRIC_ATOMS[resn]:
            if a1 not in atoms or a2 not in atoms or a1 not in native.residues[k].atoms:
                continue

            def cost(m1, m2):
                c = 0.0
                for an, mxyz in ((a1, m1), (a2, m2)):
                    if an not in native.residues[k].atoms:
                        continue
                    dr = np.linalg.norm(ref_env - native.residues[k].atoms[an], axis=1)
                    dm = np.linalg.norm(mod_env - mxyz, axis=1)
                    c += np.abs(dr - dm)[dr < 15.0].sum()
                return c

            if cost(atoms[a2], atoms[a1]) < cost(atoms[a1], atoms[a2]):
                atoms[a1], atoms[a2] = atoms[a2], atoms[a1]
    return mod_atoms


def lddt(native: Structure, model: Structure, keys, ca_only: bool = False) -> tuple[float, np.ndarray]:
    """lDDT over all native residues ``keys`` (missing model residues score 0)."""
    mod_atoms = ({k: model.residues[k].atoms for k in keys if k in model.residues}
                 if ca_only else _resolve_symmetry(native, model, [k for k in keys if k in model.residues]))
    ref, mod, rid, pres = [], [], [], []
    for i, k in enumerate(keys):
        atoms = {"CA": native.residues[k].atoms["CA"]} if ca_only else native.residues[k].atoms
        m = mod_atoms.get(k, {})
        for an, xyz in atoms.items():
            ref.append(xyz)
            rid.append(i)
            if an in m:
                mod.append(m[an])
                pres.append(True)
            else:
                mod.append(np.zeros(3))
                pres.append(False)
    return _lddt_from_atoms(np.array(ref), np.array(mod), np.array(rid), np.array(pres))


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def score_model(native: Structure, model: Structure) -> dict:
    """All metrics, evaluated on the residues present in the native structure."""
    nat_keys = native.keys
    matched = [k for k in nat_keys if k in model.residues]
    l_target = len(nat_keys)
    out = {"n_native": l_target, "n_matched": len(matched),
           "coverage": len(matched) / max(l_target, 1)}
    if len(matched) < 3:
        out.update(tm_score=0.0, gdt_ts=0.0, gdt_ha=0.0, rmsd_ca=float("nan"), lddt=0.0, lddt_ca=0.0)
        return out
    x = model.ca_coords(matched)
    y = native.ca_coords(matched)
    out.update(superposition_scores(x, y, l_target))
    r, t = kabsch(x, y)
    out["rmsd_ca"] = float(np.sqrt(np.mean(np.sum((x @ r.T + t - y) ** 2, axis=1))))
    out["lddt"], _ = lddt(native, model, nat_keys)
    out["lddt_ca"], _ = lddt(native, model, nat_keys, ca_only=True)
    return out
