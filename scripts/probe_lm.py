"""Does a language model "know" proteins?  Linear probes + sequence likelihood (E008).

    python scripts/probe_lm.py --out outputs/probe --models onehot qwen0.8b qwen0.8b_rand esm35m
    python scripts/probe_lm.py --out outputs/ppl --task ppl --models qwen0.8b esm35m

Data: native structures of CASP14/15/16 (data/casp1*/natives), residues mapped to the FASTA sequence by
residue number (all native files of a target are merged; targets whose numbering does not match the FASTA
are skipped).  Split by CASP round: --test-round (default 14) = test, the other rounds = probe training (15 % of
targets held out for layer / epoch selection).

--task probe   frozen features (no LoRA, no training of the LM) from a few layers of each model ->
               (a) 3-state secondary structure (simplified DSSP from N/CA/C/O: H = 4-turn helix, E = bridge),
                   linear softmax probe, metric Q3;
               (b) contacts (CB, CA for Gly, < 8 A), symmetric low-rank bilinear probe, metric precision of the
                   top L/5 long-range (|i-j| >= 24) predictions.
               Qwen: same input as QwenFold (prefix + one token per residue), "bidir" = forward pass on the
               sequence and on the reversed sequence, states concatenated.  ESM2: residue tokens of <cls> seq <eos>.
--task ppl     likelihood of the real sequence vs a shuffled one (same composition).  Qwen: causal NLL in bits per
               residue, (i) residue tokens " A" after the prefix, renormalised over the 20 amino acids, and
               (ii) the raw string ("MKV...") after the prefix, BPE tokens, total bits / residues.  ESM2: masked
               pseudo-likelihood (each position masked once).  Reference: unigram entropy of the composition.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qwenfold.structure_io import read_pdb

AA20 = "ACDEFGHIKLMNPQRSTVWY"
MODELS = {
    "onehot": None,
    "qwen0.8b": ("qwen", "models/Qwen3.5-0.8B-Base", None),
    "qwen0.8b_rand": ("qwen", "models/Qwen3.5-0.8B-Base", 0),
    "qwen2b": ("qwen", "models/Qwen3.5-2B-Base", None),
    "qwen9b": ("qwen", "models/Qwen3.5-9B-Base", None),
    "esm35m": ("esm", "models/esm2_t12_35M_UR50D", None),
    "esm150m": ("esm", "models/esm2_t30_150M_UR50D", None),
    "esm650m": ("esm", "models/esm2_t33_650M_UR50D", None),
    "esm3b": ("esm", "models/esm2_t36_3B_UR50D", None),
}


# ---------------------------------------------------------------- data
def read_fasta(path):
    seqs, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = ""
        elif name:
            seqs[name] += line.replace(" ", "")
    return seqs


def dssp3(N, CA, C, O, ok, prev_ok):
    """Simplified DSSP (Kabsch & Sander 1983): H-bond energy < -0.5 kcal/mol; H = two consecutive 4-turns,
    E = residue in a parallel / antiparallel bridge.  ok: residue has all backbone atoms."""
    L = len(CA)
    H = N.copy()  # amide H: N + unit(C_prev - O_prev)
    d = C[:-1] - O[:-1]
    H[1:] = N[1:] + d / np.linalg.norm(d, axis=1, keepdims=True)
    rON = np.linalg.norm(O[:, None] - N[None], axis=-1)
    rCH = np.linalg.norm(C[:, None] - H[None], axis=-1)
    rOH = np.linalg.norm(O[:, None] - H[None], axis=-1)
    rCN = np.linalg.norm(C[:, None] - N[None], axis=-1)
    with np.errstate(divide="ignore"):
        E = 0.084 * 332 * (1 / rON + 1 / rCH - 1 / rOH - 1 / rCN)
    hb = (E < -0.5) & ok[:, None] & ok[None]  # hb[i, j]: CO(i) -> NH(j)
    hb[:, ~prev_ok] = False  # no amide H without the preceding residue
    idx = np.arange(L)
    hb[np.abs(idx[:, None] - idx[None]) < 2] = False
    ss = np.zeros(L, dtype=np.int64)  # 0 = C, 1 = H, 2 = E
    turn4 = np.zeros(L, bool)
    turn4[: L - 4] = hb[idx[: L - 4], idx[: L - 4] + 4]
    for i in range(1, L - 4):
        if turn4[i - 1] and turn4[i]:
            ss[i: i + 4] = 1

    def h(a, b):
        return (a >= 0) & (a < L) & (b >= 0) & (b < L) & hb[np.clip(a, 0, L - 1), np.clip(b, 0, L - 1)]

    I, J = np.meshgrid(idx, idx, indexing="ij")
    far = np.abs(I - J) > 2
    par = (h(I - 1, J) & h(J, I + 1)) | (h(J - 1, I) & h(I, J + 1))
    anti = (h(I, J) & h(J, I)) | (h(I - 1, J + 1) & h(J - 1, I + 1))
    bridge = ((par | anti) & far).any(1)
    ss[bridge & (ss != 1)] = 2
    return ss


def load_round(casp: int, max_len: int):
    D = f"data/casp{casp}"
    fasta = read_fasta(f"{D}/sequences.fasta")
    files = {}
    for f in sorted(glob.glob(f"{D}/natives/*.pdb")):
        t = os.path.basename(f)[:-4].split("-D")[0]
        if t not in fasta:
            t = re.sub(r"v\d+$", "", t)
        if t in fasta:
            files.setdefault(t, []).append(f)
    out = []
    for t, fs in files.items():
        seq = fasta[t]
        if len(seq) > max_len:
            continue
        res = {}
        for f in fs:
            for k, r in read_pdb(f).residues.items():
                if k[1] == "":
                    res.setdefault(k[0], r)
        nums = sorted(n for n in res if 1 <= n <= len(seq))
        if len(nums) < 30:
            continue
        match = np.mean([seq[n - 1] == {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
                                        "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
                                        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
                                        "TYR": "Y", "VAL": "V"}.get(res[n].resname, "X") for n in nums])
        if match < 0.95:
            print(f"skip {t}: numbering match {match:.2f}")
            continue
        get = lambda n, a: res[n].atoms.get(a, np.full(3, np.nan))
        CA = np.stack([get(n, "CA") for n in nums])
        CB = np.stack([res[n].atoms.get("CB", res[n].atoms["CA"]) for n in nums])
        N, C, O = (np.stack([get(n, a) for n in nums]) for a in ("N", "C", "O"))
        ok = ~(np.isnan(N).any(1) | np.isnan(C).any(1) | np.isnan(O).any(1))
        N, C, O = (np.nan_to_num(x, nan=1e4) for x in (N, C, O))
        # chain breaks: no H-bond partner logic across gaps in numbering (H position needs residue n-1)
        prev_ok = np.r_[False, np.diff(nums) == 1]
        ss = dssp3(N, CA, C, O, ok, prev_ok)
        ss_ok = ok & prev_ok
        ss[~ok] = -1
        dist = np.linalg.norm(CB[:, None] - CB[None], axis=-1)
        num = np.array(nums)
        out.append({"target": f"{casp}:{t}", "seq": seq, "pos": num - 1, "ss": ss, "ss_ok": ss_ok,
                    "contact": dist < 8.0, "sep": np.abs(num[:, None] - num[None])})
    return out


# ---------------------------------------------------------------- features
def layer_pick(n_states, k=7):
    return sorted(set(int(round(x)) for x in np.linspace(0, n_states - 1, k)))


class Featurizer:
    def __init__(self, name, device):
        self.name, self.device = name, device
        spec = MODELS[name]
        if spec is None:
            self.kind = "onehot"
            self.layers = [0]
            return
        self.kind, path, rnd = spec
        if self.kind == "qwen":
            from qwenfold.qwen_encoder import load_text_backbone, PREFIX
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(path)
            self.model = load_text_backbone(path, torch.bfloat16, rnd).to(device).eval()
            self.prefix = self.tok.encode(PREFIX, add_special_tokens=False)
            self.res_ids = {a: self.tok.encode(" " + a, add_special_tokens=False)[0] for a in AA20 + "X"}
            n = self.model.config.num_hidden_layers + 1
        else:
            from transformers import AutoTokenizer, EsmModel
            self.tok = AutoTokenizer.from_pretrained(path)
            self.model = EsmModel.from_pretrained(path, dtype=torch.bfloat16, add_pooling_layer=False).to(device).eval()
            n = self.model.config.num_hidden_layers + 1
        self.layers = layer_pick(n)

    @torch.no_grad()
    def __call__(self, seq):
        """-> list over self.layers of (L, d) float16 CPU tensors."""
        L = len(seq)
        if self.kind == "onehot":
            x = torch.zeros(L, 21)
            for i, c in enumerate(seq):
                x[i, AA20.index(c) if c in AA20 else 20] = 1
            return [x.half()]
        if self.kind == "esm":
            ids = [self.tok.cls_token_id] + [self.tok.convert_tokens_to_ids(c) for c in seq] + [self.tok.eos_token_id]
            ids = torch.tensor([[self.tok.unk_token_id if i is None else i for i in ids]], device=self.device)
            hs = self.model(input_ids=ids, output_hidden_states=True).hidden_states
            return [hs[l][0, 1: L + 1].half().cpu() for l in self.layers]
        outs = []
        for s in (seq, seq[::-1]):
            ids = torch.tensor([self.prefix + [self.res_ids.get(c, self.res_ids["X"]) for c in s]], device=self.device)
            hs = self.model(input_ids=ids, output_hidden_states=True).hidden_states
            outs.append([hs[l][0, -L:].float() for l in self.layers])
        return [torch.cat([f, r.flip(0)], -1).half().cpu() for f, r in zip(*outs)]


# ---------------------------------------------------------------- probes
def standardize(train_x):
    x = torch.cat(train_x).float()
    return x.mean(0), x.std(0) + 1e-4


def ss_probe(Xtr, ytr, Xva, yva, Xte, yte, device, epochs=40):
    mu, sd = standardize(Xtr)
    f = lambda X: ((torch.cat(X).float() - mu) / sd).to(device)
    xtr, xva, xte = f(Xtr), f(Xva), f(Xte)
    ytr, yva, yte = (torch.cat(y).to(device) for y in (ytr, yva, yte))
    torch.manual_seed(0)
    lin = nn.Linear(xtr.shape[1], 3).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-2)
    best = (-1, None)
    for ep in range(epochs):
        perm = torch.randperm(len(xtr), device=device)
        for b in perm.split(1024):
            loss = F.cross_entropy(lin(xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            va = (lin(xva).argmax(1) == yva).float().mean().item()
            if va > best[0]:
                best = (va, (lin(xte).argmax(1) == yte).float().cpu())
    return best[0], best[1]


def contact_probe(Ptr, Pva, Pte, device, rank=128, epochs=30):
    """P*: lists of (X (L,d) half, contact (L,L) bool, sep (L,L) int)."""
    mu, sd = standardize([p[0] for p in Ptr])
    d = Ptr[0][0].shape[1]
    torch.manual_seed(0)
    W = nn.Linear(d, 2 * rank, bias=False).to(device)
    bias = nn.Parameter(torch.zeros(1, device=device))
    opt = torch.optim.AdamW(list(W.parameters()) + [bias], lr=1e-3, weight_decay=1e-2)

    def logits(X):
        z = W(((X.float() - mu) / sd).to(device))
        a, b = z[:, :rank], z[:, rank:]
        m = a @ b.T
        return (m + m.T) / (2 * math.sqrt(rank)) + bias

    def precision(P):
        res = []
        with torch.no_grad():
            for X, c, sep in P:
                lg = logits(X).cpu()
                sel = torch.from_numpy(sep >= 24) & torch.triu(torch.ones_like(lg, dtype=torch.bool), 1)
                if sel.sum() == 0:
                    continue
                k = max(1, len(X) // 5)
                top = lg[sel].topk(min(k, int(sel.sum()))).indices
                res.append(torch.from_numpy(c)[sel][top].float().mean().item())
        return res

    best = (-1, None)
    for ep in range(epochs):
        for i in np.random.RandomState(ep).permutation(len(Ptr)):
            X, c, sep = Ptr[i]
            lg = logits(X)
            m = torch.from_numpy(sep >= 6).to(device) & torch.triu(torch.ones_like(lg, dtype=torch.bool), 1)
            y = torch.from_numpy(c).to(device)[m].float()
            loss = F.binary_cross_entropy_with_logits(lg[m], y, pos_weight=torch.tensor(5.0, device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        va = float(np.mean(precision(Pva)))
        if va > best[0]:
            best = (va, precision(Pte))
    return best[0], best[1]


def run_probe(args, data):
    rng = random.Random(0)
    tag = f"{args.test_round}:"
    train_all = [p for p in data if not p["target"].startswith(tag)]
    test = [p for p in data if p["target"].startswith(tag)]
    rng.shuffle(train_all)
    nva = max(5, int(0.15 * len(train_all)))
    val, train = train_all[:nva], train_all[nva:]
    print(f"proteins: train {len(train)}  val {len(val)}  test {len(test)}", flush=True)
    results = {}
    for name in args.models:
        t0 = time.time()
        fz = Featurizer(name, args.device)
        feats = {p["target"]: fz(p["seq"]) for p in data}
        del fz
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None
        print(f"[{name}] features {time.time() - t0:.0f}s", flush=True)
        res = {"layers": {}}
        n_layers = len(next(iter(feats.values())))
        for li in range(n_layers):
            def sel(P, ss=False):
                if ss:
                    X = [feats[p["target"]][li][torch.from_numpy(p["pos"])][torch.from_numpy(p["ss_ok"])] for p in P]
                    y = [torch.from_numpy(p["ss"][p["ss_ok"]]) for p in P]
                    return X, y
                return [(feats[p["target"]][li][torch.from_numpy(p["pos"])], p["contact"], p["sep"]) for p in P]
            q3_va, q3_te = ss_probe(*sel(train, True), *sel(val, True), *sel(test, True), args.device)
            pc_va, pc_te = contact_probe(sel(train), sel(val), sel(test), args.device)
            res["layers"][li] = {"q3_val": q3_va, "q3_test": q3_te.mean().item(),
                                 "p_l5_val": pc_va, "p_l5_test": float(np.mean(pc_te)),
                                 "p_l5_test_per_protein": pc_te}
            print(f"[{name}] layer#{li}  Q3 val {q3_va:.3f} test {q3_te.mean():.3f} | "
                  f"P@L/5 LR val {pc_va:.3f} test {np.mean(pc_te):.3f}", flush=True)
        bq = max(res["layers"], key=lambda k: res["layers"][k]["q3_val"])
        bc = max(res["layers"], key=lambda k: res["layers"][k]["p_l5_val"])
        res["best"] = {"q3_layer": bq, "q3_test": res["layers"][bq]["q3_test"],
                       "p_l5_layer": bc, "p_l5_test": res["layers"][bc]["p_l5_test"]}
        print(f"[{name}] BEST  Q3 {res['best']['q3_test']:.3f}  P@L/5 {res['best']['p_l5_test']:.3f}", flush=True)
        results[name] = res
        del feats
        json.dump({"n_train": len(train), "n_val": len(val), "n_test": len(test), "results": results},
                  open(os.path.join(args.out, "probe.json"), "w"), indent=1)


# ---------------------------------------------------------------- likelihood
@torch.no_grad()
def run_ppl(args, data):
    seqs = [p["seq"] for p in data if set(p["seq"]) <= set(AA20)]
    rng = random.Random(0)
    shuf = ["".join(rng.sample(s, len(s))) for s in seqs]
    comp = np.array([sum(s.count(a) for s in seqs) for a in AA20], float)
    comp /= comp.sum()
    out = {"n_seqs": len(seqs), "n_residues": sum(map(len, seqs)),
           "unigram_bits": float(-(comp * np.log2(comp)).sum()), "uniform_bits": math.log2(20), "models": {}}
    for name in args.models:
        kind, path, rnd = MODELS[name]
        t0 = time.time()
        r = {}
        if kind == "qwen":
            from transformers import AutoModelForImageTextToText, AutoTokenizer
            from qwenfold.qwen_encoder import PREFIX
            tok = AutoTokenizer.from_pretrained(path)
            full = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16)
            lm, head = full.model.language_model.to(args.device).eval(), full.lm_head.to(args.device)
            pre = tok.encode(PREFIX, add_special_tokens=False)
            aa_ids = torch.tensor([tok.encode(" " + a, add_special_tokens=False)[0] for a in AA20], device=args.device)
            pre_raw = tok.encode(PREFIX + "\n", add_special_tokens=False)

            def nll_tokens(s):
                ids = torch.tensor([pre + [int(aa_ids[AA20.index(c)]) for c in s]], device=args.device)
                h = lm(input_ids=ids).last_hidden_state[0, len(pre) - 1: -1]
                lg = head(h).float()
                tgt = ids[0, len(pre):]
                full_nll = F.cross_entropy(lg, tgt, reduction="sum").item()
                lg20 = lg[:, aa_ids]
                t20 = torch.tensor([AA20.index(c) for c in s], device=args.device)
                return full_nll, F.cross_entropy(lg20, t20, reduction="sum").item()

            def nll_raw(s):
                sid = tok.encode(s, add_special_tokens=False)
                ids = torch.tensor([pre_raw + sid], device=args.device)
                h = lm(input_ids=ids).last_hidden_state[0, len(pre_raw) - 1: -1]
                return F.cross_entropy(head(h).float(), ids[0, len(pre_raw):], reduction="sum").item(), len(sid)

            for tag, S in (("real", seqs), ("shuffled", shuf)):
                a = [nll_tokens(s) for s in S]
                b = [nll_raw(s) for s in S]
                n = sum(map(len, S))
                r[tag] = {"token_bits_full_vocab": sum(x[0] for x in a) / n / math.log(2),
                          "token_bits_aa20": sum(x[1] for x in a) / n / math.log(2),
                          "raw_bits": sum(x[0] for x in b) / n / math.log(2),
                          "raw_tokens_per_residue": sum(x[1] for x in b) / n,
                          "per_seq_aa20_bits": [x[1] / len(s) / math.log(2) for x, s in zip(a, S)]}
            del full, lm, head
        else:
            from transformers import AutoTokenizer, EsmForMaskedLM
            tok = AutoTokenizer.from_pretrained(path)
            m = EsmForMaskedLM.from_pretrained(path, dtype=torch.bfloat16).to(args.device).eval()
            aa_ids = torch.tensor(tok.convert_tokens_to_ids(list(AA20)), device=args.device)

            def pll(s):
                ids = torch.tensor([tok.cls_token_id] + tok.convert_tokens_to_ids(list(s)) + [tok.eos_token_id],
                                   device=args.device)
                L, tot = len(s), 0.0
                for st in range(0, L, 32):
                    pos = torch.arange(st, min(L, st + 32), device=args.device) + 1
                    x = ids.repeat(len(pos), 1)
                    x[torch.arange(len(pos)), pos] = tok.mask_token_id
                    lg = m(input_ids=x).logits[torch.arange(len(pos)), pos].float()[:, aa_ids]
                    t = torch.tensor([AA20.index(s[p - 1]) for p in pos.tolist()], device=args.device)
                    tot += F.cross_entropy(lg, t, reduction="sum").item()
                return tot

            for tag, S in (("real", seqs), ("shuffled", shuf)):
                a = [pll(s) for s in S]
                n = sum(map(len, S))
                r[tag] = {"pseudo_bits_aa20": sum(a) / n / math.log(2),
                          "per_seq_aa20_bits": [x / len(s) / math.log(2) for x, s in zip(a, S)]}
            del m
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None
        r["seconds"] = time.time() - t0
        out["models"][name] = r
        print(name, json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.startswith("per_seq")}
                                if isinstance(v, dict) else v for k, v in r.items()}), flush=True)
        json.dump(out, open(os.path.join(args.out, "ppl.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["probe", "ppl"], default="probe")
    ap.add_argument("--models", nargs="+", required=True, choices=list(MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", nargs="+", type=int, default=[14, 15, 16])
    ap.add_argument("--test-round", type=int, default=14, help="probe: CASP round used as the test set (others = training)")
    ap.add_argument("--max-len", type=int, default=1000)
    ap.add_argument("--max-proteins", type=int, default=0, help="per round (smoke tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = []
    for c in args.rounds:
        r = load_round(c, args.max_len)
        if args.max_proteins:
            r = r[: args.max_proteins]
        ss = np.concatenate([p["ss"][p["ss_ok"]] for p in r])
        print(f"CASP{c}: {len(r)} proteins, {sum(len(p['pos']) for p in r)} residues, "
              f"SS C/H/E = {np.bincount(ss, minlength=3) / len(ss)}", flush=True)
        data += r
    (run_probe if args.task == "probe" else run_ppl)(args, data)


if __name__ == "__main__":
    main()
