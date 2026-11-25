"""Download CASP target sequences, native structures and selected groups' models.

Default: CASP14 with AlphaFold2 (group 427) models.  The script discovers file
names from the predictioncenter.org directory listings, so it does not depend
on exact archive names.

    python scripts/download_casp.py --casp 14 --groups 427
    python scripts/download_casp.py --casp 15 --groups 270 446   # NBIS-AF2-standard, ColabFold
    python scripts/download_casp.py --casp 16 --groups 304 145   # AF3-server, colabfold_baseline

Layout written under data/casp<N>/:
    sequences.fasta          all target sequences (T*, H* ...)
    natives/                 whole-target and domain native PDBs (T1024.pdb, T1024-D1.pdb ...)
    predictions/<group>/     <target>.pdb  (model 1 of that group, by default)

NOTE: only public targets have natives; some CASP targets were never released.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import tarfile
import time

import requests

BASE = "https://predictioncenter.org/download_area/CASP{n}/"


def listing(url: str) -> list[str]:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return sorted(set(h for h in re.findall(r'href="([^"?/][^"]*)"', r.text) if not h.startswith("http")))


def get(url: str, retries: int = 3) -> bytes:
    print(f"  GET {url}", file=sys.stderr)
    for i in range(retries):
        try:
            r = requests.get(url, timeout=1800)
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            if i == retries - 1:
                raise
            print(f"    retry {i + 1}/{retries - 1}: {e}", file=sys.stderr)
            time.sleep(10 * (i + 1))


def extract_pdbs(blob: bytes, outdir: str, name_filter=None, rename=None, overwrite=True) -> int:
    n = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            if name_filter and not name_filter(name):
                continue
            data = tar.extractfile(m).read()
            out = rename(name) if rename else (name if name.endswith(".pdb") else name + ".pdb")
            dst = os.path.join(outdir, out)
            if not overwrite and os.path.exists(dst):
                if open(dst, "rb").read() != data:
                    print(f"    WARNING: {out} already exists with different content; kept the first one", file=sys.stderr)
                continue
            with open(dst, "wb") as fh:  # basename only: no path traversal
                fh.write(data)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--casp", type=int, default=14)
    ap.add_argument("--groups", nargs="*", default=["427"], help="CASP group numbers (427 = AlphaFold2 @ CASP14)")
    ap.add_argument("--model-index", type=int, default=1, help="which of the 5 submitted models to keep")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--natives-only", action="store_true", help="skip the (large) prediction archives")
    args = ap.parse_args()

    base = BASE.format(n=args.casp)
    out = args.outdir or f"data/casp{args.casp}"
    nat_dir = os.path.join(out, "natives")
    os.makedirs(nat_dir, exist_ok=True)

    # 1. sequences
    # CASP14/15: casp<N>.seq.txt   CASP16: casp16.T1.seq.txt (monomers), .H1 (oligomers), .RDM1 (RNA)
    seq_files = [f for f in listing(base + "sequences/") if f.endswith(".txt") or f.endswith(".fasta")]
    seq_file = next((f for f in seq_files if re.search(rf"casp{args.casp}\.(T1\.)?seq", f)), None)
    if seq_file is None:
        sys.exit(f"no monomer sequence file found; listing: {seq_files}")
    print(f"  sequences: {seq_file}", file=sys.stderr)
    with open(os.path.join(out, "sequences.fasta"), "wb") as fh:
        fh.write(get(base + "sequences/" + seq_file))

    # 2. natives: whole targets + domain definitions
    tgt_files = [f for f in listing(base + "targets/") if f.endswith((".tar.gz", ".tgz"))]
    # CASP14: casp14.targets.T.public_*.tar.gz (whole) + casp14.targets.T-dom.public_*.tar.gz (domains)
    # CASP15: casp15.targets.TS-domains.public_*.tar.gz (domains only)
    # CASP16: casp16.targets_monomer.tgz (whole) + casp16.targets_monomer_trimmed2domains.tgz (domains)
    # Oligomer / RNA / ligand archives are skipped on purpose (this project predicts monomers).
    wanted = [f for f in tgt_files
              if re.search(r"targets\.(T|T-dom|TS-domains)\.public|targets_monomer(_trimmed2domains)?\.t", f)]
    if not wanted:
        sys.exit(f"no monomer target archive matched; listing: {tgt_files}")
    for f in wanted:
        # keep a pristine per-archive copy too, so file-name clashes between archives can be inspected
        blob = get(base + "targets/" + f)
        raw_dir = os.path.join(out, "natives_raw", re.sub(r"\.(tar\.gz|tgz)$", "", f))
        os.makedirs(raw_dir, exist_ok=True)
        extract_pdbs(blob, raw_dir)
        n = extract_pdbs(blob, nat_dir, overwrite=False,
                         name_filter=lambda s: re.match(r"^T\d{4}", s) is not None)
        print(f"  {f}: {n} native files", file=sys.stderr)

    if args.natives_only:
        print("done (natives only):", out)
        return

    # 3. predictions of the requested groups (one archive per target)
    pred_root = base + "predictions/regular/"
    archives = [f for f in listing(pred_root) if re.match(r"^T\d{4}.*\.tar\.gz$", f)]
    for g in args.groups:
        os.makedirs(os.path.join(out, "predictions", f"group{g}"), exist_ok=True)
    for f in archives:
        target = f.split(".")[0]
        done = [os.path.exists(os.path.join(out, "predictions", f"group{g}", f"{target}.pdb")) for g in args.groups]
        if all(done):
            continue  # resumable: already extracted for every requested group
        blob = get(pred_root + f)
        for g in args.groups:
            pat = re.compile(rf"^{target}TS{g}_{args.model_index}$")
            n = extract_pdbs(blob, os.path.join(out, "predictions", f"group{g}"),
                             name_filter=lambda s: pat.match(s) is not None,
                             rename=lambda s: f"{target}.pdb")
            if n == 0:
                print(f"  {target}: no model from group {g}", file=sys.stderr)
    print("done:", out)


if __name__ == "__main__":
    main()
