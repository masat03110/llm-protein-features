"""Resumable, parallel (HTTP range) download of a Hugging Face repo into a plain folder.

Usage (pin --revision to the commit listed in README.md to reproduce the paper):
    python scripts/fetch_hf_repo.py facebook/esmfold_v1 models/esmfold_v1 --revision <commit> --workers 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

CHUNK = 32 * 1024 * 1024


def fetch_range(url, path, start, end):
    part = f"{path}.part{start}"
    want = end - start + 1
    for attempt in range(20):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have >= want:
            return part
        try:
            with requests.get(url, headers={"Range": f"bytes={start + have}-{end}"}, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(part, "ab") as fh:
                    for b in r.iter_content(1 << 20):
                        fh.write(b)
        except Exception as e:
            print(f"retry {os.path.basename(part)} ({attempt}): {e}", file=sys.stderr)
            time.sleep(min(30, 2 * (attempt + 1)))
    raise RuntimeError(f"failed: {part}")


def fetch_file(repo, fname, outdir, workers, revision="main"):
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{fname}"
    path = os.path.join(outdir, fname)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    head = requests.head(url, allow_redirects=True, timeout=60)
    size = int(head.headers.get("Content-Length", 0))
    if os.path.exists(path) and os.path.getsize(path) == size:
        print(f"have {fname}")
        return
    if size < CHUNK:
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        open(path, "wb").write(r.content)
        print(f"got {fname} ({len(r.content)} B)")
        return
    ranges = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch_range, url, path, s, e) for s, e in ranges]
        for f in futs:
            f.result()
            done += 1
            mb = done * CHUNK / 1e6
            print(f"{fname}: {done}/{len(ranges)} chunks, {mb / (time.time() - t0):.2f} MB/s", flush=True)
    with open(path + ".tmp", "wb") as out:
        for s, _ in ranges:
            with open(f"{path}.part{s}", "rb") as fh:
                while b := fh.read(1 << 24):
                    out.write(b)
    assert os.path.getsize(path + ".tmp") == size
    os.replace(path + ".tmp", path)
    for s, _ in ranges:
        os.remove(f"{path}.part{s}")
    print(f"got {fname} ({size / 1e9:.2f} GB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("outdir")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--revision", default="main", help="branch, tag or commit hash")
    args = ap.parse_args()
    info = requests.get(f"https://huggingface.co/api/models/{args.repo}/revision/{args.revision}", timeout=60).json()
    for s in info["siblings"]:
        fetch_file(args.repo, s["rfilename"], args.outdir, args.workers, args.revision)


if __name__ == "__main__":
    main()
