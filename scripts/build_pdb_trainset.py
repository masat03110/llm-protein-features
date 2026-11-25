"""Build a non-redundant PDB training set released before a date cutoff.

Leakage control: CASP14 targets were released after 2020-05-01, which is also
the PDB cutoff of the ESMFold folding trunk.  For CASP15/16 evaluation, pass a
correspondingly later cutoff (the trunk itself stays at 2020-05).

Output: data/trainset/pdb_<cutoff>_id30.tsv with columns
    entity_id, chain_id, length, resolution, release_date, sequence
plus a deterministic train/valid split column.

Usage:
    python scripts/build_pdb_trainset.py --cutoff 2020-05-01 --min-len 40 --max-len 512
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import requests

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")


def search_representatives(cutoff: str, min_len: int, max_len: int, max_res: float, identity: int) -> list[str]:
    ids, start, rows = [], 0, 10000
    while True:
        query = {
            "query": {"type": "group", "logical_operator": "and", "nodes": [
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_accession_info.initial_release_date", "operator": "less", "value": cutoff}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "entity_poly.rcsb_entity_polymer_type", "operator": "exact_match", "value": "Protein"}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "entity_poly.rcsb_sample_sequence_length", "operator": "range",
                    "value": {"from": min_len, "to": max_len, "include_lower": True, "include_upper": True}}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined", "operator": "less_or_equal", "value": max_res}},
            ]},
            "return_type": "polymer_entity",
            "request_options": {
                "paginate": {"start": start, "rows": rows},
                "results_content_type": ["experimental"],
                "group_by": {"aggregation_method": "sequence_identity", "similarity_cutoff": identity,
                             "ranking_criteria_type": {"sort_by": "rcsb_entry_info.resolution_combined",
                                                       "direction": "asc"}},
                "group_by_return_type": "representatives",
            },
        }
        r = requests.post(SEARCH_URL, json=query, timeout=300)
        r.raise_for_status()
        res = r.json()
        batch = [x["identifier"] for x in res.get("result_set", [])]
        ids.extend(batch)
        total = res.get("group_by_count", res.get("total_count", 0))
        print(f"  search: {len(ids)}/{total}", file=sys.stderr)
        if not batch or len(ids) >= total:
            return ids
        start += rows


def fetch_entities(entity_ids: list[str], batch: int = 300) -> list[dict]:
    q = """query($ids:[String!]!){polymer_entities(entity_ids:$ids){
      rcsb_id
      entity_poly{pdbx_seq_one_letter_code_can}
      rcsb_polymer_entity_container_identifiers{auth_asym_ids}
      entry{rcsb_accession_info{initial_release_date} rcsb_entry_info{resolution_combined}}
    }}"""
    out = []
    for i in range(0, len(entity_ids), batch):
        for attempt in range(5):
            try:
                r = requests.post(GRAPHQL_URL, json={"query": q, "variables": {"ids": entity_ids[i:i + batch]}},
                                  timeout=300)
                r.raise_for_status()
                out.extend(r.json()["data"]["polymer_entities"] or [])
                break
            except Exception as e:  # network hiccup -> retry
                print(f"  retry {attempt}: {e}", file=sys.stderr)
                time.sleep(3 * (attempt + 1))
        print(f"  graphql: {min(i + batch, len(entity_ids))}/{len(entity_ids)}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2020-05-01")
    ap.add_argument("--min-len", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-resolution", type=float, default=2.5)
    ap.add_argument("--identity", type=int, default=30, choices=[30, 40, 50, 70, 90, 95, 100])
    ap.add_argument("--valid-frac", type=float, default=0.03)
    ap.add_argument("--outdir", default="data/trainset")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ids = search_representatives(args.cutoff, args.min_len, args.max_len, args.max_resolution, args.identity)
    ents = fetch_entities(ids)
    out_path = os.path.join(args.outdir, f"pdb_{args.cutoff}_id{args.identity}.tsv")
    n_keep = 0
    with open(out_path, "w") as fh:
        fh.write("entity_id\tchain_id\tlength\tresolution\trelease_date\tsplit\tsequence\n")
        for e in sorted(ents, key=lambda e: e["rcsb_id"]):
            seq = (e["entity_poly"] or {}).get("pdbx_seq_one_letter_code_can") or ""
            seq = seq.replace("\n", "").upper()
            if not (args.min_len <= len(seq) <= args.max_len):
                continue
            if sum(c not in CANONICAL for c in seq) > 0.05 * len(seq):
                continue
            chains = (e["rcsb_polymer_entity_container_identifiers"] or {}).get("auth_asym_ids") or ["A"]
            info = e["entry"]
            res = (info["rcsb_entry_info"].get("resolution_combined") or [None])[0]
            date = info["rcsb_accession_info"]["initial_release_date"][:10]
            assert date < args.cutoff, (e["rcsb_id"], date)
            # deterministic split from the id hash
            h = int(hashlib.md5(e["rcsb_id"].encode()).hexdigest(), 16) % 10000
            split = "valid" if h < args.valid_frac * 10000 else "train"
            fh.write(f"{e['rcsb_id']}\t{chains[0]}\t{len(seq)}\t{res}\t{date}\t{split}\t{seq}\n")
            n_keep += 1
    meta = vars(args) | {"n_representatives": len(ids), "n_kept": n_keep, "file": out_path,
                         "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(out_path.replace(".tsv", ".meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
