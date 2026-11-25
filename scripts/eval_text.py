"""General-knowledge check of the Qwen backbone with and without the protein-trained LoRA.

    python scripts/eval_text.py --qwen models/Qwen3.5-0.8B-Base --out outputs/text/base08.json
    python scripts/eval_text.py --ckpt runs/bidir_s0/best.pt --out outputs/text/bidir_s0.json

Zero-shot MMLU-Pro (TIGER-Lab/MMLU-Pro, test split), a fixed random subset (--n, --seed), scored by
which option letter (A..J) gets the highest next-token logit after "Answer:".  Calpha (arXiv:2611.20986)
reports 77.4% -> 70.3% on a 350-item MMLU-Pro subset after protein training; this measures the same kind
of change for our LoRA.  The absolute numbers are not comparable with Calpha's (different prompt/scoring).
With --ckpt, the LoRA weights of that encoder checkpoint are active (ESM injection is not used: text only).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import pandas as pd
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from qwenfold.qwen_encoder import QwenProteinEncoder, load_text_backbone

LETTERS = "ABCDEFGHIJ"


def lm_head_weight(qwen_path: str, backbone) -> torch.Tensor:
    """lm_head from the checkpoint files; tied models fall back to the input embedding."""
    idx = os.path.join(qwen_path, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else \
        [os.path.basename(f) for f in glob.glob(os.path.join(qwen_path, "*.safetensors"))]
    for f in files:
        with safe_open(os.path.join(qwen_path, f), framework="pt") as fh:
            for k in fh.keys():
                if k.endswith("lm_head.weight"):
                    return fh.get_tensor(k)
    return backbone.get_input_embeddings().weight.detach()


def prompt(row) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(row["options"]))
    return (f"The following is a multiple choice question about {row['category']}. "
            f"Answer with the letter of the correct option.\n\nQuestion: {row['question']}\n"
            f"Options:\n{opts}\nAnswer:")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", default=None, help="plain Qwen (no LoRA)")
    ap.add_argument("--ckpt", default=None, help="encoder checkpoint: its Qwen with the trained LoRA")
    ap.add_argument("--data", default="data/mmlu_pro")
    ap.add_argument("--n", type=int, default=350)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        assert ck.get("encoder") != "esm2", "not a Qwen checkpoint"
        hp = dict(ck["hparams"])
        hp.pop("esm_inject", None)  # text only: the ESM featuriser is not needed
        enc = QwenProteinEncoder(ck["qwen_path"], dtype=torch.bfloat16, gradient_checkpointing=False, **hp)
        missing = enc.load_state_dict({k: v for k, v in ck["state_dict"].items() if not k.startswith("esm_inject.")},
                                      strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
        backbone, qwen_path = enc.backbone, ck["qwen_path"]
    else:
        qwen_path = args.qwen
        backbone = load_text_backbone(qwen_path, torch.bfloat16)
    backbone = backbone.to(dev).eval()
    tok = AutoTokenizer.from_pretrained(qwen_path)
    W = lm_head_weight(qwen_path, backbone).to(dev, torch.bfloat16)
    letter_ids = []
    for c in LETTERS:
        t = tok.encode(" " + c, add_special_tokens=False)
        assert len(t) == 1, c
        letter_ids.append(t[0])

    files = sorted(glob.glob(os.path.join(args.data, "**", "test-*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rows = df.iloc[sorted(random.Random(args.seed).sample(range(len(df)), args.n))]

    recs, correct = [], 0
    for _, r in rows.iterrows():
        ids = tok(prompt(r), return_tensors="pt").input_ids[:, -args.max_tokens:].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = backbone(input_ids=ids).last_hidden_state[0, -1]
        k = len(r["options"])
        logits = (W[letter_ids[:k]] @ h.to(W.dtype)).float()
        pred = LETTERS[int(logits.argmax())]
        ok = pred == r["answer"]
        correct += ok
        recs.append({"question_id": int(r["question_id"]), "category": r["category"], "answer": r["answer"],
                     "pred": pred, "correct": bool(ok)})
    res = {"qwen": qwen_path, "ckpt": args.ckpt, "n": len(recs), "accuracy": correct / len(recs),
           "seed": args.seed, "items": recs}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "items"}))


if __name__ == "__main__":
    main()
