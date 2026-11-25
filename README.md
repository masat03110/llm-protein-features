# General-Purpose Local LLMs as Feature Extractors for Protein Structure Prediction

Code and results to reproduce the experiments in the poster

> **General-Purpose Local LLMs as Feature Extractors for Protein Structure Prediction:
> A Systematic Assessment of Accuracy, Memory, and Encoded Structural Information on CASP16**
> Masahito Tsukahara, Tetsuo Shibuya (The University of Tokyo)

We replace the protein language model of ESMFold (ESM2-3B) with a general-purpose LLM (Qwen3.5, 0.8B–9B) and keep
ESMFold's structure module frozen. The new language model and a small adapter are trained by distillation to
reproduce the ESM2-3B features inside ESMFold. Every model is evaluated on CASP16 targets.

```
Original ESMFold : sequence -> ESM2-3B --------------------------> features -> structure module (frozen) -> 3D structure
This study       : sequence -> Qwen3.5 / small ESM2 (+ adapter) -> features -> structure module (frozen) -> 3D structure
```

## Main results (CASP16, 74 evaluation units, TM-score)

| Model | TM-score | GPU memory (mean) |
|---|---|---|
| ESMFold (ESM2-3B) | 0.765 | 9.0 GB |
| ESM2-650M (frozen + adapter, int8-quantized) | 0.725 | 1.9 GB |
| ESM2-35M (1,000 steps) | 0.571 | 3.8 GB |
| Qwen3.5-0.8B / 2B / 9B (1,000 steps) | 0.275 / 0.271 / 0.277 | 5.2 / 7.3 / 18.7 GB |
| No language model (structure module only) | 0.131 | 3.7 GB |

- Scaling Qwen from 0.8B to 9B, or training for 6,000 steps, does not close the gap.
- Pretrained and randomly initialized Qwen give the same accuracy, and feeding ESM2 features into Qwen
  (as in Calpha) does not improve on ESM2 alone.
- Linear probes on frozen features find little structural information in Qwen (long-range contacts at the
  one-hot baseline).
- Training on proteins for 6,000 steps lowers MMLU-Pro accuracy (0.8B: 19.7% -> 14.6%; 9B: 54.0% -> 42.0%).

All numbers above can be recomputed without a GPU from the files in `results/` (see
[Checking the statistics without a GPU](#checking-the-statistics-without-a-gpu)).

## Repository layout

| Path | Contents |
|---|---|
| `qwenfold/esmfold_utils.py` | Splits ESMFold into the language-model part and the structure module |
| `qwenfold/qwen_encoder.py` | Qwen encoder (one token per residue, LoRA, `causal` / `echo` / `bidir` reading, optional ESM2 feature injection, random initialization) |
| `qwenfold/esm_encoder.py` | Small ESM2 encoders (LoRA or frozen) and the ESM2 featurizer used for injection |
| `qwenfold/predictors.py` | Common prediction interface for ESMFold / replaced models / structure module only |
| `qwenfold/quant.py` | Weight-only int8 quantization |
| `qwenfold/metrics.py`, `qwenfold/structure_io.py` | TM-score, GDT_TS, lDDT (NumPy) and PDB reading |
| `qwenfold/profiling.py` | Time and peak-memory measurement |
| `scripts/fetch_hf_repo.py` | Downloads a Hugging Face model at a given revision |
| `scripts/build_pdb_trainset.py` | PDB training set (release-date cutoff, 30% identity) |
| `scripts/download_casp.py` | CASP sequences, experimental structures and server predictions |
| `scripts/cache_esm_targets.py` | Teacher features (ESM2-3B features inside ESMFold) for distillation |
| `scripts/train_distill.py` | Distillation training |
| `scripts/predict.py`, `jobs/casp_predict_set.sh` | Structure prediction on CASP targets (with time and GPU memory) |
| `scripts/score.py` | Scores all methods with the same code |
| `scripts/probe_lm.py` | Linear probes (secondary structure, contacts) on frozen features |
| `scripts/eval_text.py` | MMLU-Pro accuracy with the trained LoRA enabled |
| `analysis/poster_stats.py` | Mean TM-score with 95% bootstrap CIs and paired tests against ESMFold |
| `splits/` | The exact training set used (PDB entity IDs, chains, sequences) |
| `results/` | Per-unit scores, statistics, probe results and MMLU-Pro results behind the poster |
| `requirements-lock.txt` | Exact package versions of the environment used |

## Setup

### Environment

We used Python 3.12.16, PyTorch 2.14.1 with CUDA 12.6 (cuDNN 9.10.2), transformers 5.23.0, peft 0.21.3 and
flash-linear-attention 0.6.0 on Linux. `requirements-lock.txt` lists every package and version.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
pip install -e . --no-deps
pip install scipy   # analysis/poster_stats.py
```

The versions matter for the randomly initialized Qwen runs (`--qwen-random-init`): their frozen weights are not
saved but rebuilt from the seed, so a different PyTorch / transformers version gives different weights and the
checkpoint cannot be reused.

The scripts in `jobs/` call `.venv/bin/python`; create the virtual environment at `.venv` or edit them.

### Hardware, disk and time (approximate)

- GPU: 24 GB for the 0.8B and 2B models and the ESM2 models; 40 GB for the 9B model (peak about 22 GB).
- Disk: about 50 GB of model weights, 1.8 GB of teacher features for 4,000 sequences and 7.7 GB for all
  sequences.
- Time on one GPU: Qwen-0.8B, 1,000 steps: about 20 min; 6,000 steps: about 1.7 h. Qwen-9B, 1,000 steps:
  about 35 min; 6,000 steps: about 3 h. Prediction on the CASP16 targets: about 15–40 s per target.

### Models (pinned revisions)

| Model | Hugging Face repo | Revision (commit) |
|---|---|---|
| ESMFold | `facebook/esmfold_v1` | `75a3841ee059df2bf4d56688166c8fb459ddd97a` |
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B-Base` | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| Qwen3.5-2B | `Qwen/Qwen3.5-2B-Base` | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B-Base` | `68c46c4b3498877f3ef123c856ecfde50c39f404` |
| ESM2-35M | `facebook/esm2_t12_35M_UR50D` | `6fbf070e65b0b7291e7bbcd451118c216cff79d8` |
| ESM2-150M | `facebook/esm2_t30_150M_UR50D` | `a695f6045e2e32885fa60af20c13cb35398ce30c` |
| ESM2-650M | `facebook/esm2_t33_650M_UR50D` | `08e4846e537177426273712802403f7ba8261b6c` |
| ESM2-3B | `facebook/esm2_t36_3B_UR50D` | `476b639933c8baad5ad09a60ac1a87f987b656fc` |
| MMLU-Pro (dataset) | `TIGER-Lab/MMLU-Pro` | `b189ec765aa7ed75c8acfea42df31fdae71f97be` |

For ESMFold and Qwen3.5-0.8B the weight files were checked against these commits by their SHA-256; the other
revisions are the ones recorded at download time.

```bash
python scripts/fetch_hf_repo.py facebook/esmfold_v1    models/esmfold_v1    --revision 75a3841ee059df2bf4d56688166c8fb459ddd97a
python scripts/fetch_hf_repo.py Qwen/Qwen3.5-0.8B-Base models/Qwen3.5-0.8B-Base --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
python scripts/fetch_hf_repo.py Qwen/Qwen3.5-2B-Base   models/Qwen3.5-2B-Base   --revision b1485b2fa6dfa1287294f269f5fb618e03d52d7c
python scripts/fetch_hf_repo.py Qwen/Qwen3.5-9B-Base   models/Qwen3.5-9B-Base   --revision 68c46c4b3498877f3ef123c856ecfde50c39f404
python scripts/fetch_hf_repo.py facebook/esm2_t12_35M_UR50D  models/esm2_t12_35M_UR50D  --revision 6fbf070e65b0b7291e7bbcd451118c216cff79d8
python scripts/fetch_hf_repo.py facebook/esm2_t30_150M_UR50D models/esm2_t30_150M_UR50D --revision a695f6045e2e32885fa60af20c13cb35398ce30c
python scripts/fetch_hf_repo.py facebook/esm2_t33_650M_UR50D models/esm2_t33_650M_UR50D --revision 08e4846e537177426273712802403f7ba8261b6c
python scripts/fetch_hf_repo.py facebook/esm2_t36_3B_UR50D   models/esm2_t36_3B_UR50D   --revision 476b639933c8baad5ad09a60ac1a87f987b656fc
```

### Data

```bash
# Training set: use the released split (the RCSB search behind build_pdb_trainset.py changes over time)
mkdir -p data/trainset && cp splits/pdb_2020-05-01_id30.* data/trainset/
#   (to rebuild a new one instead: python scripts/build_pdb_trainset.py --cutoff 2020-05-01)

# Teacher features: 4,000 sequences (model comparisons) and all 17,468 sequences (6,000-step runs)
python scripts/cache_esm_targets.py --tsv data/trainset/pdb_2020-05-01_id30.tsv --limit 4000 --out data/esm_targets
python scripts/cache_esm_targets.py --tsv data/trainset/pdb_2020-05-01_id30.tsv --out data/esm_targets_full

# CASP targets (CASP14/15 train the linear probes; CASP16 is the evaluation set)
python scripts/download_casp.py --casp 14 --groups 427      # 427 = AlphaFold2
python scripts/download_casp.py --casp 15 --natives-only
python scripts/download_casp.py --casp 16 --groups 304 145  # 304 = AlphaFold3 server, 145 = ColabFold

# MMLU-Pro (for eval_text.py), pinned revision
python -c "from huggingface_hub import list_repo_files, hf_hub_download as d; r='b189ec765aa7ed75c8acfea42df31fdae71f97be'; [d('TIGER-Lab/MMLU-Pro', f, repo_type='dataset', revision=r, local_dir='data/mmlu_pro') for f in list_repo_files('TIGER-Lab/MMLU-Pro', repo_type='dataset', revision=r) if f.endswith('.parquet')]"
```

`splits/pdb_2020-05-01_id30.tsv` has 17,468 PDB chains released before 2020-05-01 (40–512 residues, resolution
<= 2.5 A, one representative per 30% identity cluster); `.meta.json` records the options. The teacher features
for the 4,000-sequence set are taken from the first 4,000 training rows of this file.

## Reproducing the experiments

All runs use seed 0, bidirectional reading for Qwen (`--mode bidir`) and LoRA r = 16.

```bash
C="--steps 1000 --seed 0 --lora-compute autocast --grad-ckpt on --fold-eval 8 --resume --targets data/esm_targets"
L="--steps 6000 --eval-every 1000 --seed 0 --lora-compute autocast --grad-ckpt on --fold-eval 8 --resume --targets data/esm_targets_full"
```

### Training

| Run name | Model | Command |
|---|---|---|
| `pre_1k` | Qwen3.5-0.8B | `python scripts/train_distill.py --run runs/pre_1k --mode bidir $C` |
| `qwen2b` | Qwen3.5-2B | `python scripts/train_distill.py --run runs/qwen2b --qwen models/Qwen3.5-2B-Base --mode bidir $C` |
| `qwen9b` | Qwen3.5-9B | `python scripts/train_distill.py --run runs/qwen9b --qwen models/Qwen3.5-9B-Base --mode bidir $C` |
| `esm35`, `esm150`, `esm650` | ESM2-35M / 150M / 650M (LoRA) | `python scripts/train_distill.py --run runs/esm35 --encoder esm2 --esm models/esm2_t12_35M_UR50D $C` (`esm2_t30_150M_UR50D`, `esm2_t33_650M_UR50D` likewise) |
| `rand_1k` | Qwen-0.8B, random init | `python scripts/train_distill.py --run runs/rand_1k --qwen-random-init 0 --mode bidir $C` |
| `qwen_6k` | Qwen-0.8B, 6,000 steps | `python scripts/train_distill.py --run runs/qwen_6k --mode bidir $L` |
| `rand_6k` | Qwen-0.8B random init, 6,000 steps | `python scripts/train_distill.py --run runs/rand_6k --qwen-random-init 0 --mode bidir $L` |
| `frozen35`, `frozen650` | ESM2 only (frozen + adapter) | `python scripts/train_distill.py --run runs/frozen35 --encoder esm2 --esm models/esm2_t12_35M_UR50D --lora-r 0 $C` (650M likewise) |
| `inj35`, `inj650` | ESM2 -> Qwen-0.8B (Calpha-style injection) | `python scripts/train_distill.py --run runs/inj35 --esm-inject models/esm2_t12_35M_UR50D --mode bidir $C` (650M likewise) |
| `qwen9b_6k` | Qwen3.5-9B, 6,000 steps | `python scripts/train_distill.py --run runs/qwen9b_6k --qwen models/Qwen3.5-9B-Base --mode bidir $L` |

### Prediction on CASP16

`jobs/casp_predict_set.sh 16 <method> <checkpoint|-> <run> <trunk-dtype:lm-quant:trunk-quant> ...` predicts all
CASP16 targets that have an experimental structure (up to 1,000 residues) and writes
`outputs/casp16/<run>_<trunk-dtype>_lm<lm-quant>_tr<trunk-quant>/`.

```bash
bash jobs/casp_predict_set.sh 16 esmfold   - esmfold   fp32:none:none bf16:int8:int8
bash jobs/casp_predict_set.sh 16 trunkonly - trunkonly fp32:none:none
for r in pre_1k qwen2b qwen9b esm650 rand_1k qwen_6k rand_6k frozen35 inj35 inj650 qwen9b_6k; do
  bash jobs/casp_predict_set.sh 16 qwenfold runs/$r/best.pt $r fp32:none:none
done
for r in esm35 esm150; do bash jobs/casp_predict_set.sh 16 qwenfold runs/$r/best.pt $r fp32:none:none bf16:none:none; done
bash jobs/casp_predict_set.sh 16 qwenfold runs/frozen650/best.pt frozen650 fp32:none:none bf16:int8:int8
```

### Scoring

Each method gets a name in the score files; the table maps the names to the runs and to the poster.

| Name in `results/` | Prediction folder (`outputs/casp16/`) | Poster |
|---|---|---|
| `ESMFold` | `esmfold_fp32_lmnone_trnone` | Results 1–3 |
| `ESMFold_int8` | `esmfold_bf16_lmint8_trint8` | Result 3 |
| `TrunkOnly` | `trunkonly_fp32_lmnone_trnone` | Results 1, 2 |
| `AF3server_g304` | `data/casp16/predictions/group304` | Result 1 |
| `ESM35_1k`, `ESM150_1k`, `ESM650_1k` | `esm35_…`, `esm150_…`, `esm650_fp32_lmnone_trnone` | Result 1 |
| `ESM35_1k_bf16`, `ESM150_1k_bf16` | `esm35_bf16_lmnone_trnone`, `esm150_bf16_lmnone_trnone` | Result 3 |
| `Pre_1k`, `Qwen2B_1k`, `Qwen9B_1k` | `pre_1k_…`, `qwen2b_…`, `qwen9b_fp32_lmnone_trnone` | Results 1–3 |
| `Rand_1k`, `Qwen_6k`, `Rand_6k` | `rand_1k_…`, `qwen_6k_…`, `rand_6k_fp32_lmnone_trnone` | Results 2, 3 |
| `Frozen35`, `Frozen650` | `frozen35_…`, `frozen650_fp32_lmnone_trnone` | Results 2, 3 |
| `Frozen650_int8` | `frozen650_bf16_lmint8_trint8` | Result 3 |
| `Qwen08+ESM35`, `Qwen08+ESM650` | `inj35_…`, `inj650_fp32_lmnone_trnone` | Result 2 (ESM2 -> Qwen) |
| `Qwen9B_6k` | `qwen9b_6k_fp32_lmnone_trnone` | (not on the poster) |

```bash
python scripts/score.py --natives data/casp16/natives \
  --pred AF3server_g304=data/casp16/predictions/group304 \
  --pred ColabFold_g145=data/casp16/predictions/group145 \
  --pred ESMFold=outputs/casp16/esmfold_fp32_lmnone_trnone \
  --pred Pre_1k=outputs/casp16/pre_1k_fp32_lmnone_trnone \
  --out outputs/casp16/scores.csv          # one --pred per row of the table above

python analysis/poster_stats.py --csv 16=outputs/casp16/scores.csv --out outputs/stats.csv
```

The 74 evaluation units are the CASP16 domains with a released experimental structure, from targets of up to
1,000 residues, that every one of our methods predicted. The CASP server predictions are excluded when the
common units are chosen, so adding or removing one of our methods can change the unit count.

### Follow-up 1: linear probes

```bash
python scripts/probe_lm.py --test-round 16 --out outputs/probe16 \
  --models onehot esm35m esm150m esm650m esm3b qwen0.8b_rand qwen0.8b qwen2b qwen9b
```

Probes are trained on CASP14/15 targets and evaluated on CASP16 targets; the best layer of each model is selected
on held-out CASP14/15 targets.

### Follow-up 2: MMLU-Pro

```bash
python scripts/eval_text.py --qwen models/Qwen3.5-0.8B-Base --out outputs/text/base08.json      # before training
python scripts/eval_text.py --ckpt runs/pre_1k/best.pt        --out outputs/text/pre_1k.json      # trained LoRA enabled
python scripts/eval_text.py --ckpt runs/qwen_6k/best.pt       --out outputs/text/qwen_6k.json
python scripts/eval_text.py --qwen models/Qwen3.5-9B-Base     --out outputs/text/base9b.json
python scripts/eval_text.py --ckpt runs/qwen9b/best.pt        --out outputs/text/qwen9b.json
python scripts/eval_text.py --ckpt runs/qwen9b_6k/best.pt     --out outputs/text/qwen9b_6k.json
```

A fixed set of 350 questions (seed 0) is used.

## Results behind the poster

| File | Contents |
|---|---|
| `results/casp16/scores_all.csv` | Per-unit CASP16 scores of all methods except the structure-module-only baseline |
| `results/casp16/scores_trunkonly.csv` | Per-unit CASP16 scores of the structure-module-only baseline |
| `results/casp16/scores_qwen9b_6k.csv` | Per-unit CASP16 scores of Qwen3.5-9B trained for 6,000 steps |
| `results/stats.csv` | Statistics for CASP14, CASP15 and CASP16 (mean TM-score, 95% CI, GPU memory, paired difference from ESMFold) |
| `results/probe16.json` | Linear-probe results (Q3 and long-range contact precision per model and layer) |
| `results/mmlu_pro/*.json` | MMLU-Pro accuracy and per-question answers (file names follow the run names above) |

### Checking the statistics without a GPU

```bash
python analysis/poster_stats.py --csv 16=results/casp16/scores_all.csv --csv 16=results/casp16/scores_trunkonly.csv \
  --out stats16.csv
```

This reproduces the CASP16 rows of `results/stats.csv` exactly.

## Notes

- The ESMFold structure module was trained on PDB entries up to May 2020; our training set uses the same cutoff,
  and all CASP16 targets were released later.
- All methods, including the CASP server predictions, are scored with the same code (`scripts/score.py`).
  TM-score agrees with TM-align (`tmtools`) to three decimal places.
- Seeds are fixed, but GPU kernels are not run in deterministic mode, so retraining can give slightly different
  numbers.
- Model weights, teacher features and predicted structures are not included in this repository.

## License

MIT License (see `LICENSE`). ESMFold, ESM2, Qwen3.5 and MMLU-Pro are distributed under their own licenses.
