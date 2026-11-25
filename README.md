# General-Purpose Local LLMs as Feature Extractors for Protein Structure Prediction

Code to reproduce the experiments in the poster

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
| ESM2-35M (1,000 steps) | 0.571 | – |
| Qwen3.5-0.8B / 2B / 9B (1,000 steps) | 0.275 / 0.271 / 0.277 | 5.2 / 7.3 / 18.7 GB |
| No language model (structure module only) | 0.131 | – |

- Scaling Qwen from 0.8B to 9B, or training for 6,000 steps, does not close the gap.
- Pretrained and randomly initialized Qwen give the same accuracy, and feeding ESM2 features into Qwen
  (as in Calpha) does not improve on ESM2 alone.
- Linear probes on frozen features find little structural information in Qwen (long-range contacts at the
  one-hot baseline).
- Training on proteins for 6,000 steps lowers MMLU-Pro accuracy (0.8B: 19.7% -> 14.6%; 9B: 54.0% -> 42.0%).

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
| `scripts/fetch_hf_repo.py` | Downloads Hugging Face models |
| `scripts/build_pdb_trainset.py` | PDB training set (release-date cutoff, 30% identity) |
| `scripts/download_casp.py` | CASP sequences, experimental structures and server predictions |
| `scripts/cache_esm_targets.py` | Teacher features (ESM2-3B features inside ESMFold) for distillation |
| `scripts/train_distill.py` | Distillation training |
| `scripts/predict.py`, `jobs/casp_predict_set.sh` | Structure prediction on CASP targets (with time and GPU memory) |
| `scripts/score.py` | Scores all methods with the same code |
| `scripts/probe_lm.py` | Linear probes (secondary structure, contacts) on frozen features |
| `scripts/eval_text.py` | MMLU-Pro accuracy with the trained LoRA enabled |
| `analysis/poster_stats.py` | Mean TM-score with 95% bootstrap CIs and paired tests against ESMFold |

## Setup

Python >= 3.11 and a CUDA GPU. Training used 24 GB GPUs for the 0.8B and 2B models and a 40 GB GPU for the 9B model
(peak about 22 GB).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install scipy pyarrow huggingface_hub   # analysis, MMLU-Pro and downloads
```

The scripts in `jobs/` call `.venv/bin/python`; create the virtual environment at `.venv` or edit them.

### Models

```bash
python scripts/fetch_hf_repo.py facebook/esmfold_v1        models/esmfold_v1
python scripts/fetch_hf_repo.py Qwen/Qwen3.5-0.8B-Base     models/Qwen3.5-0.8B-Base
python scripts/fetch_hf_repo.py Qwen/Qwen3.5-2B-Base       models/Qwen3.5-2B-Base
python scripts/fetch_hf_repo.py Qwen/Qwen3.5-9B-Base       models/Qwen3.5-9B-Base
for m in esm2_t12_35M_UR50D esm2_t30_150M_UR50D esm2_t33_650M_UR50D esm2_t36_3B_UR50D; do
  python scripts/fetch_hf_repo.py facebook/$m models/$m
done
```

### Data

```bash
# Training set: PDB chains released before 2020-05-01, 30% sequence identity
python scripts/build_pdb_trainset.py --cutoff 2020-05-01

# Teacher features: 4,000 sequences (model comparisons) and all sequences (6,000-step runs)
python scripts/cache_esm_targets.py --tsv data/trainset/pdb_2020-05-01_id30.tsv --limit 4000 --out data/esm_targets
python scripts/cache_esm_targets.py --tsv data/trainset/pdb_2020-05-01_id30.tsv --out data/esm_targets_full

# CASP targets (CASP14/15 train the linear probes; CASP16 is the evaluation set)
python scripts/download_casp.py --casp 14 --groups 427      # 427 = AlphaFold2
python scripts/download_casp.py --casp 15 --natives-only
python scripts/download_casp.py --casp 16 --groups 304      # 304 = AlphaFold3 server

# MMLU-Pro (for eval_text.py)
python -c "from huggingface_hub import list_repo_files, hf_hub_download as d; [d('TIGER-Lab/MMLU-Pro', f, repo_type='dataset', local_dir='data/mmlu_pro') for f in list_repo_files('TIGER-Lab/MMLU-Pro', repo_type='dataset') if f.endswith('.parquet')]"
```

## Reproducing the experiments

All runs use seed 0, bidirectional reading for Qwen (`--mode bidir`) and LoRA r = 16.

```bash
C="--steps 1000 --seed 0 --lora-compute autocast --grad-ckpt on --fold-eval 8 --resume --targets data/esm_targets"
L="--steps 6000 --eval-every 1000 --seed 0 --lora-compute autocast --grad-ckpt on --fold-eval 8 --resume --targets data/esm_targets_full"
```

### Training

| Poster | Model | Command |
|---|---|---|
| Result 1, 3 | Qwen3.5-0.8B | `python scripts/train_distill.py --run runs/pre_1k --mode bidir $C` |
| Result 1, 3 | Qwen3.5-2B | `python scripts/train_distill.py --run runs/qwen2b --qwen models/Qwen3.5-2B-Base --mode bidir $C` |
| Result 1, 3 | Qwen3.5-9B | `python scripts/train_distill.py --run runs/qwen9b --qwen models/Qwen3.5-9B-Base --mode bidir $C` |
| Result 1 | ESM2-35M / 150M / 650M (LoRA) | `python scripts/train_distill.py --run runs/esm35 --encoder esm2 --esm models/esm2_t12_35M_UR50D $C` (likewise `esm2_t30_150M_UR50D`, `esm2_t33_650M_UR50D`) |
| Result 2 | Qwen-0.8B, random init | `python scripts/train_distill.py --run runs/rand_1k --qwen-random-init 0 --mode bidir $C` |
| Result 2, 3 | Qwen-0.8B, 6,000 steps | `python scripts/train_distill.py --run runs/qwen_6k --mode bidir $L` |
| Result 2 | Qwen-0.8B random init, 6,000 steps | `python scripts/train_distill.py --run runs/rand_6k --qwen-random-init 0 --mode bidir $L` |
| Result 2, 3 | ESM2 only (frozen + adapter) | `python scripts/train_distill.py --run runs/frozen35 --encoder esm2 --esm models/esm2_t12_35M_UR50D --lora-r 0 $C` (likewise 650M) |
| Result 2 | ESM2 -> Qwen (Calpha-style injection) | `python scripts/train_distill.py --run runs/inj35 --esm-inject models/esm2_t12_35M_UR50D --mode bidir $C` (likewise 650M) |
| Follow-up 2 | Qwen3.5-9B, 6,000 steps | `python scripts/train_distill.py --run runs/qwen9b_6k --qwen models/Qwen3.5-9B-Base --mode bidir $L` |

### Prediction on CASP16

`jobs/casp_predict_set.sh <round> <method> <checkpoint|-> <name> <trunk-dtype:lm-quant:trunk-quant> ...` predicts all
targets with an experimental structure (up to 1,000 residues) and writes `outputs/casp16/<name>_<setting>/`.

```bash
bash jobs/casp_predict_set.sh 16 esmfold   -                   esmfold   fp32:none:none bf16:int8:int8
bash jobs/casp_predict_set.sh 16 trunkonly -                   trunkonly fp32:none:none
bash jobs/casp_predict_set.sh 16 qwenfold  runs/pre_1k/best.pt  pre_1k    fp32:none:none
bash jobs/casp_predict_set.sh 16 qwenfold  runs/frozen650/best.pt frozen650 fp32:none:none bf16:int8:int8   # int8 point of Result 3
bash jobs/casp_predict_set.sh 16 qwenfold  runs/esm35/best.pt   esm35     fp32:none:none bf16:none:none     # bf16 points of Result 3
# ... likewise for every trained run
```

### Scoring and statistics

```bash
python scripts/score.py --natives data/casp16/natives \
  --pred AF3server_g304=data/casp16/predictions/group304 \
  --pred ESMFold=outputs/casp16/esmfold_fp32_lmnone_trnone \
  --pred Pre_1k=outputs/casp16/pre_1k_fp32_lmnone_trnone \
  --out outputs/casp16/scores.csv          # add one --pred per method

python analysis/poster_stats.py --csv 16=outputs/casp16/scores.csv --out outputs/stats.csv
```

### Follow-up 1: linear probes

```bash
python scripts/probe_lm.py --test-round 16 --out outputs/probe16 \
  --models onehot esm35m esm150m esm650m esm3b qwen0.8b_rand qwen0.8b qwen2b qwen9b
```

Probes are trained on CASP14/15 targets and evaluated on CASP16 targets; the best layer of each model is selected
on held-out CASP14/15 targets.

### Follow-up 2: MMLU-Pro

```bash
python scripts/eval_text.py --qwen models/Qwen3.5-0.8B-Base --out outputs/text/base08.json     # before training
python scripts/eval_text.py --ckpt runs/pre_1k/best.pt        --out outputs/text/pre_1k.json     # with the trained LoRA
python scripts/eval_text.py --ckpt runs/qwen_6k/best.pt       --out outputs/text/qwen_6k.json
```

A fixed set of 350 questions (seed 0) is used.

## Notes

- The ESMFold structure module was trained on PDB entries up to May 2020; our training set uses the same cutoff,
  and all CASP16 targets were released later.
- All methods, including the AlphaFold3 server predictions, are scored with the same code (`scripts/score.py`).
  TM-score agrees with TM-align (`tmtools`) to three decimal places.
- Model weights, teacher features and predictions are not included in this repository.

## License

MIT License (see `LICENSE`). ESMFold, ESM2 and Qwen3.5 are distributed under their own licenses.
