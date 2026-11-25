#!/bin/bash
# Predict the 33 CASP14 targets under several precision settings, sequentially in one GPU job (one start-up cost).
#   bash jobs/casp14_predict_set.sh <esmfold|qwenfold> <ckpt|-> <out-prefix> <trunk:lm:trunkq[:chunk]> [...]
#   e.g. bash jobs/casp14_predict_set.sh qwenfold runs/esm650/best.pt esm650 fp32:none:none bf16:int8:none
set -u
method=$1 ckpt=$2 prefix=$3; shift 3
T="T1024 T1025 T1026 T1027 T1029 T1030 T1031 T1032 T1033 T1035 T1036 T1037 T1038 T1039 T1040 T1041 T1042 T1043 T1044 T1046 T1049 T1050 T1054 T1056 T1064 T1067 T1073 T1074 T1079 T1080 T1082 T1090 T1099"
for cfg in "$@"; do
  IFS=: read -r td ql qt ch <<< "$cfg"; ch=${ch:-64}
  name=${prefix}_${td}_lm${ql}_tr${qt}; [ "$ch" != 64 ] && name=${name}_ch$ch
  echo "### $name $(date '+%T')"
  extra=(); [ "$ckpt" != - ] && extra=(--ckpt "$ckpt")
  .venv/bin/python scripts/predict.py --method "$method" "${extra[@]}" --fasta data/casp14/sequences.fasta \
    --outdir outputs/casp14/$name --targets $T --trunk-dtype "$td" --quant-lm "$ql" --quant-trunk "$qt" --chunk-size "$ch"
done
echo "### done $(date '+%T')"
