#!/bin/bash
# Predict all targets of one CASP round that have a native structure and a sequence, under several settings,
# sequentially in one GPU job (one start-up cost).  Generalises jobs/casp14_predict_set.sh to CASP15/16.
#   bash jobs/casp_predict_set.sh <14|15|16> <esmfold|qwenfold|trunkonly> <ckpt|-> <out-prefix> <trunk:lm:trunkq[:chunk]> [...]
# Targets: native file names T1234[s1|v1][-D1].pdb -> T1234 / T1234s1; variants (T1214v1) map to the FASTA entry
# T1214 when the FASTA has no T1214v1.  Outputs: outputs/casp<N>/<prefix>_<setting>/.
set -u
casp=$1 method=$2 ckpt=$3 prefix=$4; shift 4
D=data/casp$casp
fasta_ids=$(grep '^>' $D/sequences.fasta | sed -E 's/^>([^ ]+).*/\1/' | sort -u)
T=""
for f in $D/natives/*.pdb; do
  t=$(basename "$f" .pdb); t=${t%%-D*}
  grep -qx "$t" <<< "$fasta_ids" || t=$(sed -E 's/v[0-9]+$//' <<< "$t")
  grep -qx "$t" <<< "$fasta_ids" && T="$T $t"
done
T=$(tr ' ' '\n' <<< "$T" | sort -u | tr '\n' ' ')
echo "### CASP$casp targets ($(wc -w <<< "$T")): $T"
for cfg in "$@"; do
  IFS=: read -r td ql qt ch <<< "$cfg"; ch=${ch:-64}
  name=${prefix}_${td}_lm${ql}_tr${qt}; [ "$ch" != 64 ] && name=${name}_ch$ch
  echo "### $name $(date '+%T')"
  extra=(); [ "$ckpt" != - ] && extra=(--ckpt "$ckpt")
  .venv/bin/python scripts/predict.py --method "$method" "${extra[@]}" --fasta $D/sequences.fasta \
    --outdir outputs/casp$casp/$name --targets $T --target-regex '^T\d{4}([sv]\d+)?$' \
    --trunk-dtype "$td" --quant-lm "$ql" --quant-trunk "$qt" --chunk-size "$ch"
done
echo "### done $(date '+%T')"
