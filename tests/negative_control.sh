#!/bin/bash
# Container: demonstrates the gate goes red. Per variant at TP2 (bf16, algo 1, 1K rows): the clean run must PASS,
# FP4AR_NEGCTL=stale (input of the next set) and =flip (one corrupted output bit on the last rank) must FAIL.
# Env: NEGCTL_VARIANTS (default: all from --list), NEGCTL_ARGS (extra fp4ar_bench args), BIN.
. "$(dirname "$0")/../scripts/env.sh"
BIN=${BIN:-$BUILD/latest/fp4ar_bench}; [ -x $BIN ] || { echo "no binary $BIN"; exit 2; }
[ "$NCCL_HOME" != /usr ] && export LD_LIBRARY_PATH=$NCCL_LIBDIR:$LD_LIBRARY_PATH
export HOME=$WORK/home; mkdir -p $HOME
VARS=${NEGCTL_VARIANTS:-$($BIN --list | awk '{print $1}' | paste -sd,)}
O=${NEGCTL_OUT:-$RUNS/negctl-$(date +%m%d-%H%M%S)}; mkdir -p $O; M=$O/manifest.tsv; : > $M
A="--tp 2 --input bf16 --rows 1024 --algo 1 --check --iters 0 --reps 200 --graph-reps 100 --skew-reps 20 ${NEGCTL_ARGS:-}"
for v in ${VARS//,/ }; do for nc in clean stale flip; do
  c=${v}_$nc
  if [ $nc = clean ]; then env -u FP4AR_NEGCTL $BIN --variant $v $A --tag negctl_$c --out $O/$c.jsonl > $O/$c.log 2>&1
  else FP4AR_NEGCTL=$nc $BIN --variant $v $A --tag negctl_$c --out $O/$c.jsonl > $O/$c.log 2>&1; fi
  rc=$?; printf '%s\t%s\t%s\t%s\n' $v $nc $rc $O/$c.jsonl >> $M; echo "$c rc=$rc"
done; done
python3 "$(dirname "$0")/gate_check.py" negctl $M
