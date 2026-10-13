#!/bin/bash
# Container: full correctness gate for every linked variant at every tier; exit 0 only if all applicable cells PASS.
# Tiers: TP {2,4} x input {bf16,nvfp4,nvfp4-window} x algo {1,2} x rows 1K-16K; each cell = bitwise vs the FP32
# rank-order reference, ranks identical, 1000 b2b + 1000 per-call + 10x1000 graph replays + 200 skewed calls.
# Env: GATE_VARIANTS (default: all from --list), GATE_TPS, GATE_INPUTS, GATE_ROWS, BIN, GATE_OUT.
# Also runs tests/gate_fused_norm.sh (GATE_SKIP_FUSED_NORM=1 to skip). Run tests/negative_control.sh too: a gate that cannot go red proves nothing.
. "$(dirname "$0")/../scripts/env.sh"
BIN=${BIN:-$BUILD/latest/fp4ar_bench}; [ -x $BIN ] || { echo "no binary $BIN"; exit 2; }
[ "$NCCL_HOME" != /usr ] && export LD_LIBRARY_PATH=$NCCL_LIBDIR:$LD_LIBRARY_PATH
export HOME=$WORK/home; mkdir -p $HOME
NG=$(echo ${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES} | tr , '\n' | wc -l)
VARS=${GATE_VARIANTS:-$($BIN --list | awk '{print $1}' | paste -sd,)}
TPS=${GATE_TPS:-2 4}; INPUTS=${GATE_INPUTS:-bf16 nvfp4 nvfp4-window}; ROWS=${GATE_ROWS:-1024,2048,4096,8192,16384}
O=${GATE_OUT:-$RUNS/gate-$(date +%m%d-%H%M%S)}; mkdir -p $O; M=$O/manifest.tsv; : > $M
echo "# gate bin=$(readlink -f $BIN) variants=$VARS tps=$TPS inputs=$INPUTS rows=$ROWS -> $O"
for v in ${VARS//,/ }; do for tp in $TPS; do
  [ $tp -gt $NG ] && { echo "# tp=$tp needs $tp GPUs, CVD has $NG"; exit 2; }
  for in in $INPUTS; do
    c=${v}_tp${tp}_$in
    timeout ${GATE_TMO:-1800} $BIN --tp $tp --variant $v --input $in --rows $ROWS --algo 1,2 --check --iters 0 \
      --tag gate_$c --out $O/$c.jsonl > $O/$c.log 2>&1; rc=$?
    printf '%s\t%s\t%s\t%s\t%s\n' $v $tp $in $rc $O/$c.jsonl >> $M
    echo "$c rc=$rc $(grep -c '# CHECK PASS' $O/$c.log) pass $(grep -c '# CHECK FAIL' $O/$c.log) fail $(grep -o '# skip.*' $O/$c.log | head -1)"
  done
done; done
python3 "$(dirname "$0")/gate_check.py" gate $M
rc=$?
if [ -z "$GATE_SKIP_FUSED_NORM" ]; then bash "$(dirname "$0")/gate_fused_norm.sh" || rc=1; fi
exit $rc
