#!/bin/bash
# Container: runs one micro program in a step. Usage: step.sh <outname> <prog> [args]; JSON -> $RUNS/micro/<outname>.jsonl
. "$(dirname "$0")/../../scripts/env.sh"
[ "$NCCL_HOME" != /usr ] && export LD_LIBRARY_PATH=$NCCL_LIBDIR:$LD_LIBRARY_PATH
mkdir -p $RUNS/micro; out=$1 prog=$2; shift 2
echo "# $(date +%T) $(hostname) CVD=$CUDA_VISIBLE_DEVICES $prog $*" >&2
$BUILD/micro/$prog "$@" | tee $RUNS/micro/$out.jsonl
echo "# rc=${PIPESTATUS[0]} $(date +%T)" >&2
