#!/bin/bash
# Container: one measurement cell. Usage: run_cell.sh <cell> [VAR=val ...] -- <fp4ar_bench args>
# JSON lines -> $RUNS/<cell>.jsonl, log -> $LOGS/<cell>.log. BIN overrides the binary; TMO the timeout (s).
. "$(dirname "$0")/env.sh"
CELL=$1; shift
while [ $# -gt 0 ] && [ "$1" != "--" ]; do export "$1"; shift; done; shift
mkdir -p $RUNS $LOGS $WORK/home; export HOME=$WORK/home
[ "$NCCL_HOME" != /usr ] && export LD_LIBRARY_PATH=$NCCL_LIBDIR:$LD_LIBRARY_PATH
# production-like NCCL env for the BF16 baseline (NCCL_PROTO is set per communicator by the harness)
if [ -n "$NCCL_DEFAULTS" ]; then unset NCCL_ALGO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
else export NCCL_ALGO=${NCCL_ALGO:-Ring} NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-64}; fi
BIN=${BIN:-$BUILD/latest/fp4ar_bench}
{
echo "# CELL $CELL $(date) host=$(hostname) step=${SLURM_JOB_ID}.${SLURM_STEP_ID} cvd=$CUDA_VISIBLE_DEVICES bin=$(readlink -f $BIN)"
env | grep -E '^NCCL_' | sort | tr '\n' ' '; echo
nvidia-smi --query-gpu=index,clocks.mem,clocks.sm,memory.used --format=csv,noheader | tr '\n' ';'; echo
if [ -n "$MEMCLK_LOCK" ] && [ -z "$NO_CLOCK_GUARD" ]; then
  BAD=$(nvidia-smi --query-gpu=clocks.mem --format=csv,noheader,nounits | grep -vc "^$MEMCLK_LOCK$")
  [ "$BAD" != 0 ] && { echo "CLOCK_ABORT: clocks.mem != $MEMCLK_LOCK on $BAD GPU(s)"; exit 3; }
fi
timeout ${TMO:-840} $BIN --out $RUNS/$CELL.jsonl --tag $CELL "$@"; rc=$?
echo "# CELL $CELL rc=$rc $(date)"
nvidia-smi --query-gpu=index,clocks.mem,memory.used --format=csv,noheader | tr '\n' ';'; echo
exit $rc
} 2>&1 | tee -a $LOGS/$CELL.log
exit ${PIPESTATUS[0]}
