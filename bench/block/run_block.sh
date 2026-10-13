#!/bin/bash
# Container (2 GPUs): vLLM/FlashInfer TP2 AR + residual + RMSNorm (+NVFP4) block timing.
# Usage: run_block.sh <tag> <NCCL proto> <sets e.g. norm,nccl,fi> [rows]; REG=1 = registered NCCL buffers; FIB = FlashInfer backend.
. "$(dirname "$0")/../../scripts/env.sh"
D=$(cd "$(dirname "$0")" && pwd); mkdir -p $RUNS/block $WORK/home; export HOME=$WORK/home
export FIB=${FIB:-trtllm}
export NCCL_ALGO=Ring NCCL_MIN_NCHANNELS=64 NCCL_MAX_NCHANNELS=64 NCCL_PROTO=$2 SETS=$3 TAG=$1 OUT=$RUNS/block/$1.jsonl
[ -n "$4" ] && export ROWS=$4
[ -n "$REG" ] && export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_CUMEM_ENABLE=1
nvidia-smi --query-gpu=index,clocks.mem,clocks.sm --format=csv,noheader | tr '\n' ';'; echo
if [ -n "$MEMCLK_LOCK" ]; then
  BAD=$(nvidia-smi --query-gpu=clocks.mem --format=csv,noheader,nounits | grep -vc "^$MEMCLK_LOCK$"); [ "$BAD" != 0 ] && { echo CLOCK_ABORT; exit 3; }
fi
cd $D && timeout ${TMO:-540} torchrun --nproc-per-node 2 --master-port $((29000 + RANDOM % 1000)) $D/block_bench.py 2>&1 | grep -v "Warning\|deprecated"
