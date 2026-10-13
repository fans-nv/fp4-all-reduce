#!/bin/bash
# vLLM's own communicator benchmarks (image /vllm-workspace/benchmarks/kernels), M3 hidden 6144.
# Args: <n> <rep> <part: comm|fused|all> [comm env tag]
. "$(dirname "$0")/../../scripts/env.sh"
V=${VLLM_BENCH_DIR:-/vllm-workspace/benchmarks/kernels}; n=$1; rep=$2; part=${3:-all}; et=${4:-default}
O=$RUNS/baselines/vllm/tp$n/$rep; mkdir -p $O $BUILD/vllm_bench; cd $O
# hidden size is a module constant in benchmark_device_communicators.py; copy with it set to 6144 (only change)
sed 's/^HIDDEN_SIZE = 8192$/HIDDEN_SIZE = 6144/' $V/benchmark_device_communicators.py > $BUILD/vllm_bench/bdc_h6144.py
grep -n '^HIDDEN_SIZE' $BUILD/vllm_bench/bdc_h6144.py
case $et in
  default) EV="";;
  tunedS) EV="NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=64 NCCL_MAX_NCHANNELS=64";;
  tunedLL128) EV="NCCL_ALGO=Ring NCCL_PROTO=LL128 NCCL_MIN_NCHANNELS=64 NCCL_MAX_NCHANNELS=64";;
esac
if [ $part = comm ] || [ $part = all ]; then
  env $EV NCCL_DEBUG=VERSION timeout 600 torchrun --nproc_per_node=$n --master_port=$((29500+RANDOM%1000)) $BUILD/vllm_bench/bdc_h6144.py \
    --sequence-lengths 1024 2048 4096 8192 16384 --num-warmup 20 --num-trials 200 --output-json $O/comm_$et.json > $O/comm_$et.log 2>&1
  echo "comm rc=$?"; grep -v 'INFO\|WARNING' $O/comm_$et.log | tail -25
fi
if [ $part = fused ] || [ $part = all ]; then
  timeout 700 torchrun --nproc_per_node=$n --master_port=$((29500+RANDOM%1000)) $V/benchmark_fused_collective.py \
    --num-tokens 1024 2048 4096 --hidden-dim 6144 --quant-modes none,fp4 --warmup 20 --trials 500 --output-file $O/fused.md > $O/fused.log 2>&1
  echo "fused rc=$?"; tail -5 $O/fused.log; cat $O/fused.md 2>/dev/null | head -60
fi
