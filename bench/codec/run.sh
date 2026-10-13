#!/bin/bash
# Container (1 GPU): build + run the codec bench. Usage: run.sh [test|bench|all] [rows] [nvcc -D flags...]
set -e
. "$(dirname "$0")/../../scripts/env.sh"
D=$(cd "$(dirname "$0")" && pwd); R=$D/../..
OUT=${OUT:-$BUILD/codec-$ARCH}; mkdir -p $OUT
what=${1:-all}; rows=${2:-1024,2048,4096,8192,16384}; shift 2 || true
$NVCC -O3 -std=c++17 -gencode arch=compute_$ARCH,code=sm_$ARCH -lineinfo -diag-suppress 177 -Xcompiler -fopenmp -lgomp \
  -I$R/include -I$R/harness -o $OUT/codec_bench $D/codec_bench.cu "$@"
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader
$OUT/codec_bench $what $rows
