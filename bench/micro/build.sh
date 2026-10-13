#!/bin/bash
# Container: builds the micro programs into $WORK/build/micro (or OUT). Env: PROGS (default all five).
set -e
. "$(dirname "$0")/../../scripts/env.sh"
D=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$BUILD/micro}; mkdir -p $OUT
[ -e $NCCL_LIBDIR/libnccl.so ] || ln -sf libnccl.so.2 $NCCL_LIBDIR/libnccl.so 2>/dev/null || true
for p in ${PROGS:-bw lat fence tear validity}; do
  $NVCC -O3 -std=c++17 -gencode arch=compute_$ARCH,code=sm_$ARCH -lineinfo -diag-suppress 821 -I$D -I$NCCL_HOME/include \
    -o $OUT/$p $D/$p.cu -L$NCCL_LIBDIR -lnccl -lpthread
  echo "built $OUT/$p"
done
