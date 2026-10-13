#!/bin/bash
# Container: build + run the standalone fused-norm driver (NVFP4 AR + residual + RMSNorm) against kernels/fused_norm.
# Usage: run.sh <tag> "<nvcc -D flags or ->" "<drv args>" ...  (rebuilds only if missing; prefix args with NEG for LN_NEGCTL=1)
. "$(dirname "$0")/../../scripts/env.sh"
D=$(cd "$(dirname "$0")" && pwd); R=$D/../..; K=${LNORM_DIR:-$R/kernels/fused_norm}
T=$1; F=$2; shift 2; [ "$F" = - ] && F=""; B=$BUILD/LN-$T; mkdir -p $B $RUNS $WORK/home; export HOME=$WORK/home
[ "$NCCL_HOME" != /usr ] && export LD_LIBRARY_PATH=$NCCL_LIBDIR:$LD_LIBRARY_PATH
NV="$NVCC -O3 -std=c++17 -gencode arch=compute_$ARCH,code=sm_$ARCH -lineinfo -I$R/include -I$K -I$NCCL_HOME/include"
if [ ! -x $B/drv ]; then
  $NV $F -Xptxas -v -c -o $B/l.o $K/lnorm.cu > $B/build.log 2>&1 && $NV -c -o $B/drv.o $D/drv_norm.cu >> $B/build.log 2>&1 && \
    $NV -o $B/drv $B/drv.o $B/l.o -L$NCCL_LIBDIR -lnccl -lpthread >> $B/build.log 2>&1 || { echo build failed; tail -30 $B/build.log; exit 1; }
  grep -A1 "lnorm_kernel<2" $B/build.log | grep -o "Used [0-9]* registers\|[0-9]* bytes spill stores" | paste -sd' ' | head -c 600; echo
fi
nvidia-smi --query-gpu=index,clocks.mem,clocks.sm --format=csv,noheader
for a in "$@"; do
  case $a in NEG*) a=${a#NEG}; export LN_NEGCTL=1;; *) unset LN_NEGCTL;; esac
  echo "## $a negctl=${LN_NEGCTL:-0}"; (cd $B; timeout ${TMO:-240} ./drv $a --json $RUNS/LN-$T.jsonl 2>&1); echo "rc=$?"
done
