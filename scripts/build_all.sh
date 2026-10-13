#!/bin/bash
# Container: the top-level build.sh with the cluster profile's compiler/NCCL -> $BUILD/<tag>/{fp4ar_bench,fp4ar_norm_bench} (+ latest).
# Usage: build_all.sh [tag]
. "$(dirname "$0")/env.sh"
R=$(cd "$(dirname "$0")/.." && pwd)
[ -e $NCCL_LIBDIR/libnccl.so ] || ln -sf libnccl.so.2 $NCCL_LIBDIR/libnccl.so 2>/dev/null
ARCH=$ARCH NVCC=$NVCC NCCL_HOME=$NCCL_HOME NCCL_LIB=$NCCL_LIBDIR BUILD_DIR=$BUILD exec bash $R/build.sh all "$@"
