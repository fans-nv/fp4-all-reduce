#!/bin/bash
# Container: builds nccl-tests (source at NCCL_TESTS_SRC, a git checkout) against $NCCL_HOME into $BUILD/nccl-tests.
. "$(dirname "$0")/../../scripts/env.sh"
S=${NCCL_TESTS_SRC:-$WORK/nccl-tests}; [ -d $S ] || { echo "clone https://github.com/NVIDIA/nccl-tests to $S first"; exit 2; }
cd $S && make -j32 MPI=0 CUDA_HOME=$(dirname $(dirname $NVCC)) NCCL_HOME=$NCCL_HOME NVCC_GENCODE="-gencode=arch=compute_$ARCH,code=sm_$ARCH" \
  BUILDDIR=$BUILD/nccl-tests src.build > $BUILD/nccl-tests.build.log 2>&1; rc=$?
tail -3 $BUILD/nccl-tests.build.log; ls $BUILD/nccl-tests; exit $rc
