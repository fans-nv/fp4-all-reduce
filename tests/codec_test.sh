#!/bin/bash
# Container (one GPU): builds and runs codec_test.cu, the codec vs an independent CPU E2M1/E4M3 oracle.
. "$(dirname "$0")/../scripts/env.sh"; B=$BUILD/codec_test; mkdir -p $B
$NVCC -O3 -std=c++17 -gencode arch=compute_$ARCH,code=sm_$ARCH -I$SRC/include -o $B/codec_test $SRC/tests/codec_test.cu "$@" && $B/codec_test
