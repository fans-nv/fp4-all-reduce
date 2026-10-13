#!/bin/bash
# Container (1 GPU): A/B builds of the codec bench, built in parallel, run back to back. Usage: ab.sh <what> <rows> <tag:flags>...
. "$(dirname "$0")/../../scripts/env.sh"
D=$(cd "$(dirname "$0")" && pwd); R=$D/../..; what=$1 rows=$2; shift 2
for spec in "$@"; do
  tag=${spec%%:*}; flags=${spec#*:}; O=$BUILD/codec-ab-$tag; mkdir -p $O
  ( $NVCC -O3 -std=c++17 -gencode arch=compute_$ARCH,code=sm_$ARCH -lineinfo -diag-suppress 177 -Xcompiler -fopenmp -lgomp \
      -I$R/include -I$R/harness $flags -o $O/codec_bench $D/codec_bench.cu > $O/build.log 2>&1 ) &
done; wait
for spec in "$@"; do tag=${spec%%:*}; echo "=== $tag ${spec#*:}"; grep -i error $BUILD/codec-ab-$tag/build.log; $BUILD/codec-ab-$tag/codec_bench $what $rows; done
