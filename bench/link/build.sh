#!/bin/bash
# Container: builds the CE-vs-SM peer bandwidth probe into $WORK/build/link/ce.
. "$(dirname "$0")/../../scripts/env.sh"; mkdir -p $BUILD/link
$NVCC -O3 -gencode arch=compute_$ARCH,code=sm_$ARCH -o $BUILD/link/ce "$(dirname "$0")/ce.cu" && echo "built $BUILD/link/ce"
