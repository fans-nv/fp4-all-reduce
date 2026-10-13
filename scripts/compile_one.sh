#!/bin/bash
# Container: single-file compile with ptxas -v. Usage: compile_one.sh <file relative to repo> [nvcc flags]
. "$(dirname "$0")/env.sh"
F=$1; shift; O=$(mktemp -d)
$NVCC -O3 -std=c++17 -gencode arch=compute_$ARCH,code=sm_$ARCH -lineinfo -Xptxas -v -I$SRC/include -I$SRC/harness -I$NCCL_HOME/include -c $SRC/$F -o $O/out.o "$@" 2>&1 | \
  grep -v -E "^ptxas info +: (Compiling|Function properties)|bytes stack frame, 0 bytes spill" ; rc=${PIPESTATUS[0]}
rm -rf $O; echo "compile_one: $F arch=$ARCH rc=$rc"; exit $rc
