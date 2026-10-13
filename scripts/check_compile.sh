#!/bin/bash
# Offline compile check (x86, no GPU): pip nvcc 13.4 for sm_107a + NCCL 2.31 headers (setup_toolchain.sh).
# Usage: check_compile.sh [--sass] [--sm100] <file.cu> [extra nvcc flags, e.g. -DSTAGES=4]
#        check_compile.sh --remote <file relative to repo> [flags]  -> sync, then compile_one.sh in the image of $CLUSTER (NODE=, GPU=)
set -o pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
TC=${FP4AR_TOOLCHAIN:-${XDG_CACHE_HOME:-$HOME/.cache}/fp4ar-toolchain}
if [ "$1" = "--remote" ]; then
  shift; . $R/scripts/env.sh; bash $R/scripts/sync.sh >/dev/null || exit 2
  ssh -o BatchMode=yes $LOGIN "bash $SRC/scripts/gpurun.sh compile ${NODE:?set NODE} ${GPU:-0} 300 $SRC/scripts/compile_one.sh $*"; exit $?
fi
CU=$(ls -d $TC/venv/lib/python3*/site-packages/nvidia/cu13 2>/dev/null | head -1)
[ -x "$CU/bin/nvcc" ] || { echo "toolchain missing at $TC; run scripts/setup_toolchain.sh"; exit 2; }
[ -f "$CU/include/cuda_profiler_api.h" ] || echo "note: cuda_profiler_api.h missing (harness needs it); rerun setup_toolchain.sh, which installs nvidia-cuda-profiler-api"
SASS=0; ARCH="-gencode arch=compute_107a,code=sm_107a"
while [ "${1:0:2}" = "--" ]; do
  case $1 in --sass) SASS=1 ;; --sm100) ARCH="-gencode arch=compute_100a,code=sm_100a" ;; *) echo "unknown $1"; exit 2 ;; esac; shift
done
F=$1; shift
[ -f "$F" ] || { echo "usage: check_compile.sh [--sass] [--sm100] <file.cu> [nvcc flags]"; exit 2; }
X=""; case "$F" in *.cuh|*.h) X="-x cu" ;; esac
O=$(mktemp -d); trap 'rm -rf $O' EXIT
export PATH=$CU/bin:$PATH
$CU/bin/nvcc -O3 -std=c++17 $ARCH -lineinfo -Xptxas -v -I$R/include -I$R/harness -I$TC/nccl/include \
  $X -c "$F" -o $O/out.o "$@" 2> $O/log; rc=$?
grep -v -E "^ptxas info|bytes stack frame|^$" $O/log
echo "--- per-kernel: regs | stack/spill"
awk '/Compiling entry function/ {match($0, /'"'"'[^'"'"']+'"'"'/); k=substr($0, RSTART+1, RLENGTH-2)}
     /bytes stack frame/ {sp=$0; gsub(/^ +/, "", sp)}
     /Used [0-9]+ registers/ {match($0, /Used [0-9]+ registers/); print substr($0, RSTART+5, RLENGTH-15) " regs | " sp " | " k}' $O/log | \
  (command -v c++filt >/dev/null && c++filt || cat) | cut -c1-220
grep -q -E "[1-9][0-9]* bytes spill (stores|loads)" $O/log && echo "WARNING: register spills present"
if [ $rc -eq 0 ] && [ $SASS = 1 ]; then
  echo "--- SASS opcode histogram (top 40):"
  $CU/bin/cuobjdump -sass $O/out.o | grep -E '^\s+/\*[0-9a-f]{4}\*/' | sed -E 's#^\s+/\*[0-9a-f]+\*/\s+##; s#^@!?U?P[T0-9]+\s+##' | awk '{print $1}' | sort | uniq -c | sort -rn | head -40
fi
echo "check_compile: $F rc=$rc"; exit $rc
