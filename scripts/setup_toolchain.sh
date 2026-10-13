#!/bin/bash
# Recreates the offline x86 toolchain for check_compile.sh: pip nvcc 13.4 (knows sm_107a) + NCCL headers.
# Usage: [FP4AR_TOOLCHAIN=<dir>] [NCCL_INCLUDE=<local dir with nccl.h, nccl_device/>] setup_toolchain.sh
# Without NCCL_INCLUDE the headers are unsquashed from $IMG on $LOGIN (cluster.env) into $WORK/toolchain (read of the image only).
set -e
TC=${FP4AR_TOOLCHAIN:-${XDG_CACHE_HOME:-$HOME/.cache}/fp4ar-toolchain}
mkdir -p $TC && cd $TC
[ -d venv ] || uv venv -q venv --python 3.10
VIRTUAL_ENV=$TC/venv uv pip install -q "nvidia-cuda-nvcc==13.4.92" "nvidia-cuda-runtime==13.4.*" "nvidia-cuda-crt==13.4.*" "nvidia-nvvm==13.4.*" nvidia-cuda-cccl "nvidia-cuda-cuobjdump==13.4.*" nvidia-cuda-nvdisasm "nvidia-cuda-profiler-api==13.4.*"
mkdir -p nccl/include
if [ -n "$NCCL_INCLUDE" ]; then cp -r $NCCL_INCLUDE/. nccl/include/
else
  . "$(dirname "$0")/env.sh"; W=$WORK/toolchain
  ssh -o BatchMode=yes $LOGIN "[ -f $W/nccl-hdr/usr/include/nccl.h ] || unsquashfs -q -d $W/nccl-hdr $IMG usr/include/nccl.h usr/include/nccl_device >/dev/null"
  ssh -o BatchMode=yes $LOGIN "cd $W/nccl-hdr/usr && tar czf - include" | tar xzf - -C nccl
fi
echo "toolchain ready at $TC"
