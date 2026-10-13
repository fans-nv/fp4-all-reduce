#!/bin/bash
# Container: link facts for the GPUs/fabric of this node (no load).
echo "# FACTS2 $(hostname) $(TZ=America/Los_Angeles date) cvd=$CUDA_VISIBLE_DEVICES"
x() { echo; echo "### $*"; timeout 60 "$@" 2>&1; }
x nvidia-smi nvlink -dr -i 0
x nvidia-smi nvlink -info -i 0
x nvidia-smi nvlink -gBwMode
x nvidia-smi nvlink -gLWidth -i 0
x nvidia-smi nvlink -gLowPwrInfo -i 0
x nvidia-smi nvlink -R -i 0
echo; echo "### fabric per GPU"; for i in 0 1 2 3; do echo "GPU $i"; nvidia-smi -q -i $i | grep -A20 '^    Fabric' | grep -E 'State|Status|Clique|ClusterUUID|Bandwidth|Pointer|LE'; nvidia-smi -q -i $i | grep -E 'Fabric GUID|Serial|Module Id|Platform|Chassis|Slot|Tray|Host ID|Peer Type' ; done
x nvidia-smi -q -i 0 -d PLATFORM
x nvidia-smi --query-gpu=index,clocks.mem,clocks.sm,utilization.gpu,memory.used --format=csv
