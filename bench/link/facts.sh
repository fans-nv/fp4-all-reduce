#!/bin/bash
# Link facts (no GPU load): nvlink status/caps/counters, fabric, topo, tool discovery. Runs inside the container.
echo "# FACTS $(hostname) $(TZ=America/Los_Angeles date) cvd=$CUDA_VISIBLE_DEVICES"
x() { echo; echo "### $*"; timeout 60 "$@" 2>&1; }
x nvidia-smi
x nvidia-smi topo -m
x nvidia-smi nvlink -h
x nvidia-smi nvlink -s
x nvidia-smi nvlink --status -i 0
x nvidia-smi nvlink -c -i 0
x nvidia-smi nvlink -gt d -i 0
x nvidia-smi nvlink -gt r -i 0
x nvidia-smi nvlink -e -i 0
x nvidia-smi -q -i 0
x nvidia-smi -q -d FABRIC
x nvidia-smi --query-gpu=index,pci.bus_id,clocks.mem,clocks.sm,clocks.max.sm,power.draw,memory.used,utilization.gpu --format=csv
x nvidia-smi dmon -h
echo; echo "### tools"
for t in all_reduce_perf sendrecv_perf all_gather_perf reduce_scatter_perf alltoall_perf dcgmi nsys ncu nv-fabricmanager nvlsm; do echo "$t: $(command -v $t)"; done
ls -d /usr/local/bin/*perf* /opt/*nccl* /workspace/nccl* /usr/local/nccl-tests* 2>/dev/null
find / -xdev -name 'all_reduce_perf*' -type f 2>/dev/null | head
ls /usr/lib/*/libnccl* /usr/lib/libnccl* 2>/dev/null
P=$(command -v all_reduce_perf || find / -xdev -name all_reduce_perf -type f 2>/dev/null | head -1); [ -n "$P" ] && x $P -h
nvcc --version 2>/dev/null | tail -2; ls /usr/local/ | grep -i cuda
