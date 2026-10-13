#!/bin/bash
# Container: nccl-tests driver (NT = nccl-tests bin dir, default /usr/local/bin). Usage: nc.sh <cellsfile> <outdir> [first_line] [last_line]
# cell line: name | VAR=val ... | <binary> <args>   (one process, -g = #GPUs in CVD unless args give -g)
C=$1 O=$2 F=${3:-1} T=${4:-100000}; mkdir -p $O
NG=$(echo $CUDA_VISIBLE_DEVICES | tr , '\n' | wc -l)
snap() { echo "# SNAP $1 $(date +%T) $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,clocks.mem,clocks.sm --format=csv,noheader | tr '\n' ';')"; }
snap start
sed -n "${F},${T}p" $C | while IFS='|' read -r name envs cmd; do
  name=$(echo $name); cmd=$(echo $cmd); [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  echo "=== CELL $name | $envs | $cmd | $(date +%T)"
  ( unset NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_NTHREADS
    for e in $envs; do export "$e"; done
    timeout 240 ${NT:-/usr/local/bin}/$cmd -g $NG > $O/$name.txt 2>&1; echo "rc=$?" >> $O/$name.txt )
  grep -E '^ +[0-9]+ ' $O/$name.txt | awk '{print "  ", $1, $(NF-5), $(NF-4)}' | tail -4
  grep -E 'Avg bus bandwidth|rc=' $O/$name.txt
done
snap end
