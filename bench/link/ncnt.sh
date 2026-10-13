#!/bin/bash
# Container: NVLink counter brackets around steady nccl-tests runs at one size. Usage: ncnt.sh <bytes> <iters>  (parse: ncnt.py)
. "$(dirname "$0")/common.sh"; SZ=${1:-1G}; N=${2:-2000}
snap start
r() { local n=$1 envs=$2; shift 2; cnt "pre $n"; ( for e in $envs; do export $e; done; $NT/$@ -g $NG -b $SZ -e $SZ -n $N -w 20 -c 0 | grep -E '^ +[0-9]+ |Avg' ); cnt "post $n"; }
r ar_ring_simple64 "NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=64 NCCL_MAX_NCHANNELS=64" all_reduce_perf -d bfloat16
r ar_ring_ll128_64 "NCCL_ALGO=Ring NCCL_PROTO=LL128 NCCL_MIN_NCHANNELS=64 NCCL_MAX_NCHANNELS=64" all_reduce_perf -d bfloat16
r ar_default "" all_reduce_perf -d bfloat16
r sendrecv_simple64 "NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=64 NCCL_MAX_NCHANNELS=64" sendrecv_perf -d bfloat16
r alltoall_default "" alltoall_perf -d bfloat16
r ag_default_R2 "" all_gather_perf -d bfloat16 -R 2
snap end
