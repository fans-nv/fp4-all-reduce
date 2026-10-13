#!/bin/bash
# Container, CVD=0,1,2,3: NVLS vs Ring per-op NVLink bytes at TP4 (busbw-formula check); nccl-tests runs oop+ip, so ops = 2*N.
. "$(dirname "$0")/common.sh"; N=500
snap start
r() { local n=$1 envs=$2; shift 2; cnt "pre $n"; ( for e in $envs; do export $e; done; $NT/$@ -g $NG -b 1G -e 1G -n $N -w 0 -c 0 | grep -E '^ +[0-9]+ |Avg' ); cnt "post $n"; }
r nvls_ar "NCCL_ALGO=NVLS" all_reduce_perf -d bfloat16
r nvls_ag "NCCL_ALGO=NVLS" all_gather_perf -d bfloat16
r nvls_rs "NCCL_ALGO=NVLS" reduce_scatter_perf -d bfloat16
r ring_ag "NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=64 NCCL_MAX_NCHANNELS=64" all_gather_perf -d bfloat16
snap end
