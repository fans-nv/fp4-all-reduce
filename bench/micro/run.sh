#!/bin/bash
# Login node: micro campaign, one program per step via scripts/step.sh. Usage: NODE=<node> run.sh [step-name ...]
# No args = all steps in order (~20 min). Build first: scripts/step.sh <node> 0 600 $SRC/bench/micro/build.sh
set -u
. "$(dirname "$0")/../../scripts/env.sh"
N=${NODE:?set NODE}; G2=${G2:-0,1}; G4=${G4:-0,1,2,3}; NSM=${NSM:-16,32,64,106,0}; T=${TAGP:-$CLUSTER}
declare -A S
# (1) bandwidth: push vs TMA-store vs pull, chunk size, comm-SM count
S[bw_tp2]="$G2 bw --n 2 --method push_st,tma_st,tma_ld,tma_ld_cluster,pull_ld --pattern uni,a2a,self --chunk 1K,2K,4K,8K,16K,32K --bytes 1M,4M,16M,64M --cpsm 1,2 --nsm $NSM --iters 30 --tag $T"
S[bw_tp4]="$G4 bw --n 4 --method push_st,tma_st,tma_ld,pull_ld,mc_st --pattern uni,fanout,a2a --chunk 1K,4K,16K,32K --bytes 1M,4M,16M,64M --cpsm 1,2 --nsm $NSM --iters 30 --tag $T"
S[bw_tp2_inner]="$G2 bw --n 2 --method push_st,tma_st,tma_ld,pull_ld --pattern a2a --chunk 2K,4K,8K,16K --bytes 512K,1M,2M,4M --cpsm 1 --nsm $NSM --inner 16 --iters 30 --tag ${T}_inner16"
S[bw_tp4_inner]="$G4 bw --n 4 --method push_st,tma_st,tma_ld,pull_ld,mc_st --pattern a2a --chunk 2K,4K,8K,16K --bytes 512K,1M,2M,4M --cpsm 1 --nsm $NSM --inner 16 --iters 30 --tag ${T}_inner16"
# (2) latency: Lamport vs flag vs pull round trip at small sizes
S[lat_tp2]="$G2 lat --n 2 --iters 2000 --reps 2 --tag $T"
S[lat_tp4]="$G4 lat --n 4 --peer 3 --iters 2000 --reps 1 --tag $T"
# (4) fence drain cost and per-tile publication throughput
S[fence_tp2]="$G2 fence --n 2 --tag $T"
S[fence_tp4]="$G4 fence --n 4 --lwarps 8 --tag $T"
# (2) 16 B / 8 B tear over NVLink (scalar4/split are positive controls: must tear)
S[tear_tp2]="$G2 tear --n 2 --ms 2000 --bytes 64K,1M --tag $T"
S[tear_tp4]="$G4 tear --n 4 --ms 1000 --bytes 64K --tag $T"
# (3) TMA validity (sm_107a only); layout v0 is separate because it may fault
S[validity_sem]="$G2 validity --n 2 --part sem --layout 1 --tag $T"
S[validity_xgpu]="$G2 validity --n 2 --part xgpu --layout 1 --iters 20000 --tag $T"
S[validity_sem_v0]="$G2 validity --n 2 --part sem --layout 0 --variant 3 --tag ${T}_v0"
ORDER="bw_tp2 bw_tp4 lat_tp2 fence_tp2 tear_tp2 validity_sem validity_xgpu lat_tp4 fence_tp4 tear_tp4 bw_tp2_inner bw_tp4_inner validity_sem_v0"
for s in ${@:-$ORDER}; do
  set -- ${S[$s]}; g=$1; shift
  LANE=micro bash "$(dirname "$0")/../../scripts/step.sh" $N $g ${STEP_TMO:-600} $SRC/bench/micro/step.sh $s "$@"
done
