#!/bin/bash
# nccl-tests sweep. Args: <n> <coll: ar|ag|a2a> <rep tag> [algos] [protos] [chans]
# ar: BF16 sum, 12..192 MiB (1K..16K rows x 6144). ag/a2a: uint8 NVFP4 wire, W=rows*3460 B per rank.
. "$(dirname "$0")/../../scripts/env.sh"
[ "$NCCL_HOME" != /usr ] && export LD_LIBRARY_PATH=$NCCL_LIBDIR:$LD_LIBRARY_PATH
T=${NT:-$BUILD/nccl-tests}; n=$1; coll=$2; rep=$3
ALGOS=${4:-"Ring Tree NVLS"}; PROTOS=${5:-"LL LL128 Simple"}; CHS=${6:-"16 32 64"}
O=$RUNS/baselines/nccltests/tp$n/$rep; mkdir -p $O
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu --format=csv,noheader > $O/nvsmi_$coll.txt
case $coll in
  ar)   cmds=("all_reduce_perf -d bfloat16 -o sum -b 12582912 -e 201326592 -f 2");;
  ag)   cmds=("all_gather_perf -d uint8 -b $((3543040*n)) -e $((56688640*n)) -f 2" "all_gather_perf -d uint8 -b 3543040 -e 56688640 -f 2");;
  a2a)  cmds=("alltoall_perf -d uint8 -b 3543040 -e 56688640 -f 2");;
esac
run() { # tag env...
  local tag=$1; shift
  for ci in "${!cmds[@]}"; do
    local c=${cmds[$ci]}; local f=$O/${coll}${ci}_$tag.log
    echo "# env: $* cmd: $c" > $f
    env "$@" timeout 120 $T/$c -g $n -w 20 -n 100 -c 0 -a 3 >> $f 2>&1 || echo "# rc=$?" >> $f
  done
}
run default NCCL_DEBUG=VERSION
for a in $ALGOS; do
  run ${a}_auto NCCL_ALGO=$a
  for p in $PROTOS; do for c in $CHS; do
    if [ $a = NVLS ]; then run ${a}_${p}_$c NCCL_ALGO=$a NCCL_PROTO=$p NCCL_MIN_NCHANNELS=$c NCCL_MAX_NCHANNELS=$c NCCL_NVLS_NCHANNELS=$c
    else run ${a}_${p}_$c NCCL_ALGO=$a NCCL_PROTO=$p NCCL_MIN_NCHANNELS=$c NCCL_MAX_NCHANNELS=$c; fi
  done; done
done
echo "# sweep done $(date +%T)"
