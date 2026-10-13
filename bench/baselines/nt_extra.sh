#!/bin/bash
# Extra nccl-tests configs: symmetric registration (-R 2), channel counts beyond grid, NTHREADS, CUDA graph.
# Args: <n> <coll ar|ag|a2a> <rep>
. "$(dirname "$0")/../../scripts/env.sh"
[ "$NCCL_HOME" != /usr ] && export LD_LIBRARY_PATH=$NCCL_LIBDIR:$LD_LIBRARY_PATH
T=${NT:-$BUILD/nccl-tests}; n=$1; coll=$2; rep=$3
O=$RUNS/baselines/nccltests/tp$n/$rep; mkdir -p $O
case $coll in
  ar)   cmds=("all_reduce_perf -d bfloat16 -o sum -b 12582912 -e 201326592 -f 2");;
  ag)   cmds=("all_gather_perf -d uint8 -b $((3543040*n)) -e $((56688640*n)) -f 2" "all_gather_perf -d uint8 -b 3543040 -e 56688640 -f 2");;
  a2a)  cmds=("alltoall_perf -d uint8 -b 3543040 -e 56688640 -f 2");;
esac
run() { # tag "extra args" env...
  local tag=$1 xa=$2; shift 2
  for ci in "${!cmds[@]}"; do
    local c=${cmds[$ci]}; local f=$O/${coll}${ci}_$tag.log
    echo "# env: $* cmd: $c $xa" > $f
    env "$@" timeout 120 $T/$c -g $n -w 20 -n 100 -c 0 -a 3 $xa >> $f 2>&1 || echo "# rc=$?" >> $f
  done
}
run R2_default "-R 2" NCCL_DEBUG=VERSION
for a in Ring NVLS Tree; do run R2_${a}_auto "-R 2" NCCL_ALGO=$a; done
for p in LL128 Simple; do for c in 64; do run R2_Ring_${p}_$c "-R 2" NCCL_ALGO=Ring NCCL_PROTO=$p NCCL_MIN_NCHANNELS=$c NCCL_MAX_NCHANNELS=$c; done; done
for c in 32 64; do run R2_NVLS_Simple_$c "-R 2" NCCL_ALGO=NVLS NCCL_PROTO=Simple NCCL_NVLS_NCHANNELS=$c NCCL_MIN_NCHANNELS=$c NCCL_MAX_NCHANNELS=$c; done
for p in LL128 Simple; do for c in 48 96 128; do run Ring_${p}_$c "" NCCL_ALGO=Ring NCCL_PROTO=$p NCCL_MIN_NCHANNELS=$c NCCL_MAX_NCHANNELS=$c; done
  run Ring_${p}_64_nt512 "" NCCL_ALGO=Ring NCCL_PROTO=$p NCCL_MIN_NCHANNELS=64 NCCL_MAX_NCHANNELS=64 NCCL_NTHREADS=512; done
run Ring_LL128_64_G "-G 10" NCCL_ALGO=Ring NCCL_PROTO=LL128 NCCL_MIN_NCHANNELS=64 NCCL_MAX_NCHANNELS=64
run Ring_Simple_64_G "-G 10" NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=64 NCCL_MAX_NCHANNELS=64
run Ring_Simple_64_dbg "" NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=64 NCCL_MAX_NCHANNELS=64 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING
echo "# extra done $(date +%T)"
