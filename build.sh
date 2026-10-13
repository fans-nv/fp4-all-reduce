#!/bin/bash
# Builds every shipped variant into build/<tag>/fp4ar_bench (+ fp4ar_norm_bench), or compile-checks them offline.
# Usage: [ARCH=107a|100a] [NVCC=...] [NCCL_HOME=...] [NCCL_INC=...] [NCCL_LIB=...] [FP4AR_DIAG=1] ./build.sh [all|check|verify] [tag]
#   all    (default) objects + link; needs libnccl (>= 2.31 for ncclGetLsaDevicePointer) for ARCH's host
#   check  compile every variant object, the harness object and the norm objects; no link (offline x86 is fine)
#   verify md5-check kernels/include against their frozen campaign-r2 origins (kernels/verify_origin.sh)
# A variant whose object fails to compile is skipped (logged in build.log), so one lane cannot block the others.
set -o pipefail
R=$(cd "$(dirname "$0")" && pwd)
CMD=${1:-all}; TAG=${2:-$(date +%m%d-%H%M)}
[ "$CMD" = verify ] && exec "$R/kernels/verify_origin.sh"
ARCH=${ARCH:-107a}                      # 107a = VR200 (CUDA >= 13.4), 100a = GB200
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
NCCL_INC=${NCCL_INC:-${NCCL_HOME:+$NCCL_HOME/include}}; NCCL_INC=${NCCL_INC:-/usr/include}
NCCL_LIB=${NCCL_LIB:-${NCCL_HOME:+$NCCL_HOME/lib}}; NCCL_LIB=${NCCL_LIB:-/usr/lib/$(uname -m)-linux-gnu}
B=${BUILD_DIR:-$R/build}/$TAG; mkdir -p "$B"
# -gencode, not -arch=sm_107a: the latter rejects cvt.e2m1x2 (TOOLCHAIN note).
NV="$NVCC -O3 -std=c++17 -gencode arch=compute_$ARCH,code=sm_$ARCH -lineinfo -Xptxas -v -I$R/include -I$R/harness -I$NCCL_INC"
LISTS=$(ls "$R"/kernels/*/variants.txt); [ "${FP4AR_DIAG:-0}" = 1 ] && LISTS="$LISTS $(ls "$R"/kernels/*/variants.diag.txt)"
echo "# $CMD $TAG arch=$ARCH nvcc=$($NVCC --version | tail -1) nccl_inc=$NCCL_INC" | tee "$B/build.log"
OBJS=(); LIST=""; FAIL=0
while IFS='|' read -r name src flags; do
  name=$(echo $name); src=$(echo $src)
  [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  $NV $flags -DFP4AR_NAME=$name -DFP4AR_REGISTER=fp4ar_register_$name -c "$R/$src" -o "$B/v_$name.o" > "$B/v_$name.log" 2>&1
  rc=$?; echo "variant $name rc=$rc ($src $flags) regs: $(grep -o 'Used [0-9]* registers' "$B/v_$name.log" | sort | uniq -c | tr -s ' ' | paste -sd';')" | tee -a "$B/build.log"
  grep -q -E '[1-9][0-9]* bytes spill' "$B/v_$name.log" && echo "  WARNING: $name spills" | tee -a "$B/build.log"
  if [ $rc = 0 ]; then OBJS+=("$B/v_$name.o"); LIST="$LIST V($name)"; else FAIL=1; tail -5 "$B/v_$name.log" | sed 's/^/  /' | tee -a "$B/build.log"; fi
done < <(cat $LISTS)
# Fused norm: not an Fp4ArVariant; its own driver (bench/fused_norm/drv_norm.cu, HARNESS-owned) links lnorm.o.
$NV -c "$R/kernels/fused_norm/lnorm.cu" -o "$B/lnorm.o" > "$B/lnorm.log" 2>&1; rc=$?
echo "fused_norm lnorm rc=$rc" | tee -a "$B/build.log"; [ $rc = 0 ] || { FAIL=1; tail -5 "$B/lnorm.log"; }
DRV=$R/bench/fused_norm/drv_norm.cu
if [ -f "$DRV" ]; then
  $NV -I"$R/kernels/fused_norm" -c "$DRV" -o "$B/drv_norm.o" > "$B/drv_norm.log" 2>&1; rc=$?
  echo "fused_norm driver rc=$rc" | tee -a "$B/build.log"; [ $rc = 0 ] || { FAIL=1; tail -5 "$B/drv_norm.log"; }
fi
$NV "-DFP4AR_VARIANT_LIST=$LIST" -c "$R/harness/fp4ar_bench.cu" -o "$B/fp4ar_bench.o" > "$B/harness.log" 2>&1 || { tail -20 "$B/harness.log"; exit 1; }
echo "harness object ok, variants:$LIST" | tee -a "$B/build.log"
if [ "$CMD" = check ]; then echo "# check done (no link) fail=$FAIL" | tee -a "$B/build.log"; exit $FAIL; fi
[ -e "$NCCL_LIB/libnccl.so" ] && LN="-lnccl" || LN="-l:libnccl.so.2"
$NV -o "$B/fp4ar_bench" "$B/fp4ar_bench.o" "${OBJS[@]}" -L"$NCCL_LIB" $LN -lpthread -ldl >> "$B/harness.log" 2>&1 || { tail -20 "$B/harness.log"; exit 1; }
[ -f "$B/drv_norm.o" ] && { $NV -o "$B/fp4ar_norm_bench" "$B/drv_norm.o" "$B/lnorm.o" -L"$NCCL_LIB" $LN -lpthread >> "$B/harness.log" 2>&1 || { tail -20 "$B/harness.log"; exit 1; }; }
echo "# built $B/fp4ar_bench variants:$LIST" | tee -a "$B/build.log"
ln -sfn "$B" "${BUILD_DIR:-$R/build}/latest"
