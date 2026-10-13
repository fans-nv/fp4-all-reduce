#!/bin/bash
# Proves each packaged file is its frozen campaign-r2 origin: drop our 1-line header, undo the one include-path edit, compare md5.
R=$(cd "$(dirname "$0")/.." && pwd); rc=0
while read -r f want origin; do
  got=$(tail -n +2 "$R/$f" | sed 's#^\#include "codec_fast.cuh"#\#include "../codec/codec_fast.cuh"#' | md5sum | cut -c1-32)
  [ "$got" = "$want" ] && s=OK || { s=MISMATCH; rc=1; }
  printf '%-8s %-34s %s  (origin %s)\n' $s "$f" "$got" "$origin"
done <<'T'
include/fp4ar_api.h              2961b4c716373116810aa36464fc445f include/fp4ar_api.h
include/fp4_codec.cuh            37e0905a3fed452d74430ae69a3740a7 include/fp4_codec.cuh
include/codec_fast.cuh           a3f51937b202eb93c3bed9668c919b24 kernels/codec/codec_fast.cuh
kernels/push/tp2_l2.cu           891fd77acfebdb729966d03195891e4e kernels/lamport/snap/f1/lamport.cu
kernels/push/tp4_l4.cu           3e4a869f46ef975fd0f565eb9fc590d1 kernels/lamport_tp4/l4.cu
kernels/pull/tp2_p.cu            1cfd0d69535e1cdeaa5ab898cd0e06ca kernels/pull/fp4ar_pull.cu
kernels/pull/tp2_pq.cu           ded06a9b2aa272d6c5b2d4dd636b8181 kernels/pull/fp4ar_pullq.cu
kernels/pull/tp4_p4.cu           60a9a226a328270581b61eb224825765 kernels/pull_tp4/fp4ar_pull4.cu
kernels/pull/tp4_p4q.cu          15af347d5ab3f0096b653fb1287f4a00 kernels/pull_tp4/fp4ar_pull4q.cu
kernels/fused_norm/lnorm.cu      f95ded56abc71cc637f4683e7e39a1ff kernels/lamport_norm/lnorm.cu
kernels/fused_norm/lnorm_epi.cuh ceeed7f709d80e6d2601ebc074c1ce89 kernels/lamport_norm/lnorm_epi.cuh
kernels/control/r1.cu            3870251fdb637af3b11d01707d9fcf4d kernels/r1/r1.cu
T
exit $rc
