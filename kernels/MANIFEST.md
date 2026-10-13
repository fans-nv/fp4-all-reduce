# Kernel manifest

Every kernel source here is a frozen round-2 file, copied from `campaign-r2/` with exactly two edits:
1. A one-line header comment is prepended.
2. `#include "../codec/codec_fast.cuh"` becomes `#include "codec_fast.cuh"`, because the headers now sit flat in `include/`.

The kernel logic is unchanged. `kernels/verify_origin.sh` (or `./build.sh verify`) reverses both edits and checks each file against its origin md5. That check was run, and it also passed a negative control: a one-byte append makes it report MISMATCH and exit 1.

Each origin md5 was checked twice:
- against the local `campaign-r2/` copy;
- against the cluster copy `$ROOT/src/r2/` (`$ROOT` = the measurement campaign's work directory), together with the freeze manifests `$ROOT/P4/build/b6/MD5SUMS`, `$ROOT/P4/frozen-b6/MD5SUMS` and `$ROOT/tune/snap-z0255/MD5SUMS`.

All three agree for every file, and no mismatch was found.

## Files

Headlines are graph-mode medians in µs at 1K/2K/4K/8K/16K rows, VR200, from `campaign-r2/RESULTS-vr200.md` and `kernels/lamport_norm/NOTES.md`.

| Packaged file | Origin (campaign-r2/) | Origin md5 | Packaged md5 | Frozen build (binary md5) | Variants | Measured headline |
| --- | --- | --- | --- | --- | --- | --- |
| `kernels/push/tp2_l2.cu` | `kernels/lamport/snap/f1/lamport.cu` | 891fd77acfebdb729966d03195891e4e | 8555692e34862f4482faf0beb70e4e7a | `L-f1` (91914d4b) | L2, L2q | L2: 10.8/15.2/21.9/34.4/60.7 (2.86-3.64x tuned BF16). L2q: 8.6/12.5/18.4/30.8/55.4 (3.58-4.20x). TP2 dispatch winner |
| `kernels/push/tp4_l4.cu` | `kernels/lamport_tp4/l4.cu` | 3e4a869f46ef975fd0f565eb9fc590d1 | c709fea5487752023dfc57f4308818b0 | `tp4push-f1` (5d064fab) | L4, L4q | Two-shot. L4: 19.8/25.6/34.8/56.2/103.8 (2.03-3.15x). L4q: 16.4/21.2/30.7/50.5/98.3 (2.45-3.50x). TP4 dispatch winner |
| `kernels/pull/tp2_p.cu` | `kernels/pull/fp4ar_pull.cu` | 1cfd0d69535e1cdeaa5ab898cd0e06ca | b66f9ab2c27dbc8dc912d5a6dcc8e759 | `q4` (b69bd896) | P (+ diagnostics) | P: 16.1/20.5/30.3/45.7/74.5 (1.92-2.91x) |
| `kernels/pull/tp2_pq.cu` | `kernels/pull/fp4ar_pullq.cu` | ded06a9b2aa272d6c5b2d4dd636b8181 | ba4ae94b7314485bbf9c09dc33720300 | `q4` (b69bd896) | Pq | Pq window, tuned: 9.9/12.9/22.3/37.1/65.1 (3.12-3.92x) |
| `kernels/pull/tp4_p4.cu` | `kernels/pull_tp4/fp4ar_pull4.cu` | 60a9a226a328270581b61eb224825765 | 3d879a4e4badbd1ab8a085bb98bfe6b5 | `b6` (d482d6b1) | P4, P4t | P4, two-shot, BF16: 32.1/37.1/48.0/68.2/116.3. P4t BF16: 26.1/29.5/44.6 (1K-4K only) |
| `kernels/pull/tp4_p4q.cu` | `kernels/pull_tp4/fp4ar_pull4q.cu` | 15af347d5ab3f0096b653fb1287f4a00 | c3d023da3f4fa4221d13b12abe424e22 | in `tp4push-f1`, not in `b6` | P4q | Not measured in the final pass (lane reference kernel) |
| `kernels/fused_norm/lnorm.cu` | `kernels/lamport_norm/lnorm.cu` | f95ded56abc71cc637f4683e7e39a1ff | 2a53bc2b48bd013ccf5e4aa283f682ee | `LN-t10` (drv 44fb7670), source defaults | lnorm_* API (not an Fp4ArVariant) | TP2 AR+residual+RMSNorm. BF16 out: 16.9/22.9/34.9/61.2/113.4. NVFP4 in: 14.1/21.8/34.9/55.2/96.8. NVFP4 out: 17.7/24.2/37.4/75.5/131.0. MXFP8 out: 18.1/24.4/37.7/75.7/130.9 |
| `kernels/fused_norm/lnorm_epi.cuh` | `kernels/lamport_norm/lnorm_epi.cuh` | ceeed7f709d80e6d2601ebc074c1ce89 | ec615ae194c89db3de6d03c298e2081b | `LN-t10` | (epilogue math) | — |
| `kernels/control/r1.cu` | `kernels/r1/r1.cu` | 3870251fdb637af3b11d01707d9fcf4d | 0830fd645499ddbe340945d106a127f1 | `L-f1` / `tp4push-f1` | r1 | TP2: 20.9/31.0/43.4/80.2/128.3. TP4: 30.5/48.6/77.1/112.9/169.3 |
| `include/fp4ar_api.h` | `include/fp4ar_api.h` | 2961b4c716373116810aa36464fc445f | 1423cbcf3956ded1c3990c7f1d4978b1 | all | — | — |
| `include/fp4_codec.cuh` | `include/fp4_codec.cuh` | 37e0905a3fed452d74430ae69a3740a7 | 0c2b8811b4e6dd29433df9290f4db5ba | all | — | — |
| `include/codec_fast.cuh` | `kernels/codec/codec_fast.cuh` | a3f51937b202eb93c3bed9668c919b24 | 65c1eb1d1d49174cca38f6b384e30948 | all | — | — |

LN-t10 has no MD5SUMS file. Its source was identified from three facts:
- the local `lnorm.cu` was last edited at 02:30:58, and LN-t10 was built at 02:32;
- t10 is the last iteration in NOTES, and the only later builds (t11u*) changed `-DLNU` only;
- the later snapshot `tune/snap-z0255/MD5SUMS` lists f95ded56.

This is strong evidence but not proof from a byte-level manifest.

## Variants (`kernels/*/variants.txt`, format `name | source | extra nvcc flags`)

Each line is built with `-DFP4AR_NAME=<name> -DFP4AR_REGISTER=fp4ar_register_<name>`, and names match the measured results 1:1.

Two pairs are the same object under different names:
- L2 / L2q;
- L4 / L4q.

The harness picks the input format at run time with `--input bf16|nvfp4|nvfp4-window`.

| Name | Source | Flags | Role |
| --- | --- | --- | --- |
| L2, L2q | push/tp2_l2.cu | `-DLAMPORT_V2=1` | Shipped: TP2 one-shot |
| L4, L4q | push/tp4_l4.cu | — | Shipped: TP4 two-shot |
| P, Pq | pull/tp2_p.cu, tp2_pq.cu | — | Comparison: pull family |
| P4, P4t | pull/tp4_p4.cu | P4t: `-DAG_PUSH=2` | Comparison |
| P4q | pull/tp4_p4q.cu | — | Reference |
| r1 | control/r1.cu | `-DMINB=3 -DR1_NAME=r1` | Round-1 control |
| P_enc0, P_flagsys, P_gpudiag | pull/tp2_p.cu | `-DENC_FAST=0`; `-DPROTO_LAMP=0`; `-DPROTO_LAMP=0 -DPUB_GPU=1` | Diagnostics, in `pull/variants.diag.txt`. Built only with `FP4AR_DIAG=1` |

**Dropped (not packaged):**
- **lamport.cu modes:** L (the old warp-specialized default, built without `LAMPORT_V2`) and LT/LTsw (TMA validity, 214.7 µs at 16K). These modes remain in the source behind its defines, but no variant line builds them.
- **Lamport snapshots:** it6/it10/it11/q3 and the experimental knobs L2g2/L2w8/L2l2.
- **r1c4:** the round-1 MINB=4 configuration.
- **Unbuilt P4 diagnostics:** P4h, P4t8 and P4tnp, which were never in a frozen binary.
- **L4p (lane-only):** `-DTS_OLD=0`.
- **Lane tooling:** drv.cu, debug_ts.cu, tma_peer_test.cu, and the lane scripts.

The fused-norm driver `drv_norm.cu` belongs to HARNESS (`bench/fused_norm/`).

## Offline compile check (2026-10-13)

`./build.sh check` was run with pip nvcc 13.4.92 and NCCL 2.31.2 headers on x86, for both `ARCH=107a` and `ARCH=100a`, with `FP4AR_DIAG=1`. All 13 variant objects and `lnorm.o` compiled with rc=0, and so did `drv_norm.o`.

The sm_107a register counts for L2 (120/148/158/168/30/44/96) and P/P4 match the cluster nvcc 13.5 `tp4push-f1` build.log. The "spills" warnings also appear in that frozen log.

The harness object needs `cuda_profiler_api.h`, which the pip toolchain lacks. With a 3-line stub declaring `cudaProfilerStop`/`cudaProfilerStart`, `fp4ar_bench.cu` compiles with all 10 default variants.

Not verified offline:
- linking (no aarch64 libnccl);
- nvcc 13.5 codegen;
- any run.
