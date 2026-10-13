# What vLLM integration would need

Status: **not started.** The kernels run only in a standalone C++ benchmark (`fp4ar_bench`) and a standalone fused-norm driver. Everything below is a plan derived from reading vLLM 0.30.0+65c589c9 (the VR200 image) and from the kernels' preconditions. File:line references are relative to the vLLM package in that image.

## 1. Where it plugs in today

M3 TP2 prefill production recipe: `--tensor-parallel-size 2`, `cudagraph_mode FULL`, `--max-num-batched-tokens 16384`, default `-O2`; env `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm NCCL_ALGO=Ring NCCL_MIN_NCHANNELS=64 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1`. Checkpoint `nvidia_minimax-m3-nvfp4`: attention, shared experts and dense MLP are MXFP8; routed experts NVFP4.

Two entry points:

| Path | Where | Today on sm_107 TP2 |
| --- | --- | --- |
| Compile-time fusion `AllReduceFusionPass` | `compilation/passes/fusion/allreduce_rms_fusion.py` | on at O2 for TP>1, capability family 10.x. Applies only to compile ranges with `end <= 5461` rows (`FI_ALLREDUCE_FUSION_MAX_SIZE_MB[107][2] = 64` MiB / 12 KiB, `:104-128`, `:1022-1034`, `:1134-1138`). One-shot if <= 32 MiB (`:132-155`), else two-shot. For M3 only `kARResidualRMSNorm` (BF16 out, Gemma variant) fires |
| Runtime `CudaCommunicator.all_reduce` | `distributed/device_communicators/cuda_communicator.py:327-400` | NCCL symm-mem AR (off at TP2: needs world size >= 4) -> FlashInfer standalone AR (<= 64 MiB) -> custom AR / torch symm_mem (<= 4 MiB) -> pynccl |

Resulting path per tier: 1K/2K FlashInfer fused one-shot; 4K FlashInfer fused two-shot (1.35x slower than one-shot); 8K/16K pynccl + separate `fused_add_rms_norm` (2 kernels per block).

## 2. Torch custom op and communicator

- Expose two ops: `fp4_all_reduce(x) -> out` (BF16 in, BF16 out) and `fp4_all_reduce_add_rms_norm(x, residual, weight, eps, weight_bias) -> (out, residual)` (residual updated in place), each with an NVFP4-input overload taking `(xq, gs[])`.
- Register them as torch custom ops with fake/meta implementations so `torch.compile` and the fusion pass can trace them.
- Add a branch in `CudaCommunicator.all_reduce` ahead of pynccl, gated by an env flag, TP size (2 or 4), dtype BF16, hidden 6144 (or any H the codec supports), and a row range.
- One context per TP group, created at communicator init after the NCCL communicator exists.

## 3. Symmetric-window allocation (NCCL 2.31)

- Allocate with `ncclMemAlloc` and register with `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)`; peer pointers from `ncclGetLsaDevicePointer`. NCCL >= 2.31 is required (2.28 lacks the host LSA pointer API).
- Size (push): 3 one-shot + 3 two-shot + 3 NVFP4-input receive buffers plus an input slot, about **1 GB per rank at TP2 16K rows**, linear in P and max rows. Pull needs ~455 MB at 16K. This competes with KV-cache memory and must be accounted for in `determine_available_memory` (allocate before the profiling run, or reserve it explicitly).
- Only allocate the regions actually dispatched (e.g. no two-shot region at TP2, no one-shot region at TP4) to cut the footprint.
- Init sentinel-fills every buffer, then runs one NCCL all-reduce so no rank pushes before all are armed.

## 4. CUDA graph capture

- Call index is device-side (per-CTA slot, read after `griddepcontrol.wait`), so the kernel needs no host per-call argument and is graph-capturable. Verified: 10,000 graph calls per gate cell, plus alternating row counts in one process.
- Every call on a context must be issued on all ranks in the same order (it is: AllReduce is collective).
- PDL is used (`griddepcontrol.wait` / `launch_dependents`); the launch must set the programmatic-stream-serialization attribute, inside graphs too.
- Eager warmup and graph capture can share the context: the device call counter just keeps counting.

## 5. Call-index state and fixed grid

- Per-CTA call state assumes **every kernel in a context uses the same grid** (`sm_count`), independent of rows. The launcher must not shrink the grid for small row counts, and must not share a context between kernels launched with different grids.
- If another kernel family (e.g. fused-norm and plain AR) uses the same buffers, it must use the same grid and the same call counter. Simplest: one context per (TP group, kernel family) with separate windows; or merge the families into one call-state region.
- Pull NVFP4-input window mode has an extra **slot precondition**: two consecutive AR calls never get the same slot. With the period-5 host rotation this fails only when AR calls per captured graph `K ≡ 1 (mod 5)`; M3 has 121 AR+norm blocks per step (`121 mod 5 = 1`), so **the period must change** (or insert a dummy `input_slot` per graph). The push kernels have no such precondition.

## 6. Lifting the 64 MiB cap

- The fusion pass splits compile ranges at 5461 rows (`config/vllm.py:2433-2453`); 8K and 16K chunks fall outside and run unfused NCCL. Our kernel has no such cap (tested to 16K rows = 192 MiB).
- Needed: a per-backend max size (`FI_ALLREDUCE_FUSION_MAX_SIZE_MB` equivalent) set to the window's max rows (16384 for `max-num-batched-tokens 16384`), so the compile range covers the whole prefill range and the pass applies to every chunk size.
- The one-shot/two-shot selector must be replaced by the measured dispatch: TP2 one-shot always; TP4 two-shot always.

## 7. Fusion-pass hookup for AR + RMSNorm

- Register a replacement for the `kARResidualRMSNorm` Gemma pattern (`allreduce_rms_fusion.py:1076-1125`) that emits `fp4_all_reduce_add_rms_norm`. M3: `eps = 1e-6`, `use_gemma_norm = true`, `weight_bias = 1.0`.
- Semantics to match: output `y = bf16(x * rinv * (1 + w))`, residual updated to `bf16(sum + residual)`. Known deviation: our numerator uses the BF16-rounded residual sum (<= one extra 2^-9 rounding); FlashInfer uses FP32. Decide whether that is acceptable or fix it (costs registers).
- The fused kernel is TP2-only today. TP4 needs the epilogue on the two-shot owner/all-gather path (not built).
- `rinv` uses `rsqrt.approx.ftz` as FlashInfer does.

## 8. Scale layouts to match vLLM

| Output | Ours | vLLM expects | Work |
| --- | --- | --- | --- |
| BF16 | `[M, 6144]` | same | none |
| NVFP4 out (not used by M3 production) | row-interleaved 3456 B rows, linear scales | separate `[M, N/2]` values and `[M, N/16]` scales, scales swizzled 128x4 by default; `input_global_scale = 448*6/amax` | write separate tensors and swizzle the scale store; bit-compare against `scaled_fp4_quant` |
| MXFP8 out | `[M, 6144]` E4M3 + 192 B UE8M0 per row, linear | FlashInfer cute-dsl `mxfp8_quantize` on B200+ (layout per the consuming GEMM) | bit-compare and match the scale layout; our rounding follows vLLM's torch reference with an exact `ceil(log2(amax/448))` (torch's float `log2` can differ just above powers of two) |
| NVFP4 input (from the GEMM) | 3456 B rows (values then scales), static per-rank `gs[r]` as argument, -0 canonicalized | no producer exists | GEMM epilogue change (next section) |

## 9. NVFP4-input mode: GEMM epilogue contract

The NVFP4-input variants (`L2q`, `L4q`) are the fastest but assume the row-parallel GEMM before the AllReduce (o_proj, MLP / shared-expert down projections) emits NVFP4:
- Per-tensor static scale per rank `gs[r]`, calibrated offline (a GEMM epilogue cannot compute a row amax spanning 24+ N tiles). The benchmark emulates it with `gs = amax / 2688`.
- 3456 B rows: 3072 E2M1 bytes (low nibble = even element), then 384 E4M3 block scales; E2M1 -0 canonicalized to +0 (3 integer ops per 8 values); scales <= `0x7E`.
- All ranks know all `gs[r]` (passed as kernel arguments; graph-capturable as long as they are static).
- The epilogue must not write the window slot before its own `griddepcontrol.wait` (pull window mode only).
- Calibration quality directly sets accuracy: with a static scale, rows far below the calibrated range lose resolution. Must be evaluated with real partials.

## 10. Accuracy evaluation plan

Measured so far: synthetic only (rel-L2 vs FP32 sum 0.095 one-shot, 0.133 two-shot; BF16 transport 0.0018).

1. **Real partials.** Dump per-layer row-parallel partial outputs from an M3 TP2 (and TP4) prefill on representative prompts (short and 16K+ contexts). For each AllReduce site: rel-L2 and max error vs the exact FP32 sum for BF16, FP8 E4M3, NVFP4 dynamic row scale, NVFP4 static per-tensor scale (calibrated on a held-out set).
2. **Propagation.** Inject the NVFP4 AllReduce error per layer in a single-GPU emulation of the TP2 forward (decode-sum emulation using the same codec) and track hidden-state drift and logit KL vs BF16 transport.
3. **Task eval.** GSM8K and a long-context task at TP2 with the kernel in the serving path; compare against the BF16-transport baseline on the same image and node, with repeats to establish the noise band.
4. **Policy.** If any layer class is sensitive, allow per-layer opt-out (BF16 transport for those sites) and report the resulting per-step saving.
5. Prefill only: decode keeps its existing BF16 path.

## 11. Performance validation after integration

- Same-node A/B vs the production image at 1K/4K/16K and 64K ISL: TTFT p50 at concurrency 1, and throughput at production load (prefill+decode mixed steps).
- nsys: confirm one kernel per AR+norm block and 121 blocks per step; assert the kernel table is non-empty.
- Check the projected per-step saving ([RESULTS.md](RESULTS.md) section 5) against the measured step time.
