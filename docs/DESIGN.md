# Design

Fused NVFP4-transport tensor-parallel AllReduce for MiniMax-M3 prefill (hidden H = 6144, BF16 rows of 12 KiB) on VR200 (sm_107a) and GB200 (sm_100a).

Contents:
1. Codec spec v2
2. Wire formats and sentinels
3. Properties shared by every kernel
4. Push kernel, TP2 one-shot (`L2`, `L2q`)
5. Lamport buffer protocol and proof (push)
6. Push kernel, TP4 two-shot (`L4`, `L4q`)
7. Fused residual + RMSNorm epilogue
8. Pull kernels (`P`, `Pq`, `P4`, `P4t`) and why they lose on VR200
9. Rejected designs

Variant names (`L2`, `L2q`, ...) are the names the benchmark binary accepts in `--variant`.

## 1. Codec spec v2

All kernels and the reference use the same codec, so the correctness gate is bitwise.

| Step | Definition |
| --- | --- |
| Row scale | `rs = amax(row) / 2688` (2688 = 6 x 448); `inv = 2688 / amax`, or 0 if amax == 0 |
| Block scale, per 16 values | `s = e4m3_satfinite(rn((blockmax * inv) / 6))`; `x/6` is computed exactly (Markstein) |
| Values | `q = e2m1_satfinite(f * (inv / e4m3(s)))` |
| -0 canonicalization | E2M1 -0 (nibble `0x8`) is mapped to +0 (3 integer ops per 8 values). Output-neutral: every sum starts at +0 |
| Decode-accumulate | `acc = fmaf(e2m1(q), e4m3(s) * rs, acc)`, summed in rank order 0..P-1 from `acc = +0`, FP32 |
| Two-shot | the row owner re-encodes the FP32 sum with the same encoder; every rank (owner included) outputs the decode of that re-encoding |
| NVFP4-input mode | no row scale; per-rank static per-tensor scale `gs[r]` passed as a kernel argument. Decode is `e2m1(q) * e4m3(s) * gs[r]` |

Hardware notes:
- `cvt.rn.satfinite.e2m1x2.f32` maps NaN to `0x7` (+6). This arises only for rows with amax < ~2^-116 (inv = inf).
- Build with `-gencode arch=compute_107a,code=sm_107a` (or `100a`). Plain `-arch=sm_107a` fails in ptxas for `cvt.e2m1x2`.
- Do not pack E2M1 bytes with `mov.b32 {b0,b1,b2,b3}` from `.b8` registers. nvcc 13.0 miscompiles it when an encode feeds a decode in registers (see [HISTORY.md](HISTORY.md)). The codec crosses the asm boundary with 16-bit values.

Codec building blocks (`codec_fast.cuh`, VR200, 16K rows, cold L2, vs the 11.66 TB/s copy roof):

| Path | µs | % of copy roof |
| --- | --: | --: |
| Encode from smem (TMA-landed rows, 18 warps, 63 regs) | 28.6 | 77% |
| Encode from registers | 34.9 | 63% |
| Decode-sum P=2, staged 16 B stores | 31.8 | 85% |
| Decode-sum P=4, staged | 45.9 | 80% |
| One-shot compute P=2 (encode + decode-sum) | 57.0-59.1 | 75-78% |

Techniques that mattered (measured on VR200):
- Decode output through a 2 KB per-warp smem staging buffer (lane-contiguous 16 B stores): 58% -> 85% of roof.
- Encoder re-reads smem in XOR-swizzled order and does not cache the row in registers between its two passes: 27% -> 73%.
- Packed math: `mul.rn.f32x2`, `fma.rn.f32x2`, `cvt e4m3x2`, `cvt f16x2.e4m3x2`.

## 2. Wire formats and sentinels

| Format | Per-row layout | Bytes |
| --- | --- | --: |
| BF16 input, push receive slot | 3072 B E2M1 values (low nibble = even element), 384 B E4M3 block scales (scale k covers values 16k..16k+15), 16 B row scale (FP32 `rs` replicated 4x); planar per buffer (values, scales, row scales) | 3472 |
| BF16 input, wire payload counted in bandwidth figures | 3072 + 384 + 4 | 3460 |
| NVFP4 input (GEMM epilogue output, and its receive slot) | 3072 B values then 384 B scales, row-interleaved, 16 B aligned, row r at `xq + r*3456`; no row scale (static `gs[r]`) | 3456 |
| Pull record | 3072 B values, 384 B scales, 4 B row scale, 12 B pad | 3472 |
| Fused-norm NVFP4 output | 3072 B values then 384 B scales, linear (not vLLM's swizzled 128x4) | 3456 |
| Fused-norm MXFP8 output | 6144 B E4M3 values then 192 B UE8M0 scales (one per 32), linear | 6336 |

Why the row scale is 16 B: TMA bulk copies move 16 B multiples, and the Rubin TMA `report::validity::per_16bytes::80000000` check inspects only the first element of each 16 B granule. Replicating `rs` 4x makes every granule's first element meaningful, so the same slot layout is valid for both software polling and hardware-validity loads.

Sentinels (the receive buffer is pre-filled with values no writer can produce):

| Field | Sentinel | Why a writer never produces it |
| --- | --- | --- |
| Value word (32 bit) | `0x88888888` (every nibble = E2M1 -0) | encoder canonicalizes -0 to +0; a receiver treats any `0x8` nibble as "not yet written" |
| Scale byte | `0xFF` (push) / `0x88` (pull NVFP4-input slots) | valid scales are <= `0x7E` |
| Row scale | `0x80000000` (FP32 -0) | `rs >= +0` |

Receivers check each 32-bit value word, each scale byte and each row-scale word independently. A word without sentinels is entirely from the current call, so correctness does not depend on 16 B store atomicity. (Measured anyway on VR200: 0 torn 16 B packets in 1.2-1.9e12 reads over NVLink.)

## 3. Properties shared by every kernel

- One fused kernel per call; one persistent CTA per SM; grid fixed at `sm_count`.
- No system-scope fence on the hot path. On VR200 a `st.release.sys` / sys fence costs 12-21 µs under NVLink load, and a per-tile release cut a push loop from 927 to 187 GB/s.
- Readiness is carried in the data (Lamport sentinels), not in flags.
- Call index lives on the device (per-CTA slots, read after `griddepcontrol.wait`), so nothing is passed from the host per call and the kernel is CUDA-graph capturable.
- PDL: barrier init and the first input TMA load are issued before or right after `griddepcontrol.wait`; every window access is after it. nsys shows each AR kernel starting 7-12 µs before its predecessor ends.
- Peers are addressed through an NCCL 2.31 symmetric window (`ncclMemAlloc` + `ncclCommWindowRegister(NCCL_WIN_COLL_SYMMETRIC)` + `ncclGetLsaDevicePointer`).

## 4. Push kernel, TP2 one-shot (`L2` BF16 input, `L2q` NVFP4 input)

Shape: 1 CTA/SM, 352 threads = 10 worker warps forming 5 two-warp row teams, plus 1 reset warp. About 225 KB smem: per warp a 12 KB x buffer, a 2 x 3472 B own-row ring and 2 KB store staging; plus 3 sentinel pattern pages. 140-168 registers, no spills.

Rows: each CTA gets a contiguous chunk, identical on every rank. Team t takes rows `r0 + t + 5k`.

Per row (leader and partner warp):

| Step | `L2` (BF16 input) | `L2q` (NVFP4 input) |
| --- | --- | --- |
| 1. Load | leader TMA-loads the 12 KB BF16 row into smem; the next load is issued as soon as the previous row is encoded | leader TMA-loads the 3456 B record into a 3-record smem ring |
| 2. Encode | each warp encodes half the pair columns; row amax exchanged through smem and a 64-thread named barrier; result written to the own-row ring in wire layout | none |
| 3. Egress | leader TMA bulk-stores values, scales, row scale to the peer's receive slot `[buf][me][row]`. No fence, no flag | one TMA bulk store of the record |
| 4. Wait (one row later) | each warp polls its 16 B value words and scale halfwords with `ld.volatile` until sentinel-free; all invalid words of a group are re-polled together | same |
| 5. Decode | own source from smem, peer from the receive slot; `dec_acc_fast` in rank order; BF16 out through staged 16 B stores | same, with `rs = gs[r]` |

The reset warp re-arms this CTA's share of the previous call's receive buffer with TMA bulk stores from the sentinel pattern pages.

Measured effects of each design step (VR200, TP2, graph, µs at 16K / 1K, BF16 input):

| Step | 16K | 1K |
| --- | --: | --: |
| Round-1 fused kernel (control) | 128.5 | 20.6 |
| Warp-specialized Lamport push, no fences | 97.2 | 13.3 |
| Warp-per-row pipeline on the fast codec | 73.3 | 13.6 |
| TMA bulk-store egress (vs `st.v4`) | 68.2 | 13.4 |
| Parallel re-poll of invalid words | 67.5 | 13.6 |
| Two-warp row teams at all sizes | 62.2 | 11.8 |
| Per-CTA call state (no exit atomic), first TMA before call-state read (final) | 60.7 | 10.6 |

## 5. Lamport buffer protocol and proof (push)

State:
- Call state per CTA: `ctrl[4*blockIdx] = {call, -, rows, algorithm+1}`. Each kernel reads its own slot after `griddepcontrol.wait`; thread 0 rewrites it at exit with plain stores (kernel completion makes them visible to the next kernel). No grid-wide atomic, fence or exit barrier.
- Precondition: every kernel in a context uses the same grid. CTAs owning no rows still update their slot.
- Buffers: 3 per region (one-shot, two-shot, NVFP4-input regions are separate, so BF16-input and NVFP4-input calls can alternate). Call c uses buffer `c % 3`.
- Reset target: during call c each CTA re-arms its share of buffer `(c+2) % 3` (the one call c-1 used), sized by call c-1's rows and type. One-shot never writes the own slot (own row stays in smem), so the own slot is never re-armed.
- Init: sentinel-fill every buffer, then one NCCL all-reduce so no rank pushes before every rank is armed.

Proof sketch (no stale data is consumed; rank skew is safe):
1. A rank finishes call c only after receiving every peer's call-c data, which peers send inside their own call c. So no peer is more than one call ahead. While I am in call c, nobody writes my buffer `(c+2) % 3`: its next writer is a peer's call c+2.
2. That peer's call c+2 starts only after my call c+1 data reaches it. I send that only after `griddepcontrol.wait` in call c+1, i.e. after call c, including its re-arm stores, completed. (The same kernel-boundary causality chain FlashInfer/TRT-LLM Lamport relies on.)
3. I re-arm a buffer only after my own previous use of it is complete: call c-1 finished, because I am past `griddepcontrol.wait` in call c.
4. Writers never produce a sentinel and checks are per nibble/byte/word, so a sentinel-free word is entirely from the current call.
5. A delayed rank only makes others spin on sentinels in the current buffer. The gate covers this (one rank delayed 50 µs per call, both directions).

Costs: re-arm writes (P-1) x rows x 3472 B per call (57 MB at TP2 16K, ~5 µs of HBM, overlapped). Window size for 3 one-shot + 3 two-shot + 3 NVFP4-input buffers plus an input slot is about 1 GB at TP2 16K and grows linearly with P.

## 6. Push kernel, TP4 two-shot (`L4`, `L4q`)

One-shot never wins at TP4 (each rank would receive 3 full encoded tensors), so TP4 uses two-shot with row ownership. Same CTA shape as `L2` (10 worker warps, warp per row, 1 reset warp, 3-buffer rotation).

| Phase | Work |
| --- | --- |
| A (every row) | encode (or, for `L4q`, TMA-load the 3456 B record), TMA bulk-store the whole row to the owner's reduce-scatter slot. Owner = `row % 4`, slot index = `row / 4` (self included) |
| B (owned rows) | poll and decode-sum the 4 sources in FP32 (rank order), row amax, re-encode, output = decode of the re-encoding; TMA bulk-store the re-encoded row to the 3 peers' all-gather slot `[owner]` |
| C (other rows) | poll and decode the all-gather row (one source) |

- Each warp runs A, then B, then C, so there is no cross-rank wait cycle.
- Egress per rank = 2 x 3/4 x rows x 3460 B = 1.5 x rows x 3460 B.
- Owner two-warp teams (single pass, FP32 sum in registers, amax through smem and a 64-thread barrier) when rows/CTA <= 40; B/C processed by quad when rows/CTA > 64.
- `L4q` shares the receive buffers; no row scale on the A-phase wire (`rs = gs[src]`); the owner still re-encodes, so the all-gather carries a row scale.

TP2 two-shot exists in the code (same ownership rule, `row % 2`) but loses to one-shot at every tier, so it is not dispatched.

## 7. Fused residual + RMSNorm epilogue (TP2)

Built on the TP2 push kernel. M3 uses Gemma RMSNorm: `eps = 1e-6`, `y = x * rinv * (1 + w)` (vLLM `MiniMAXGemmaRMSNorm` -> FlashInfer `gemma_fused_add_rmsnorm`, `weight_bias = 1.0`).

Per row, lane l owning pairs `l + 32j` (j = 0..5); the whole 6144-wide row is in one warp after the reduction:
1. `acc` = FP32 rank-order decode-sum (codec v2).
2. `z = acc + float(res_in)`; `res_out = bf16_rn(z)` (residual updated in place).
3. `ss = sum(z*z)` in FP32 (two partial sums, even/odd elements), xor butterfly.
4. `rinv = rsqrt.approx.ftz(ss / 6144 + eps)` (as FlashInfer computes it).
5. `y = (float(res_out) * rinv) * (wbias + float(w))`.
6. Output: BF16 (`bf16_rn(y)`), NVFP4 (vLLM `scaled_fp4_quant` arithmetic on `bf16_rn(y)`), or MXFP8 (E4M3 + UE8M0 per 32, vLLM torch-reference semantics with an exact bit-level `ceil(log2(amax/448))`).

Implementation:
- The residual row is TMA-loaded into the (now dead) x buffer as soon as the row is encoded, i.e. during the transport wait. The next x row is loaded after the previous row's epilogue.
- Norm weight row lives in smem; pass 2 runs in XOR chunk order so staging needs no un-swizzle selects.
- 10 worker warps for BF16 input; the NVFP4-input path still uses 6 workers with a separate residual buffer.

Deviations (known, not yet production-ready):
- Step 5 multiplies the BF16-rounded `z`, not FP32 `z` (holding FP32 `z` would cost 192 registers per lane). Effect: at most one extra rounding of 2^-9 relative before the final BF16 rounding. `ss` uses exact FP32 `z`.
- NVFP4 and MXFP8 outputs use linear scale layouts; vLLM's NVFP4 default is the swizzled 128x4 layout. Neither was bit-compared to FlashInfer's production kernels.
- The fused-norm kernel is based on an earlier push build (no two-warp teams, older call-state code): its AR-only path is 67.2 µs at 16K vs 60.7 for the final push kernel.

## 8. Pull kernels and why they lose on VR200

Design (`P`, BF16 input): every write goes to the writer's own window; peers only read.
- 640 threads per CTA: a fetch warp (lane s TMA-pulls source s of a stage of k rows into a 4-slot smem ring), a publisher warp, 8 encoder warps, 8 decode-reduce warps.
- Records are 3472 B; a stage is k consecutive rows, one `cp.async.bulk` per source.
- Readiness: decoders validate sentinels in smem and re-pull invalid sources (bounded, with backoff). A weak (`st.relaxed.sys`, unfenced) per-CTA progress tag is a latency hint only; correctness is the smem check.
- 4 buffers; call N refills the buffer of call N-2 (= N+2), then one `fence.acq_rel.sys` per call, off the critical path.
- Proof: (a) no peer still reads buffer N-2 when I refill it in call N: in call N-1 I validated every peer's N-1 data, which each peer wrote only after its N-2 kernel (including all pulls of my N-2 data) ended. (b) the refill is visible before any peer reads it for call N+2: that peer must first validate my N+1 data, written after my call N (refill + sys fence) completed. Assumption, as for every Lamport scheme: real-time order after a completed sys fence holds for NVLink readers.

NVFP4-input pull (`Pq`, window placement): the GEMM writes directly into a symmetric-window slot from `input_slot()`, so the kernel moves no own data and peers pull it at kernel start.
- 5 slots rotated on the host; each call refills the slot of the previous call (recorded per device-call parity).
- **Slot precondition:** the slot sequence is identical on all ranks, and two consecutive AR calls never use the same slot (reuse distance >= 2). With period-5 rotation this breaks only at a CUDA-graph replay boundary when the AR calls per graph `K ≡ 1 (mod 5)`; an integration must then change the period or insert a dummy `input_slot` call. The kernel detects `prev == cur` but cannot make it safe.
- Producer contract: the GEMM must not write its slot before its own `griddepcontrol.wait` (i.e. before the previous AR kernel completed).

TP4 pull: `P4` (pull two-shot, single-pass owner) and `P4t` (pull reduce-scatter + TMA-store push all-gather; wins over `P4` at <= 4K).

Why pull loses on VR200 (it matched or beat push on GB200):

| Factor | Push (TMA bulk store) | Pull (TMA bulk load) |
| --- | --: | --: |
| One-way handoff latency | 1.57 µs (`st.v4`), 1.62 µs (TMA store) | 5.84 µs (poll + TMA round trip) |
| a2a egress roof, TP2, 64 MB/peer | 1048-1094 GB/s | 890-925 GB/s |
| a2a roof, TP4 | 1179 GB/s | 982 GB/s (83%) |
| Wire cost | 22 B header per 256 B packet | 22 B header per 128 B response plus a 22 B request on the reverse link; in a2a every link carries both, payload ceiling ~1015 GB/s |

Result: pull is 1.1-1.5x slower than push at every tier at TP2 and TP4 (see [RESULTS.md](RESULTS.md)). At 1K-2K the extra traversal is a floor push does not have; at 16K the a2a bandwidth ceiling is ~15% lower.

## 9. Rejected designs (measured)

| Design | Result |
| --- | --- |
| Per-tile `st.release.sys` flags (round 1) | 12-21 µs per release under NVLink load; corrected overlap 0-0.35 |
| Warp-specialized push (loader / sender / encoder warpgroups / decoders, `setmaxnreg`) | encoder-bound, 97 µs at 16K; a second encoder warpgroup did not help (register cap at 640 threads) |
| TMA `report::validity` receive (`LT`) | correct (0 false-valid in 290-364k retries), but 2-4x slower: one fetch warp serializes issue and re-issue (0.35 µs per local re-issue); `per_16bytes::*` checks only the first element of each 16 B; `mbarrier.init.layout::v0` faults, `v1` works |
| Software-pipelined decode loads | 86 µs at 16K: early loads are stale under Lamport and must be re-polled |
| Pull with a per-stage `st.release.sys` flag | 117.8 µs at 16K vs 89.1 µs for the unsafe gpu-scope bound |
| Two-shot at TP2 | loses to one-shot at every tier |
| One-shot at TP4 | 1.1-1.8x slower than two-shot at every tier |
| Quad-pipelined / warp-specialized TP4 two-shot | slower at every tier |
| NVLS `multimem.st` broadcast | 175 GB/s a2a payload per GPU on GB200; NVLS AR not faster than Ring at TP4 on VR200 |
