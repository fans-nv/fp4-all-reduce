# History

Two rounds on VR200, 2026-10-12 and 2026-10-13, with GB200 for bring-up. All µs are per call, median of max over ranks.

## Round 1 (2026-10-12): baselines and a first fused kernel

Question: can NVFP4 transport (3.56x fewer bytes than BF16) speed up M3's TP AllReduce at prefill tiers?

| Finding | Evidence |
| --- | --- |
| NCCL tuning is free and comes first | Ring + 64 channels, LL128 for small / Simple for large: TP4 1.36-2.17x over production up to 8K rows; TP2 1.16-1.29x at 1K-2K only. NCCL's own cost model picks LL128/Simple backwards on VR200 |
| AllReduce is ~12% of the M3 TP2 16K prefill step, unoverlapped | 121 AR calls per step, 29.1 ms of ~240 ms; step time minus AR constant across NCCL variants |
| vLLM leaves time on the table at 4K | picks FlashInfer two-shot (123 µs) although one-shot (90 µs) is faster |
| Unfused NVFP4 composites lose | existing encode + NCCL all-gather/all-to-all + decode kernels: 0.13-0.82x vs tuned BF16 (decoder ~9x off roofline) |
| NCCL native FP8 AR is fast but the unfused scale/quant/dequant eats the gain | FP8 AR 1.27-1.73x; composite 0.53-0.99x |
| NCCL all-gather + fused Triton decode: first net gain | 0.95-1.18x |
| Fused CUDA kernel on the NCCL 2.31 device API: best round-1 result | eager 1.31-1.79x (TP2), 1.22-1.81x (TP4); graph 21.1 / 42.4 / 128.8 µs at TP2 1K / 4K / 16K |
| Fused Triton kernel over torch symmetric memory | TP2 1.13-1.28x; TP4 loses (Triton SM stores push only ~430-500 GB/s) |
| Synthetic accuracy | rel-L2 0.095 one-shot, 0.133-0.135 two-shot, vs BF16 0.0018 |

Round-1 fused kernel design: one warp per row, 8-row tiles, encoded data stored directly into peer windows, per-(source, tile) `st.release.sys` flags with `ld.acquire.sys` spins and `__syncthreads`. Its corrected overlap of transport and compute was only ~0-0.35: sync, not bytes, was the limiter (FlashInfer's BF16 Lamport AR+norm at 1K was 22 µs vs our NVFP4 25.7 µs with 3.56x fewer bytes).

Three source studies (FlashInfer/TRT-LLM AllReduce fusion, DeepGEMM Mega MoE, NCCL device API, QuickReduce, Triton-distributed, ThunderKittens, Flux, CUTLASS) converged on: **put readiness in the data (Lamport sentinels), drop every system-scope release**. NVFP4 already has free redundant encodings for sentinels (E2M1 -0 and scale byte `0xFF`).

## Round 2 (2026-10-12 night to 2026-10-13): Lamport push and pull

Plan: microbenchmark first (NVLink push / TMA store / TMA pull bandwidth, 16 B tear test, TMA `report::validity`, fence costs), then build three candidate kernels on a common harness with a bitwise gate: push + sentinels, push + TMA-validity receive, and TMA bulk pull.

Micro results that set the design:
- sys-scope fence: 12-21 µs under NVLink load; a per-tile release cuts throughput 5x. -> no fences on the hot path.
- Lamport push handoff 1.57 µs vs pull 5.84 µs. -> push at small tiers.
- TMA bulk store uses 256 B packets (x1.086 header factor) vs 128 B for `st.v4` (x1.172). -> TMA-store egress.
- 0 torn 16 B packets over NVLink in >1e12 reads; TMA validity checks only the first element per 16 B.

Push kernel progression (TP2, graph, µs at 16K / 1K, BF16 input): warp-specialized Lamport push 97.2 / 13.3 -> warp-per-row on the fast codec 73.3 / 13.6 -> TMA-store egress 68.2 -> two-warp row teams 62.2 / 11.8 -> per-CTA call state 60.7 / 10.6. NVFP4 input (GEMM-epilogue-quantized partials, added mid-round): 55.4 / 8.6.

Pull kernel progression (TP2, graph, µs at 16K): flag protocol 117.8 (eager) -> Lamport + weak progress-tag hint 92.4 (eager) -> fast encoder 79.0 -> tuned rows-per-stage 74.1. It beat the then-current push kernel at 4K-16K for a few hours, then lost at every tier once push moved to warp-per-row with TMA-store egress.

TP4: push two-shot with row ownership (`row % 4`) 19.8 / 103.8 µs at 1K / 16K; pull two-shot and a pull/push hybrid lost by 1.1-1.3x.

Fused residual + Gemma RMSNorm epilogue (TP2): first version 182.9 µs at 16K (per-warp serial residual-load latency) -> residual TMA-loaded into the dead x buffer during the transport wait, weights in smem, XOR-ordered pass 2 -> 113.4 µs, vs 311.4 for vLLM's path.

## Bugs found

| Bug | Symptom | Root cause | Fix / guard |
| --- | --- | --- | --- |
| `.b8` E2M1 pack miscompile (nvcc 13.0, sm_100a) | two-shot reference rel-L2 0.375 instead of 0.133 | `mov.b32 %0, {b0,b1,b2,b3}` from `.b8` registers: when an encode result fed a decode in registers, ptxas forwarded the pack with the wrong byte select, zeroing elements 2-3 of every 8 | codec crosses the asm boundary with 16-bit values; CPU E2M1 oracle test (`codec_test`); never reintroduce `.b8` vector moves |
| `setmaxnreg.inc` deadlock | warp-specialized push hung in the gate | role register sum (40x128 + 96x256 + 136x256) exceeded the launch allocation (640 threads x 96) | decoders inc to 120; rule: role sum must fit threads x launch registers |
| Pull window race, 2 slots | none observed; gate did not catch it | with 2 slots, slot N+1 = N-1 is refilled during AR N, but a peer can enter AR N+1 (needs only my GEMM N data) and read my stale N-1 data | 5 slots, refill previous call's slot; excluded by proof, not by the gate (a deterministic delayed-refill test is not built) |
| Pull window race, intra-grid "last slot" | 1K bad calls; hang in local mode | a single "last slot" word that CTA 0 of the same grid overwrote | previous slot recorded per device-call parity (read `(N-1)&1`, write `N&1`) |
| Invalid NVFP4-input push build | every cell failed at ~3.3 µs | a patch moved `p.R / p.me / p.flags = ...` into a trailing comment in the launch path (R = 0) | found with device printf; rebuilt and re-gated; those runs discarded |
| Silent unknown variant | `--variant P4q` on a binary without it timed nothing, printed `done fails=0`, exit 0 | name lookup fell through | `fp4ar_bench` exits 2 on an unmatched name ("unknown variant ... see --list") |
| nsys flush trap | capture had CUDA events but no kernel table; nsys exit 0 | harness `_exit()`s (NCCL teardown hang) before CUPTI buffers flush; `--cuda-flush-interval` does not help | `cudaProfilerStop()` on every device before `_exit`; always assert the kernel table is non-empty |
| Decode load pipelining | 86 µs vs 67.5 at 16K | early-issued loads are mostly stale under Lamport and are re-polled one round trip per word | reverted; re-poll all invalid words of a group in parallel instead |
| "No reset" ablation looked 3.4 µs faster | misleading ablation | without re-arm, stale data looked valid, so there was no Lamport wait | ablations that remove sentinels are not timing evidence |

## What is next

See [../README.md](../README.md) "Status and limitations" and [PROFILING.md](PROFILING.md) section 5. In priority order: accuracy on real M3 partials; vLLM integration ([INTEGRATION.md](INTEGRATION.md)); one kernel per input format and smem-staged receive decode; fused norm rebased on the final push kernel and extended to TP4; TP8 across two nodes.
