# Results (VR200)

All numbers are measurements unless a row or column says "projected".

Hardware: VR200 (GR100, sm_107a, 212 SMs, 4 GPUs per node, NV36 through NVSwitch), two nodes, HBM locked at 4752 MHz (default 3186). Shape: M3 hidden 6144, BF16, 12 KiB per row. A tier is rows per AllReduce (= prefill chunk tokens): 1K/2K/4K/8K/16K rows = 12/24/48/96/192 MiB BF16.

## 1. Measurement protocol

| Item | Protocol |
| --- | --- |
| Statistic | median over 200 calls (after 20 warmup) of the max over ranks of CUDA-event time, µs per call |
| Primary mode | CUDA graph, 10 calls per graph, reported per call (the M3 prefill path is graph-captured). Eager is secondary |
| Rank alignment | host barrier + 1-element AllReduce before every iteration |
| Modes | `fused` (the kernel), `xport` (same bytes and sync, no codec), `compute` (codec only, no peer traffic), `empty` (empty kernel after the same alignment; fixed cost subtracted for the corrected overlap) |
| BF16 baseline | NCCL BF16 AllReduce in the same process, out-of-place, best per tier over Ring 16/32/64/128 channels or NCCL defaults x LL128/Simple x registered (`ncclMemAlloc` + `ncclCommRegister`) or plain buffers |
| Gate (every reported cell) | bitwise equal to a same-codec FP32 reference; ranks bitwise identical; 3 rotating input seeds (stale buffers cannot pass); 1000 back-to-back calls, 1000 per-call compares, 1000 replays of a 10-call graph, 200 skewed calls (one rank delayed 50 µs); live negative controls (`FP4AR_NEGCTL=stale` and `=flip` must FAIL) |
| Clocks | every cell aborts if `clocks.mem != 4752`; none aborted; clocks reset and verified at the end |

Hygiene:
- **Repeatability:** 3 back-to-back graph repeats spread <= 1.9% (mostly < 0.7%). Treat differences above 2% as real. Eager spread is up to 7% at 1K.
- **Eager drift:** the eager fixed cost (`empty` mode) drifted 3.3-7.5 µs across runs and nodes. Compare eager numbers only within one run.
- **Co-tenancy:** quiet re-time (one pair active, all 4 GPUs locked) vs a concurrent neighbour pair: push -2.1% to +1.9%; pull 1.3-2.4% faster when quiet; BF16 within 1.3%.
- **Cross-node:** block measurements repeated on the other node agree within 1.5%.

Column definitions used below:
- **Corrected overlap** = (X + C - F - E) / min(X - E, C - E), with X = xport, C = compute, F = fused, E = `empty` fixed cost. 1 = compute fully hidden, 0 = serial.
- **NVLink GB/s** = rows x 3460 B per rank / XPORT time (payload, one direction). Percentages: of the measured TMA-store a2a roof (1048 GB/s) and of the 256 B-packet payload ceiling (1197 GB/s, [NVLINK.md](NVLINK.md)).
- **% floor** = ideal fused time / measured fused time. Ideal fused (push) = 7.5/11/18/30/57 µs, (pull) = 11.5/15/23/39/68 µs (max of transport and HBM floors plus fill, handoff and launch). Floors assume BF16 input (encode included), so NVFP4-input variants can exceed 100%.

## 2. TP2 AllReduce (one-shot)

Dispatch: push at every tier. `L2` for BF16 input, `L2q` for NVFP4 input. Default knobs (`--tune 0,0,0,0`); no knob beat them by more than 1.5%.

Headline (graph, µs; speedup vs tuned BF16 NCCL):

| Rows | Tuned BF16 NCCL | Round-1 fused | Push `L2` (BF16 in) | Push `L2q` (NVFP4 in) | Pull `P` (BF16 in) | Pull `Pq` (NVFP4 in, window) |
| --: | --: | --: | --: | --: | --: | --: |
| 1K | 30.9 | 20.9 | **10.8 (2.86x)** | **8.6 (3.59x)** | 16.1 | 9.9 |
| 2K | 50.5 | 31.0 | **15.2 (3.32x)** | **12.5 (4.04x)** | 20.5 | 12.9 |
| 4K | 77.1 | 43.4 | **21.9 (3.52x)** | **18.4 (4.19x)** | 30.3 | 22.3 |
| 8K | 125.3 | 80.2 | **34.4 (3.64x)** | **30.8 (4.07x)** | 45.7 | 37.1 |
| 16K | 216.6 | 128.3 | **60.7 (3.57x)** | **55.4 (3.91x)** | 74.5 | 65.1 |

Full table (graph µs; "Quiet" = quiet re-time):

| Variant / input | Tune | Rows | Fused | Quiet | XPORT | COMPUTE | Overlap corr. | NVLink GB/s (% of 1048 / 1197) | % floor | vs BF16 |
| --- | --- | --: | --: | --: | --: | --: | --: | --- | --: | --: |
| `L2` / BF16 | default | 1K | 10.8 | 10.7 | 7.7 | 5.8 | 0.32 | 461 (44 / 39) | 69% | 2.86x |
| | | 2K | 15.2 | 15.2 | 11.2 | 9.1 | 0.49 | 633 (60 / 53) | 72% | 3.32x |
| | | 4K | 21.9 | 21.9 | 17.4 | 16.9 | 0.71 | 816 (78 / 68) | 82% | 3.53x |
| | | 8K | 34.4 | 34.5 | 30.5 | 29.1 | 0.86 | 929 (89 / 78) | 87% | 3.64x |
| | | 16K | 60.7 | 61.5 | 55.9 | 54.3 | 0.91 | 1014 (97 / 85) | 94% | 3.57x |
| `L2q` / NVFP4 (local) | default | 1K | 8.6 | 8.8 | 7.4 | 4.0 | 0.59 | 478 (46 / 40) | 87% | 3.58x |
| | | 2K | 12.5 | 12.2 | 10.4 | 6.2 | 0.59 | 681 (65 / 57) | 88% | 4.05x |
| | | 4K | 18.4 | 18.4 | 16.4 | 11.3 | 0.81 | 864 (82 / 72) | 98% | 4.20x |
| | | 8K | 30.8 | 30.8 | 28.8 | 20.7 | 0.90 | 985 (94 / 82) | 97% | 4.06x |
| | | 16K | 55.4 | 55.3 | 52.7 | 38.0 | 0.93 | 1076 (103 / 90) | 103% | 3.91x |
| `L2q` / NVFP4 (window) | default | 1K-16K | 8.7 / 12.2 / 20.0 / 33.9 / 56.7 | | 53.8 @16K | 39.0 @16K | 0.92 @16K | 1054 @16K (101 / 88) | 86-100% | 3.55-4.13x |
| `P` / BF16 | default | 1K | 16.1 | 15.7 | 12.1 | 8.2 | 0.42 | 292 (28 / 24) | 72% | 1.92x |
| | | 2K | 20.5 | 20.1 | 16.2 | 11.1 | 0.56 | 436 (42 / 36) | 73% | 2.46x |
| | | 4K | 30.3 | 29.7 | 23.8 | 17.1 | 0.59 | 596 (57 / 50) | 76% | 2.55x |
| | | 8K | 45.7 | 45.1 | 38.6 | 29.2 | 0.75 | 734 (70 / 61) | 85% | 2.74x |
| | | 16K | 74.5 | 74.0 | 66.1 | 53.9 | 0.84 | 858 (82 / 72) | 91% | 2.91x |
| `Pq` / NVFP4 (window) | `3,2,0,0` | 1K | 9.9 | 9.8 | 9.2 | 6.0 | 0.86 | 384 (37 / 32) | 116% | 3.12x |
| | `4,2,0,0` | 2K | 12.9 | 13.0 | 11.5 | 6.1 | 0.72 | 615 (59 / 51) | 116% | 3.92x |
| | `3,0,0,0` | 4K | 22.3 | 22.2 | 19.1 | 10.8 | 0.67 | 741 (71 / 62) | 103% | 3.46x |
| | `5,2,0,0` | 8K | 37.1 | 37.1 | 35.5 | 19.7 | 0.92 | 798 (76 / 67) | 105% | 3.38x |
| | default | 16K | 65.1 | 64.8 | 61.4 | 30.2 | 0.87 | 923 (88 / 77) | 104% | 3.33x |
| `Pq` / NVFP4 (local), tuned | per tier | 1K-16K | 15.8 / 22.2 / 30.9 / 46.1 / 72.3 | | 68.4 @16K | 44.2 @16K | 0.91 @16K | 829 @16K | 67-94% | 1.95-3.00x |
| Round-1 fused / BF16 | default | 1K-16K | 20.9 / 31.0 / 43.4 / 80.2 / 128.3 | | 85.8 @16K | 69.5 @16K | 0.38 @16K | 661 @16K | 35-44% | 1.48-1.78x |

Eager (same pass; compare within this table only):

| Variant | 1K | 2K | 4K | 8K | 16K |
| --- | --: | --: | --: | --: | --: |
| Tuned BF16 NCCL | 35.0 | 54.4 | 82.4 | 131.3 | 222.8 |
| `L2` | 18.1 | 22.4 | 30.2 | 43.9 | 70.6 |
| `L2q` | 15.8 | 19.2 | 25.6 | 38.0 | 62.1 |
| `P` | 23.5 | 28.1 | 37.1 | 52.6 | 81.3 |
| `Pq` window (default) | 16.5 | 20.0 | 27.5 | 42.2 | 70.3 |

What binds where:
- **16K:** link-bound. `L2` XPORT runs at 97% of the TMA-store roof; corrected overlap 0.91. The ~4 µs above max(XPORT, COMPUTE) is pipeline fill and drain.
- **1K-2K:** latency-bound. XPORT is 7.4-11 µs against a 3.4-6.8 µs wire floor; the gap is the Lamport handoff (1.6 µs one way), the kernel shell (~2.6 µs) and launch.
- **Pull at 16K:** 858 GB/s, 96% of the TMA-pull roof (890).

Tuned BF16 NCCL baseline, TP2 (reproduced 3 times within 0.5%):

| Rows | Graph µs | Eager µs | Best config |
| --: | --: | --: | --- |
| 1K | 30.6-30.9 | 34.8 | LL128, Ring 64 ch |
| 2K | 50.5 | 54.1 | LL128, Ring 32-64 ch |
| 4K | 77.1 | 81.8 | Simple, Ring 64 ch, unregistered |
| 8K | 125.1 | 131.1 | Simple, Ring 64 ch, registered (-1.5%) |
| 16K | 216.1-216.6 | 222.3 | Simple, Ring 64 ch, registered (-6.6% vs 232.6 unregistered) |

NCCL defaults are 46-395 µs (1.2-1.8x slower). 128 channels, `NCCL_NTHREADS=512` and `NCCL_BUFFSIZE=8MB` each change results by <= 0.5%.

Knob sweeps (graph fused):

| Variant | Knobs | Configs | Result |
| --- | --- | --: | --- |
| `L2` / `L2q` | grid 212/200/192/160; no-PDL; `st.v4` egress; `st.v4` reset; teams forced/off | 24 per input | all within 1.5% of default (eager: no-PDL and forced teams gain 3-5% at 1K-2K only) |
| `P` | slots 0/3/6 x rows-per-stage 0/2/3/4 | 12 | default optimal except k=2 at 1K (-1.7%) |
| `Pq` window | slots x k | 19 | k=2 at 1K-2K -12% / -9%; 3 slots at 4K -7%; 16K flat |
| `Pq` local | slots x k | 25 | k=5-7 beats default at every tier (-10% to -26%); 1K cell unstable |
| Grid < `sm_count` (all) | | | always worse |

nsys (one capture per run, graph x10, TP2 16K):

| Capture | Fused (profiled) µs | AR kernel duration µs | Consecutive AR kernel overlap | PDL early starts |
| --- | --: | --: | --: | --: |
| `L2` | 61.2 | 67.4 | 7.2 µs | 540 / 600 |
| `P` | 74.2 | 85.2 | 11.6 µs | 162 / 190 |

No launch gaps: each kernel starts 7-12 µs before its predecessor ends, so the steady-state period equals the fused time.

## 3. TP4 AllReduce (two-shot)

Dispatch: push two-shot at every tier (`L4` BF16 input, `L4q` NVFP4 input). One-shot never wins at TP4.

| Rows | Tuned BF16 NCCL | Round-1 fused (best) | Push `L4` (BF16 in) | Push `L4q` (NVFP4 in) | Best pull |
| --: | --: | --: | --: | --: | --: |
| 1K | 40.2 | 30.5 | **19.8 (2.03x)** | **16.4 (2.45x)** | 26.1 (`P4t`) |
| 2K | 60.7 | 48.6 | **25.6 (2.36x)** | **21.2 (2.86x)** | 29.5 (`P4t`) |
| 4K | 102.8 | 77.1 | **34.8 (2.95x)** | **30.7 (3.36x)** | 44.6 (`P4t`) |
| 8K | 177.0 | 112.9 | **56.2 (3.15x)** | **50.5 (3.50x)** | 68.2 (`P4`) |
| 16K | 306.9 | 169.3 | **103.8 (2.96x)** | **98.3 (3.12x)** | 116.3 (`P4`) |

Full table (graph µs; gates 59/59 PASS):

| Variant / input / algorithm | 1K | 2K | 4K | 8K | 16K | XPORT 16K | COMPUTE 16K | Overlap corr. 16K |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| Tuned BF16 (graph) | 40.2 | 60.7 | 102.8 | 177.0 | 306.9 | | | |
| Tuned BF16 (eager) | 46.4 | 66.2 | 106.2 | 183.0 | 312.9 | | | |
| `L4` / BF16 / two-shot | **19.8** | **25.6** | **34.8** | **56.2** | **103.8** | 86.3 | 68.9 | 0.74 |
| `L4` / BF16 / one-shot | 22.1 | 32.7 | 57.0 | 98.8 | 183.9 | 160.4 | 79.8 | 0.70 |
| `L4q` / NVFP4 / two-shot | **16.4** | **21.2** | **30.7** | **50.5** | **98.3** | 84.5 | 50.5 | 0.72 |
| `L4q` / NVFP4 / one-shot | 19.0 | 28.6 | 52.6 | 88.4 | 162.9 | 156.0 | 60.3 | 0.88 |
| `P4t` / BF16 / two-shot | 26.1 | 29.5 | 44.6 | - | - | | | |
| `P4` / BF16 / two-shot | 32.1 | 37.1 | 48.0 | 68.2 | 116.3 | 101.4 | 75.0 | 0.80 |
| `P4` / BF16 / one-shot | 28.3 | 39.8 | 62.8 | 105.4 | 188.0 | 180.2 | 63.1 | 0.87 |
| `P4t` / NVFP4 / two-shot | 23.8 | 28.2 | 43.2 | - | - | | | |
| `P4t` / NVFP4 window / two-shot | 22.4 | 27.7 | 45.0 | - | - | | | |
| `P4` / NVFP4 / two-shot | 29.5 | 35.7 | 45.2 | 65.5 | 109.6 | 107.6 | 70.7 | 0.97 |
| `P4` / NVFP4 window / two-shot | 28.2 | 34.9 | 48.6 | 70.4 | 105.5 | 102.9 | 49.0 | 0.94 |
| Round-1 / BF16 (one-shot <= 4K, two-shot >= 8K) | 30.5 | 48.6 | 77.1 | 112.9 | 169.3 | 131.2 | 71.8 | 0.46 |

Eager (this run only): `L4` two-shot 28.4 / 35.8 / 49.8 / 65.8 / 112.3; `L4q` 23.6 / 27.7 / 37.2 / 55.9 / 102.7.

NVLink at TP4 (two-shot egress = 1.5 x rows x 3460 B):

| Variant | Rows | XPORT µs | Egress GB/s |
| --- | --: | --: | --: |
| `L4` | 16K | 86.3 | 985 (94% of the TP2 TMA-store roof, 82% of 1197) |
| `L4` | 8K | | 936 |
| `L4q` | 16K | 84.5 | 1006 |
| `P4` two-shot | 16K | 101.4 | 838 |

No TP4 floors were derived, so % of floor is not reported for TP4. Quiet re-time (both nodes idle): `L4`, `L4q` and BF16 within 1.2% of the pass.

Tuned BF16 at TP4: LL128 wins up to 4K, Simple at 8K-16K; registration changes results by <= 1-2%.

## 4. The M3 block: AllReduce + residual + RMSNorm, BF16 out, TP2

M3's normed output feeds MXFP8 linears and the routed experts quantize inside the MoE, so the production block is BF16-out (vLLM's AR+norm+NVFP4 fusion never fires for M3).

| Rows | vLLM 0.30 as routed | Best BF16 block | Push AR + vLLM norm (unfused) | **Fused, BF16 in** | **Fused, NVFP4 in** | Fused vs best BF16 |
| --: | --: | --: | --: | --: | --: | --: |
| 1K | 22.6 | 22.6 (FI fused one-shot) | 18.9 | **16.9** | **14.1** | 1.34x / 1.60x |
| 2K | 45.9 | 45.9 (FI one-shot) | 28.4 | **22.9** | **21.8** | 2.00x / 2.11x |
| 4K | 120.2 (FI two-shot) | 89.1 (FI one-shot) | 44.4 | **34.9** | **34.9** | 2.55x / 2.55x |
| 8K | 169.3 | 167.4 (NCCL + norm) | 76.5 | **61.2** | **55.2** | 2.74x / 3.03x |
| 16K | 311.4 | 295.9 (NCCL + norm) | 140.4 | **113.4** | **96.8** | 2.61x / 3.06x |

Units: µs, graph. vLLM routing on sm_107 TP2 (production recipe, see [INTEGRATION.md](INTEGRATION.md)):
- <= 2K rows: FlashInfer trtllm fused AR+norm one-shot.
- 4K: FlashInfer two-shot (one-shot cap is 32 MiB; costs 31 µs vs one-shot).
- >= 8K: past the 64 MiB fusion cap, pynccl Ring/64 + separate `fused_add_rms_norm`.

Fused-norm kernel, all outputs (graph / eager µs):

| Path | 1K | 2K | 4K | 8K | 16K |
| --- | --: | --: | --: | --: | --: |
| AR only (same base build) | 13.9 / 20.0 | 18.1 / 23.6 | 26.8 / 36.4 | 39.8 / 50.1 | 67.2 / 77.7 |
| AR + residual + Gemma RMSNorm, BF16 out | 16.9 / 23.2 | 22.9 / 35.9 | 34.9 / 46.8 | 61.2 / 74.3 | 113.4 / 127.3 |
| same, NVFP4 out (linear scales) | 17.7 / 25.1 | 24.2 / 34.9 | 37.4 / 51.6 | 75.5 / 83.0 | 131.0 / 142.0 |
| same, MXFP8 out (linear scales) | 18.1 / 25.3 | 24.4 / 39.3 | 37.7 / 52.1 | 75.7 / 81.8 | 130.9 / 143.6 |
| NVFP4 input, BF16 out | 14.1 / 19.7 | 21.8 / 27.6 | 34.9 / 41.2 | 55.2 / 62.5 | 96.8 / 104.0 |

Norm marginal over AR-only (BF16 out, graph): +3.0 / +4.8 / +8.1 / +21.4 / +46.2 µs. The standalone vLLM add+RMSNorm is 8.4-79.7 µs (cold), HBM-bound at ~10.1 TB/s. From 4K up the fused kernel is bound by per-warp compute latency (compute mode 104.6 µs vs xport ~57 µs at 16K).

BF16-transport block alternatives (µs, graph):

| Rows | FI fused one-shot | FI fused two-shot | NCCL LL128 + norm | NCCL Simple + norm | Tuned registered NCCL + norm (additive) | FI `mnnvl` one-shot / two-shot |
| --: | --: | --: | --: | --: | --: | --- |
| 1K | 22.6 | 53.7 | 38.2 | 54.9 | 37.3 | 33.9 / 30.4 |
| 2K | 45.9 | 72.7 | 60.8 | 70.3 | 60.0 | 60.4 / 59.1 |
| 4K | 89.1 | 120.2 | 109.1 | 99.2 | 98.4 | 111.4 / 117.6 |
| 8K | 174.0 | 212.0 | 198.3 | 169.3 | 167.4 | 213.1 / 234.7 |
| 16K | 344.5 | 392.7 | 368.7 | 311.4 | 295.9 | 416.5 / 476.7 |

NCCL + norm in one graph equals AR alone + norm alone within 1% at every tier (strictly serial), so the additive row is valid. The FI workspace was sized for 16K rows, so 8K/16K FI cells run beyond vLLM's 64 MiB cap.

## 5. Per-step savings and projected TTFT (M3 TP2 prefill)

Per-step block time = block µs x 121 blocks per step (count from the round-1 production nsys profile). Measured block times, multiplied out:

| Rows per step | vLLM today (ms) | Best BF16 (ms) | Fused, BF16 in (ms) | Fused, NVFP4 in (ms) | Saved vs vLLM (ms) |
| --: | --: | --: | --: | --: | --- |
| 1K | 2.73 | 2.73 | 2.04 | 1.71 | 0.69 / 1.03 |
| 2K | 5.55 | 5.55 | 2.77 | 2.64 | 2.78 / 2.92 |
| 4K | 14.54 | 10.78 | 4.22 | 4.22 | 10.32 / 10.32 |
| 8K | 20.49 | 20.26 | 7.41 | 6.68 | 13.08 / 13.81 |
| 16K | 37.68 | 35.80 | 13.72 | 11.71 | 23.96 / 25.97 |

**Projected TTFT (projection, not measured in serving).** Measured TTFT is the round-1 TP2 production p50 at concurrency 1 (vLLM 0.30, `max-num-batched-tokens 16384`). The projection subtracts the per-step block saving per prefill chunk (64K ISL = 4 chunks of 16K). Assumptions: the block is on the compute stream with no overlap (round 1 showed step time minus AR constant across NCCL variants), 121 blocks per step, no other change; NVFP4-in additionally assumes the GEMM epilogue emits NVFP4 at no extra cost.

| ISL | Measured TTFT (ms) | Projected, fused BF16 in (ms) | Projected, fused NVFP4 in (ms) |
| --: | --: | --: | --: |
| 1K | 65.4 | 64.7 (-1.1%) | 64.4 (-1.6%) |
| 4K | 85.1 | 74.8 (-12.1%) | 74.8 (-12.1%) |
| 16K | 284.0 | 260.0 (-8.4%) | 258.0 (-9.1%) |
| 64K | 1117.6 | 1021.8 (-8.6%) | 1013.7 (-9.3%) |

## 6. Accuracy (synthetic only)

Relative L2 error vs an exact FP32 sum, synthetic randn inputs with rank-dependent scales and outliers:

| Transport | TP2 | TP4 |
| --- | --: | --: |
| BF16 (NCCL) | 0.0018 | 0.0026 |
| FP8 E4M3 (NCCL native, round 1) | 0.039 | 0.049 |
| NVFP4 one-shot | 0.095 | 0.095 |
| NVFP4 two-shot (owner re-encodes the sum) | 0.133 | 0.133 |

No real-activation or task-level accuracy has been measured. See [INTEGRATION.md](INTEGRATION.md) section 8.

## 7. Builds and provenance

Raw data (JSONL per run) is kept in the measurement campaign's work directory (`$ROOT`) and is not included in this repo. Each JSONL record carries tag, node, PCI bus ids, NVML memory clock at start and end, and tune vector.

| Build tag | Binary md5 | Contents | Key sources (md5) |
| --- | --- | --- | --- |
| `L-f1` | 91914d4b | `L2`, `L2q`, round-1 control | `lamport.cu` 891fd77a |
| `q4` | b69bd896 | `P`, `Pq` | `fp4ar_pull.cu` 1cfd0d69, `fp4ar_pullq.cu` ded06a9b, `codec_fast.cuh` a3f51937 |
| `tp4push-f1` | 5d064fab | `L4`, `L4q` | `l4.cu` 3e4a869f |
| `b6` | d482d6b1 | `P4`, `P4t` (`-DAG_PUSH=2`) | `fp4ar_pull4.cu` 60a9a226 |
| `LN-t10` | not recorded | fused residual + RMSNorm (standalone driver `drv_norm`) | `lnorm.cu`, `lnorm_epi.cuh` (md5 not recorded) |
| `tune-h0200` | 8e019ec2 | nsys captures; harness with the profiler-flush fix | `fp4ar_bench.cu` 547fe7e6 |

Toolchain: `/usr/local/cuda-13.5/bin/nvcc -gencode arch=compute_107a,code=sm_107a`, NCCL 2.31.2. The codec oracle passes on nvcc 13.5 sm_107a; SASS contains `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C`.

Block image: vLLM 0.30.0+65c589c9, FlashInfer 0.7.0, torch 2.14.0a0, NCCL 2.31. Block bench: torchrun 2 ranks, each variant captured 10x in one graph, 40 replays with a barrier before each, median per rank, max over ranks.

Raw JSONL: TP2 final pass `$ROOT/harness-r2/runs/dr-*`, quiet `fin-q-*`, TP4 `t4-*` / `t4q-*`, block `$ROOT/block/runs/t*.jsonl`.
