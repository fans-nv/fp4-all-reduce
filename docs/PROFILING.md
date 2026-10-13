# Profiling and roofline

Nsight Compute analysis on VR200 (GR100, sm_107a, 212 SMs, HBM locked at 4752 MHz), TP2, rank 0. Use ncu for counters and stalls, the benchmark harness for time.

## 1. Bottom line

- **No kernel is HBM-bound or issue-bound.** Every kernel runs at 54-62% of the 13.2 TB/s copy roof and 27-56% issue busy. The binding resource is latency:
  - 16K: load-to-use chains in the decoder (push) or encoder (pull), with too few warps to hide them (push: 2.75 warps per scheduler).
  - 1K-4K: instruction fetch (`no_instruction` stalls 12% at 16K -> 33% at 4K -> 52% at 1K in `L2q`) plus per-launch synchronization.
- **The final push build has an instruction-fetch regression.** Compiling the NVFP4-input path into the same kernel doubled SASS from 90 KB to 171 KB. On the identical BF16 path at 4K, `no_instruction` went from 19.3% to 46.6% and time per pass from 22.5 to 27.0 µs.
- The fused-mode overlap gap (~12 µs exposed at 16K) cannot be profiled with ncu or nsys in this harness (section 6).

## 2. Peaks

| Peak | VR200 |
| --- | --- |
| HBM copy roof (cudaMemcpy D2D) | 13.2 TB/s large; 11.66 at 16K rows, 10.1 at 8K, 8.2 at 4K, 6.5 at 2K, 4.0 at 1K |
| HBM theoretical (ncu) | 4032 B/cycle x 4.749 GHz = 19.15 TB/s |
| Issue | 212 SMs x 4 schedulers x ~2.40 GHz = 2.04e12 warp-instructions/s |
| NVLink egress (micro) | TMA store 1048, `st.v4` 957, TMA pull 890 GB/s; wire ceiling ~1300 per direction ([NVLINK.md](NVLINK.md)) |

## 3. Floors (TP2 one-shot, H = 6144)

Bytes per rank per row: wire 3460; HBM ~2 x 12288 (read x, write out) + 2 x 3460 (encoded write and read) = 31,496 B.

| Rows | Wire MB | Push transport floor (1048 GB/s) | Pull transport floor (890 GB/s) | HBM MB | Compute floor at 11.66 TB/s (at measured size-specific roof) | Push ideal fused | Pull ideal fused |
| --: | --: | --: | --: | --: | --- | --: | --: |
| 1K | 3.5 | 3.4 µs | 4.0 µs | 32 | 2.8 µs (~8) | ~6-9 µs | ~10-13 µs |
| 2K | 7.1 | 6.8 | 8.0 | 65 | 5.5 (~10) | ~10-12 | ~14-16 |
| 4K | 14.2 | 13.5 | 15.9 | 129 | 11.1 (~16) | ~17-19 | ~22-24 |
| 8K | 28.3 | 27.0 | 31.8 | 258 | 22.1 (~26) | ~29-31 | ~38-40 |
| 16K | 56.7 | 54.1 | 63.7 | 516 | 44.3 | ~56-58 | ~66-70 |

Ideal fused = max(transport, compute) + fill/drain + handoff latency (push 1.57 µs, pull 5.84 µs) + launch (~1.2 µs). NVFP4 input drops the HBM term to ~12288 + 2 x 3460 B per row (~27 µs at 16K).

Time budget of the push kernel (graph µs):

| Component | 1K `L2q` | 16K `L2q` | 16K `L2` |
| --- | --: | --: | --: |
| Graph launch, empty kernel | 0.94 | 0.94 | 0.94 |
| Kernel shell (prologue, exit) | ~1.7 | ~1.7 | ~1.7 |
| Compute alone | 3.9 | 37.8 | 54.2 |
| Transport alone | 7.2 | 52.5 | 55.7 |
| Fused | 8.5 | 55.4 | 60.7 |
| Overlap loss = fused - max(xport, compute) | 1.3 | 2.9 | 5.0 |

## 4. Compute-mode roofline (ncu application replay)

Bytes are DRAM at 16K and ideal HBM bytes below 16K (L2 absorbs small outputs, so DRAM write counters undercount there).

| Kernel | Rows | ncu per pass (µs) | Harness (µs) | Bytes | Achieved | Memory roof (µs) | Issue roof (µs) | Issue busy | Warps/sched | Top stall | Binding |
| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --- | --- |
| Push, warp-specialized (superseded) | 16K | 89.1 | 84.7 | 486 MB | 5.45 TB/s (41%) | 36.8 | 34.3 | 38.6% | 4.0 | long_scoreboard 41% | decoder header-load chain; half-sector stores |
| Push `L2` (pre-final build) | 16K | 58.8 | 53.1 | 461 MB | 7.84 (59%) | 34.9 | 22.6 | 41.9% | 2.75 | long_scoreboard 31% | decode load-to-use; tail 11% |
| Push `L2q` (final) | 16K | 43.8 | - | 315 MB | 7.19 (54%) | 23.9 | 13.8 | 36.6% | 2.75 | long_scoreboard 35% (31% on one LOP3) | received-row `LDG.STRONG.SYS` loads |
| Pull `P` | 16K | 66.3 | 63.7 | 467 MB | 7.04 (53%) | 35.4 | 27.2 | 47.0% | 5.0 | long_scoreboard 39% | encoder loads; consumers wait on producer |
| Push `L2` (pre-final) | 4K | 22.5 | 17.7 | 129 MB | - | 15.7 | 5.9 | 32.6% | 2.75 | barrier 18%, no_inst 19% | sync and first-row fill |
| Push `L2` (final) | 4K | 27.0 | - | 129 MB | - | 15.7 | 6.2 | 28.0% | 2.75 | no_inst 47% | instruction fetch |
| Push `L2q` (final) | 4K | 17.7 | - | 79 MB | - | 9.6 | 3.7 | 26.7% | 2.75 | no_inst 33% | instruction fetch |
| Push `L2q` (final) | 1K | 11.6 | - | 20 MB | - | 4.9 | 1.1 | 13.6% | 2.75 | no_inst 52%, barrier 18% | instruction fetch, prologue sync |
| Pull `P` | 4K | 20.6 | 17.1 | 129 MB | - | 15.7 | 7.3 | 44.8% | 5.0 | long_scoreboard 23%, barrier 15% | encoder latency, exit sync |
| Codec encode (TMA ring) | 16K | 28.6 | 30.3 | 236 MB | 8.23 (62%) | 17.8 | 11.7 | 56.4% | - | mbarrier wait 38% | memory pipeline |
| Codec decode-sum (global, staged) | 16K | 31.9 | 31.8 | 258 MB | 8.09 (61%) | 19.6 | 11.6 | 47.7% | - | long_scoreboard 63% | memory pipeline |
| Codec one-shot compute (TMA) | 16K | 58.0 | 62.7 | 462 MB | 7.97 (60%) | 35.0 | 23.5 | 50.0% | - | mbarrier wait 31% | memory pipeline |

At 16K the push compute paths are ~1.7x off the memory roof; standalone codec blocks sit at ~60% of 13.2 TB/s (~70% of the 11.66 practical roof), so fused compute could reach ~44 µs at 16K (BF16 input) without new codec work.

Instruction-fetch evidence:

| Build | Rows | SASS size (executed) | `no_instruction` share of stalls |
| --- | --: | --- | --: |
| `L2` pre-final (BF16 path only) | 16K / 4K | 90 KB (46 KB) | 11.0% / 19.3% |
| `L2` final (BF16 + NVFP4 paths in one kernel) | 4K | 171 KB (37 KB) | 46.6% |
| `L2q` final | 16K / 4K / 1K | 171 KB (21 KB) | 12.4% / 32.8% / 51.9% |
| `P` (role-specialized warps) | 4K | 45 KB (36 KB) | 5.2% |

The cost tracks total kernel size and the warp-per-row design (every warp walks the whole pipeline out of phase with the others). Top `no_instruction` samples sit right after branches.

## 5. Ranked follow-ups

Push kernel (`L2`, `L2q`):
1. **One kernel per input format (1K-4K, top small-tier item).** Make the input format a template parameter, have the host choose the kernel, roll the decode group loop. Evidence: 46.6% vs 19.3% `no_instruction`, 27.0 vs 22.5 µs per pass on the same path.
2. **TMA the received row into smem before decoding (16K).** `L2q` 16K: 31% of all samples on one `LOP3` in the E2M1 unpack, fed by `LDG.E.128.STRONG.SYS` from the receive slot. Bulk-copy the row into smem (cp.async.bulk + mbarrier) after the Lamport row-scale check, or prefetch group j+U before unpacking j; merge the u16 scale loads.
3. **Prologue and epilogue sync.** `L2q` 1K: 18% barrier at the prologue `__syncthreads` behind thread 0's call-state read; pass call state as kernel arguments or read it per warp. Exit barrier 10-11% from the 7-vs-8 rows-per-warp split: interleave rows globally.
4. **x prefetch depth and occupancy.** 6.2% (16K) / 11.4% (4K) waiting on the one-row-ahead x TMA; 2.75 warps per scheduler (140-148 regs, 225 KB smem). At small row counts use 2-4 warps per row.
5. **Rebase the fused-norm kernel on the final push kernel** (AR-only part 67.2 vs 60.7 µs at 16K; teams halve the per-row epilogue latency).

Pull kernel (`P`):
1. Encoder load path: 24 `LDG.E.128.CONSTANT` at 64 B per-lane stride give all 6.39M excess sectors (28%); 96-register cap causes 28 LDL/STL per row. Use the TMA-staged smem encoder (28.6 vs ~45 µs in-kernel).
2. Producer-bound consumers: 14% on the stage mbarrier wait.
3. 62.5% theoretical occupancy; decoder `LDS` mio_throttle; batch scale-byte loads.

Codec blocks: 31-38% `mbarrier wait` (encode, one-shot) and 63% long_scoreboard (decode). More bytes in flight per SM (more or larger stages within 222 KB smem) is the remaining lever.

## 6. Tool blind spots on VR200

| Item | Result |
| --- | --- |
| ncu in the VR200 image | none (only nsys 2026.5.2). Installed Nsight Compute 2026.3.1 (arm64 .deb from NVIDIA's public CUDA sbsa repo). Extract inside a compute step: the login node OOM-kills `xz` |
| sm_107 support | yes (CC 10.7, real counters, `--list-chips` includes gr100); no permission errors (`RmProfilingAdminOnly: 0`) |
| Kernel replay, TP2 kernels | fails ("Failed to save memory for replay"): NCCL symmetric / peer-mapped memory cannot be snapshotted |
| Application replay, COMPUTE mode | works (~2.5 min per full report; ~7 min for `L2q` 16K) |
| Application replay, FUSED / XPORT | fails ("Unexpected number of profiled kernels"): the profiled rank's kernel is serialized against a peer it spins on. Only some XPORT byte-counter runs succeeded |
| **ncu NVLink counters** | **blind:** `nvltx__bytes*` / `nvlrx__bytes*` read 0 for an XPORT kernel that pushes ~56.7 MB per call |
| **nsys GPU metrics (`gr10x`)** | **blind:** SMs Active, SM Issue, warps in flight, NVLink TX/RX read exactly 0; DRAM ~0 |
| **nsys CUDA trace flush trap** | the harness ends with `_exit()` (to avoid an NCCL teardown hang), so CUPTI buffers are never flushed: the capture has CUDA events but no `CUPTI_ACTIVITY_KIND_KERNEL` table and nsys still exits 0. `--cuda-flush-interval` does not help. Fix: `cudaProfilerStop()` on every device before `_exit` (now in the harness). Always assert the kernel table is non-empty |
| Line attribution | `--import-source` embeds the file as it is at profile time, `-lineinfo` refers to build time. Snapshot sources with the build before trusting line numbers |
| Working NVLink byte source | `nvidia-smi nvlink -e` (with headers) or `-gt d` (payload) bracketing a run |

Recipe used: `ncu --set full --import-source yes` (2026.3.1 `full` already includes PM sampling, source counters, NVLink tables and roofline charts), rank 0 only (`--devices 0`), kernel filter, `-s 2 -c 1`; bench `--graph 0 --iters 4 --warmup 2`. Analysis with the `ncu_report` Python module plus a SASS-level hotspot script for per-instruction stalls, excess sectors and bank conflicts. One capture per profiling run.

Reports: `$ROOT/prof/reports/*.ncu-repz` (`$ROOT` = the measurement campaign's work directory; reports are not included in this repo).
