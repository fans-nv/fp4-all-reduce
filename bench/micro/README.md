**Round-2 micro lane: NVLink transport, tear, TMA validity, and fence microbenchmarks**

Five self-contained CUDA programs. Each one uses a host thread per GPU and NCCL 2.31.2 symmetric-window peers (`ncclMemAlloc` + `ncclCommWindowRegister(NCCL_WIN_COLL_SYMMETRIC)` + `ncclGetLsaDevicePointer`), plus an NVLS multicast pointer via `ncclDevCommCreate(lsaMultimem)` + `ncclGetLsaMultimemDevicePointer`. Each program sweeps its comma-list arguments in one process and prints one JSON line per configuration. Timing is the median of per-iteration max-over-ranks CUDA-event time; ranks are aligned before every iteration with a host barrier plus a 1-element `ncclAllReduce`.

| File | Answers |
|---|---|
| `micro_common.cuh` | Host harness (args, NCCL window/peers/multicast, alignment, JSON) and PTX helpers |
| `bw.cu` | (1) Peer bandwidth for push `st.global.v4`, TMA bulk store smem→peer, TMA bulk pull peer→smem (`shared::cta` and `shared::cluster`), per-thread pull `ld.relaxed.sys.v4`, and `multimem.st.v4` (TP4). Swept over chunk 1-32 KB, bytes per peer 0.5-64 MB, 1 or 2 CTAs/SM, and an SM subset. Patterns: `uni` (0→1), `fanout` (0→all), `a2a` (all→all at once, the AllReduce pattern), `self` (local HBM). |
| `lat.cu` | (1) Small-transfer latency: ping-pong between ranks 0 and `--peer`. Flags: `st.release.sys`/`ld.acquire.sys`, weak `st`/`ld.volatile`, relaxed.sys. Payload protocols: Lamport `st.v4` + `ld.volatile.v4`, Lamport via TMA store, and pull (local stores + release flag → acquire + TMA pull), at 16 B-64 KB. |
| `tear.cu` | (2) 16 B / 8 B store atomicity over NVLink. A remote writer stores same-generation packets (`st.v4`, TMA bulk store, `st.v2`) while local readers poll (`ld.volatile.v4`/`.v2`, TMA bulk load) and count torn packets. Two positive controls must tear: `scalar4` (4 separate `st.u32`) and `split` (torn by construction). This proves the detector works. |
| `validity.cu` | (3) Rubin TMA `report::validity`. **sm_107a only**: it compiles to a stub elsewhere and reports `rubin_path=0`. Two parts: `sem` (semantics) and `xgpu` (stress). |
| `fence.cu` | (4) Cost of `st.release.sys` / `fence.acq_rel.{sys,gpu}` / `red.release.sys.add` / `st.release.gpu` after N outstanding remote or local 16 B stores, with and without background NVLink traffic. Also the throughput of a persistent push loop under each per-tile publication variant (none, bar, the round-1 bar+release, red, per-warp release, local gpu release). |
| `build.sh` | Container: builds all five programs with `ARCH`/`NVCC`/`NCCL_HOME` from `scripts/cluster.env` into `$WORK/build/micro`. |
| `run.sh` / `step.sh` | Login node: `NODE=<node> run.sh [step ...]` runs each program with explicit arguments as one `scripts/step.sh` step; output goes to `$WORK/runs/micro/<step>.jsonl`. About 20 min in total. |
| `summarize.py` | Builds markdown tables from a runs directory. |

**`validity.cu` part details.**
- `sem`: covers all 5 mechanisms (`per_16bytes::{80000000,8000,80,8}`, `per_element::ff`), both destination forms, and local and peer sources. It loads 1 KB in which one 16 B chunk contains:
  - one sentinel element first or last;
  - a chunk full of sentinels;
  - the whole buffer as sentinels;
  - a near-miss value;
  - for the nibble mechanism, `0x81` and `0x18` bytes.

  For each case it records primary-phase completion, `isReportSeen`, the conditional-phase parity bits, and load latency. It then re-issues a valid load on the same barrier, which tests the retry protocol.
- `xgpu`: rank 0 pushes encoded data with `st.v4` or a TMA store into rank 1's sentinel-filled buffer, after random jitter. Rank 1 TMA-loads each 4 KB chunk with validity, retries on report, and waits on the conditional phase. It checks every word, which yields `false_valid` (must be 0), then re-arms the buffer. Default is 20k iterations x 16 chunks.

**PTX used (PTX ISA 9.4).** This compiles with pip nvcc 13.4 for `sm_107a`, and `ptxas` rejects it for `sm_100a`.

```
mbarrier.init.layout::v1.shared.b64 [bar], 1;                                    // Payload64 barrier (PipelinePayloadLamport)
mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [bar], bytes;
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_16bytes::8 [smem], [gmem], bytes, [bar];
mbarrier.test_wait.parity.phase_type::primary.acquire.cta.shared::cta.b64 done|isReportSeen, [bar], parity;   // loaded? invalid?
mbarrier.try_wait.parity.phase_type::conditional.acquire.cta.shared::cta.b64 p, [bar], parity;               // consumers: valid data
```

The retry protocol, inferred from `CutlassPipeline.h` `PipelinePayloadLamport` and `Task.cpp` `generateProdTestWaitValid`:
1. A loader warp issues the load and test-waits the primary phase.
2. If the primary phase is done but `isReportSeen` is set, the loader re-issues the load (new `expect_tx`; the primary parity flips per attempt).
3. If the primary phase is done and the data is valid, the loader moves on.
4. Consumer warps only `try_wait` the conditional phase, which flips once per valid load.

`validity.cu` checks this protocol rather than assuming it.

**Decisions each result drives (kernels L = push + Lamport sentinel, LT = push + TMA validity, P = TMA pull).**

| Result | Decision | GB200 value | VR200 expectation |
|---|---|---|---|
| bw: push vs TMA store vs TMA pull peak, a2a | Transport per variant. If pull ≥ push, P's extra bytes in flight are free. If TMA store ≈ st.v4, LT's comm warp loses nothing | push = TMA store 652, pull 708 (uni). a2a: 648 vs 612 | Link 1476 GB/s/dir. At GB200's 72-79 % efficiency, expect about 1.05-1.2 TB/s |
| bw: SM count to saturate | Number of comm warps/CTAs; how many SMs stay with the codec | Push: 16 SMs. TMA pull: 16 SMs at ≥ 4 KB chunks | About 1.6x GB200's per-SM need (link is 1.64x, 212 SMs): about 24-32 SMs |
| bw: chunk size | P's TMA chunk / ring stage size, and L's packet granularity | Pull needs ≥ 4 KB (4 warps x 4 stages). 1 KB needs 64+ SMs | Same shape |
| bw inner16 at 0.5-4 MB | Whether small tiers (1K-2K rows) can approach peak | 512 KB per peer: 390-400 GB/s | |
| lat | One-shot small-size floor; L vs P crossover | Lamport 1.34-1.68 µs one-way up to 16 KB. Pull 4.2-4.75 µs. Release flag 2.16 µs vs relaxed 1.36 µs | Similar or lower |
| tear | **Gate for LT.** Any torn 16 B in st.v4 → TMA load makes per-16 B hardware validity unsafe across GPUs. LT then falls back to the software per-nibble check (L) | 0 torn in about 6e11 reads per pair. Controls tear | Must be 0. If not 0, LT is dropped |
| validity sem | Which mechanism to use for payload (`per_16bytes::8`) and scales (`per_element::ff`); whether one sentinel element anywhere in 16 B reports (needed if writes can tear); re-issue latency | n/a (Rubin-only) | report=1 for one_elem_first/last, chunk_all, all_sentinel. report=0 for all_valid and near_miss. Re-issue latency about 1-2 µs local |
| validity xgpu | LT correctness under real remote push | n/a | **false_valid must be 0.** Retries > 0 confirms the stress actually raced |
| fence | Whether any sys-scope fence can sit on the critical path | Per-tile release: +2-8 µs per tile, 644 → 189 GB/s. sys fence under load: 9-20 µs | Same qualitative result. Kernels publish without fences (Lamport) or with one release per call off the critical path |
| multimem | TP4 two-shot broadcast via NVLS | a2a payload 175 GB/s vs push 680 egress: not worth it | Re-measure: NVLS on VR200 may differ |

**Running.** From the login node (cluster profile in `scripts/cluster.env`):

```
scripts/step.sh <node> - 600 $SRC/bench/micro/build.sh
NODE=<node> bench/micro/run.sh [step ...]
python3 bench/micro/summarize.py $WORK/runs/micro > summary.md
```

- `run.sh` runs the priority steps first: `bw_tp2 bw_tp4 lat_tp2 fence_tp2 tear_tp2 validity_sem validity_xgpu ...`.
- `validity_sem_v0` (non-v1 barrier layout) is last and isolated, because it may fault.
- Pass step names to run a subset.
- Set `NSM` to change the SM subset list. The default `16,32,64,106,0` uses 0 to mean all SMs.

**Implementation notes.**
- **SM subset.** Dynamic smem pins residency to exactly `cpsm` CTAs/SM. CTAs whose `%smid >= nsm` exit, and a ticket gives dense work ids. `active_ctas_seen ≥ expected` confirms the subset was populated.
- **Pull loads.** Per-thread pull uses `ld.relaxed.sys.v4`. Weak `ld.global` was served from L1 on repeated passes and reported impossible rates (> NVLink) under `--inner`.
- **Generations.** Lamport generations in `lat` are monotonic per configuration, so no buffer reset is needed between iterations.
- **Hangs.** Every spin has a 2 s globaltimer timeout, and timeouts are counted in the JSON rather than hanging.
- **Offline check.** All five programs pass `scripts/check_compile.sh bench/micro/<file> -Ibench/micro -diag-suppress 821` (nvcc 13.4, sm_107a), with 0 spills. `validity.cu` also passes `--sm100` as a stub.
