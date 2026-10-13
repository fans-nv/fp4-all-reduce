# NVLink on VR200: where the 1.8 TB/s goes

Measured on two VR200 nodes (HBM 4752 MHz), GPUs 0,1 unless noted. Every row below is a measurement except the spec row, which is labelled.

## 1. Link facts

| Fact | Value | Source |
| --- | --- | --- |
| Links per GPU | 36, all active; every GPU pair `NV36` through the NVSwitch tray | `nvidia-smi nvlink -s`, `nvidia-smi topo -m` |
| Per-link rate | 41 GB/s **per direction** (`nvlink -dr` 40.999) | `nvidia-smi nvlink -s` + proof below |
| Per-GPU link capacity | 36 x 41 = 1476 GB/s per direction (2952 bidirectional) | derived |
| NVLink 6 spec (not measured) | 3.6 TB/s bidirectional = 1800 GB/s per direction (50 GB/s per link per direction) | public spec |
| Fabric | State Completed / Success, Health Healthy, Bandwidth Full, both nodes in one NVLink clique | `nvidia-smi -q` |
| Copy engines | `asyncEngineCount` = 4 | `cudaGetDeviceProperties` |

Per-direction proof: one-way GPU0 -> GPU1 moves 1113 GB/s (SM `st.v4`) and 1072 GB/s (copy engine). If 41 GB/s were the bidirectional total, one direction could carry at most 738 GB/s. Both directions at once carry 1081 GB/s each, so the directions are independent. The fabric reports full bandwidth, so 41 GB/s is the configured link rate, not a degraded link.

## 2. Counter semantics (`nvidia-smi nvlink`)

| Counter | What it counts | Use |
| --- | --- | --- |
| `nvlink -gt d` (data) | payload; equals bytes sent to 0.1% | payload bytes |
| `nvlink -gt r` ("raw incl. protocol overhead") | data + reverse-direction write responses only (0.3-0.5%); **no packet headers** | not useful for overhead |
| `nvlink -e` Tx packets / Tx bytes | includes packet headers | wire bytes and bytes per packet |

ncu and nsys report **zero** NVLink bytes on VR200 (see [PROFILING.md](PROFILING.md)); bracket a run with `nvidia-smi nvlink -e` or `-gt d` instead.

## 3. Packets and headers (counters around 3 s saturating transfers, GPU0 -> GPU1)

| Transfer | Payload GB/s/dir | Payload per packet | Wire bytes per packet | Wire / payload | Wire GB/s/dir | Wire / 1476 |
| --- | --: | --: | --: | --: | --: | --: |
| Copy engine (`cudaMemcpyPeerAsync`), uni | 1044 | 256 B | 278 B | 1.086 | 1134 | 77% |
| Copy engine, bidir | 831 | 256 B | 273 B | 1.087 | 903 | 61% |
| SM `st.v4` push, 212 CTAs, uni | 1110 | 128 B | 150 B | 1.172 | 1301 | 88% |
| SM `st.v4` push, 212 CTAs, bidir | 1081 | 128 B | 148 B | 1.175 | 1270 | 86% |
| SM `st.v4` push, 16 CTAs, uni | 876 | 128 B | 150 B | 1.172 | 1027 | 70% |
| SM `ld.v4` pull, 424 CTAs, uni | 1189 | 128 B read response | 142 B (+22 B request on reverse link) | 1.109 | 1319 | 89% |
| SM `ld.v4` pull, 424 CTAs, bidir | 1013 | responses + requests share each link | 82 B avg | 1.281 | 1298 | 88% |
| Copy engine 50% + SM push 106 CTAs 50%, uni | 1149 | mixed | 193 B avg | 1.129 | 1298 | 88% |
| TMA bulk store, uni (32 SMs, 4 KB chunks) | 1114 | 256 B | 278 B | 1.086 | 1210 | 82% |
| TMA bulk store, a2a (32 SMs, 2 KB chunks) | 1096 | 256 B | 273 B | 1.088 | 1192 | 81% |
| TMA bulk load (pull), uni | 1062 | 128 B read response | 142 B | 1.109 | 1178 | 80% |

- Write responses on the reverse link: ~20 B packets, 0.3-0.5% of forward bytes.
- Read requests: 22 B, one per 128 B response = 15.5% of response bytes on the reverse link.
- **~1300 GB/s per direction of packet bytes is a hard ceiling** (88% of 1476): four unrelated mixes plateau at 1298-1319. The remaining 12% is below what the counters see (flit framing, CRC/FEC, link-layer credits and acks).

Payload ceilings = 1300 / header factor:

| Packet type | Header factor | Payload ceiling (GB/s/dir) | Best measured | Limited by |
| --- | --: | --: | --- | --- |
| 256 B writes (copy engine, TMA bulk store) | 1.086 | ~1197 | 1114 TMA uni, 1096 TMA a2a, 1072 CE | injection (7-10% below ceiling) |
| 128 B writes (SM `st.v4`, NCCL Simple/LL128) | 1.172 | ~1109 | 1110 uni, 1081 bidir | wire |
| 128 B reads, one direction | 1.109 | ~1172 | 1189-1195 | wire |
| 128 B reads, both directions | 1.281 | ~1015 | 1013 | wire (requests eat the reverse link) |

NCCL's packets (1 GB steady loop, `nvlink -e`): Ring Simple and LL128 emit 128 B-payload packets (147.5 B on the wire, x1.175), the same as SM `st.v4`; only the TP2 symmetric all-gather emitted 256 B packets (272.7 B, x1.088). LL128 additionally sends 128/120 flag overhead.

## 4. Waterfall, one GPU, per direction (GB/s)

| Level | GB/s | % of spec | Cause | Recoverable in software? |
| --- | --: | --: | --- | --- |
| NVLink 6 spec (not measured) | 1800 | 100% | 50 GB/s x 36 links | |
| Link rate on these nodes (measured) | 1476 | 82% | links trained at 41.0 GB/s, full width, fabric healthy | no (platform/firmware) |
| Packet bytes on the wire (measured ceiling) | ~1300 | 72% | below the packet counters | no |
| Payload, 256 B packets | ~1197 | 66% | 22 B header per 256 B | ceiling for writes |
| Payload, 128 B packets | ~1109 | 62% | 22 B header per 128 B | use 256 B packets: +8% |
| SM pull uni / bidir | 1189 / 1013 | 66% / 56% | at wire ceiling; bidir loses ~15% to requests | do not pull in all-to-all |
| SM `st.v4` push uni / bidir | 1113 / 1081 | 62% / 60% | at the 128 B ceiling | switch to 256 B packets |
| TMA bulk store uni / a2a | 1114 / 1096 | 62% / 61% | injection-limited (wire 1192-1210) | partly: more bulk stores in flight |
| Copy engine uni / bidir | 1072 / 780-845 | 60% / 43-47% | CE-limited; bidir collapses | no; avoid CE for bidirectional |
| NCCL AR TP2, 8 GB / 192 MiB | 1096 / ~860-880 | 61% / 48% | at 128 B ceiling at large sizes; ramp and fixed cost at 192 MiB | a fused kernel avoids the ramp |
| TMA-store push a2a micro, 64 MB/peer | 1048-1096 | 58-61% | 88-92% of the 256 B ceiling | +9% max via injection |
| Push kernel `L2q` transport at 16K rows | 1076 | 60% | 90% of the 256 B ceiling | at most ~+11% (to the 256 B ceiling) |

## 5. Copy engines vs SMs (GB/s per direction, median)

| Size per direction | CE uni 1 stream | CE uni 8 streams | CE bidir best | SM push uni 212 CTAs | SM pull uni 424 CTAs |
| --: | --: | --: | --: | --: | --: |
| 1 MB | 70 | 20 | 76 | - | - |
| 16 MB | 637 | 241 | 517 | 684 | 619 |
| 64 MB | 945 | 603 | 903 | - | - |
| 256 MB | 1033 | 912 | 888 | 1080 | 1139 |
| 1 GB | 1064 | 1024 | 846 | 1113 | 1195 |
| 4 GB | 1072 | 1062 | 799 | - | - |

- SM push `st.v4` vs CTA count (1 GB uni): 8 CTAs 438, 16 878, 32 956, 64 1049, 106 1101, 212 1113, 424 1111 GB/s. Warp-specialized TMA-store staging saturates at 16-32 SMs.
- SM pull is latency-bound: 16 CTAs 165, 64 650, 212 1180, 424 1195 GB/s (~10 GB/s per CTA with 32 KB in flight, ~3 µs round trip).
- CE latency ~14-15 µs at 1 MB.

## 6. Kernel-relevant micro results (TP2, 64 MB per peer unless noted)

| Quantity | VR200 |
| --- | --- |
| a2a egress: push `st.v4` / TMA store / TMA pull | 957 / 1048 / 899 GB/s (uni: 1004 / 1079 / 1028) |
| a2a egress, TP4: TMA store / `st.v4` / TMA pull | 1179 / 1065 / 982 GB/s |
| SMs to saturate | push 16; TMA store 16 at >= 4 KB chunks; TMA pull 32 at 4-32 KB; per-thread `ld` pull never (175 GB/s at 16 SMs) |
| Small transfers (512 KB / 1 MB / 4 MB per peer) | TMA store, 4 KB chunks: 485 / 691 / 1039 GB/s |
| One-way latency | Lamport `st.v4` 1.57 µs (16 B-4 KB), 1.98 µs (16 KB), 3.26 µs (64 KB); TMA-store Lamport 1.62 µs; pull 5.84 µs; release/acquire flag 2.96 µs, relaxed flag 1.82 µs |
| sys-scope release / fence | 6.6k cycles idle; 29-52k cycles (12-21 µs) under NVLink load. A per-tile release cuts a push loop from 927 to 187 GB/s. gpu-scope local release ~free |
| 16 B tear over NVLink | 0 torn in 1.2-1.9e12 reads (`st.v4`, TMA store, `st.v2` writers; `ld.volatile` and TMA readers). Positive controls tear (4x `st.u32`: 4.4e9; split: 1.4e12) |
| TMA `report::validity`, cross-GPU | 0 false-valid with 290-364k retries. `per_16bytes::{80000000,8000,80,8}` checks only the **first** element of each 16 B (`::8` only the low nibble of byte 0); `per_element::ff` checks every byte. Peer validity load 2.7 µs, local 0.16-0.19 µs. `mbarrier.init.layout::v0` faults (sticky error); use `v1` |

## 7. nccl-tests (NCCL 2.31.2, out-of-place, busbw GB/s)

| TP | Collective | Config | 64 MB | 256 MB | 1 GB | 8 GB | 95% of asymptote at |
| --: | --- | --- | --: | --: | --: | --: | --: |
| 2 | all_reduce | default (Ring/Simple) | 579 | 869 | 967 | 1096 | 2 GB |
| 2 | all_reduce | Ring/Simple/64ch | 720 | 883 | 952 | 1062 | 4 GB |
| 2 | all_reduce | default, registered (`-R 1`) | 600 | 902 | 1047 | 1091 (4 GB) | 1 GB |
| 2 | sendrecv | default | 629 | 767 | 1045 | 1097 | 1 GB |
| 2 | all_gather | default | 545 | 723 | 794 | 956 | 8 GB |
| 2 | reduce_scatter | default | 521 | 662 | 716 | 883 | 8 GB |
| 2 | alltoall | `-R 1` | 562 | 886 | 1041 | 1081 | 1 GB |
| 4 | all_reduce | Ring/Simple/64ch | 635 | 993 | 1046 | 1098 | 1 GB |
| 4 | all_reduce | NVLS (busbw formula overstates bytes, see below) | 391 | 672 | 737 | 1102 (algbw 735) | 2 GB |
| 4 | all_reduce | Ring/LL128/64ch | 753 | 870 | 936 | 994 | 2 GB |
| 4 | all_gather | Ring/Simple/64ch | 552 | 875 | 963 | 1060 | 4 GB |
| 4 | reduce_scatter | Ring/Simple/64ch | 549 | 861 | 926 | 1041 | 4 GB |
| 4 | alltoall | `-R 2` | 673 | 954 | 1066 | 1101 | 1 GB |

TP2 all_reduce knobs: channels 16 -> 433, 32 -> 820, 64 -> 1062 GB/s (128 clamps to 64); `NCCL_NTHREADS` within 4%; LL128 caps at 736; `-R 1` +2-4% at 256 MB-1 GB; symmetric kernels (`-R 2`) poor at large sizes (AR 422); NVLS at TP2 472; uint8 = bf16.

Size dependence: `T = a + S/b` fit over 64 MB-1 GB gives a = 27 µs (TP2 AR), 43 µs (TP4). The marginal rate keeps rising (977 GB/s at 256 MB-1 GB, 1092 at 4-8 GB), so the production 192 MiB tier (~861 busbw) sits at ~80% of NCCL's own asymptote: a pipeline ramp, not a link limit.

NVLS caveat: nccl-tests reports NVLS busbw with the Ring factor 2(n-1)/n, but NVLS moves about S(1 + 1/n) per GPU (switch reduce + multicast). At TP4, NVLS algbw (735) equals Ring's (1098/1.5 = 732), so NVLS is not faster in time at any size up to 4 GB.

## 8. Conclusions for kernel design

- Use 256 B-packet writes (TMA bulk store) for egress: +8% payload ceiling over `st.v4` and NCCL.
- Push, not pull, for all-to-all: pull pays 22 B request packets on the reverse link (bidir ceiling ~1015 GB/s) and one extra traversal of latency.
- Copy engines collapse when both directions are active; NVLS does not help at TP2/TP4.
- Remaining software headroom for the push kernel at 16K: at most ~8-11% (more bulk stores in flight toward ~1197). The 1800 -> 1300 part is not recoverable in software.

Not measured: TP8 across both nodes (same clique, feasible over NVLink), NVLS wire counters.
