// fp4-allreduce: pull TP2 AllReduce, variant P (BF16 in, TMA bulk pull); frozen build q4.
// Variant P: fused NVFP4 AllReduce, encode to own window + TMA bulk pull from peers (one-shot and two-shot).
#include "fp4ar_api.h"
#include "fp4_codec.cuh"
#ifndef ENC_FAST
#define ENC_FAST 1                 // 1 (default): codec lane fp4f::enc_row_reg for the partial encode
#endif
#ifndef ENC_LDM
#define ENC_LDM 1                  // 1: L1-allocating x loads (merges the 64 B lane stride), 0: no-allocate
#endif
#if ENC_FAST
#include "codec_fast.cuh"
#endif
#include <cstdio>
#include <algorithm>

#define P_CAT2(a, b) a##b
#define P_CAT(a, b) P_CAT2(a, b)
#ifndef FP4AR_NAME
#define FP4AR_NAME P
#endif
#ifndef FP4AR_REGISTER
#define FP4AR_REGISTER P_CAT(fp4ar_register_, FP4AR_NAME)
#endif
namespace {  // per-object device module (variant contract)
using namespace fp4c;

constexpr int REC = 3472;          // row record: 3072 values | 384 scales | 4 row scale | 12 pad (16 B multiple)
constexpr int RS_OFF = VB + SB;    // row scale offset in a record
constexpr int MAXG = 1024;         // flag / epoch slots (>= grid)
constexpr int NP = 16;             // publish mbarrier ring depth (encoder run-ahead)
constexpr int NTHR = 640;          // WG0 fetch+publish, WG1-2 encoders, WG3-4 decoders
constexpr int NENC = 256, NDEC = 256, DEC0 = 384;
#ifndef ENC_REGS
#define ENC_REGS 120
#endif
#ifndef DEC_REGS
#define DEC_REGS 96
#endif
constexpr int SMEM_MAX = 232448;   // 227 KB opt-in
constexpr int HDR = 2048;          // barriers + scratch
constexpr int TAGSH = 20;          // tag = epoch << 20 | (local stage + 1)
constexpr int MAXSLOT = 8;
constexpr size_t RING_DEF = 113664;  // default ring bytes
#ifndef PROTO_LAMP
#define PROTO_LAMP 1               // 1: fence-free sentinel validation (shipped); 0: per-stage flag + release (diagnostic)
#endif
constexpr bool LAMP = PROTO_LAMP;
#ifndef REFILL_FENCE
#define REFILL_FENCE 2             // 1: fence.acq_rel.sys at call start; 2: after this CTA's refill (off the critical path); 0: none (diagnostic)
#endif
#ifndef PROBE
#define PROBE 2                    // Lamport hint: 2 weak progress tag (default), 1 poll row scales, 0 none
#endif
// PROBE 2: producers publish a weak (unfenced) per-CTA progress tag; consumers poll it as a hint only
constexpr bool HINT = LAMP && PROBE == 2;
// timing-only ablations (break correctness): skip encode math, skip decode+store, skip sentinel refill
#ifndef ABL_NOENC
#define ABL_NOENC 0
#endif
#ifndef ABL_NODEC
#define ABL_NODEC 0
#endif
#ifndef ABL_NOREFILL
#define ABL_NOREFILL 0
#endif
#ifndef PROBE_AHEAD
#define PROBE_AHEAD 4
#endif
constexpr int NBUF = LAMP ? 4 : 2; // window buffers per region, indexed by call % NBUF

struct KP {
  const __nv_bfloat16* x; __nv_bfloat16* out;
  char* win[FP4AR_MAX_RANKS];
  unsigned* epoch;
  int4* hist;                      // per call % 4: {rows, k, grid, twoshot}
  int rows, me, k, nstages, nslot, lag;
  size_t part_bytes, res_off, flag_off;
};

// ---------------- PTX helpers ----------------
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mb_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mb_expect(uint32_t b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(tx) : "memory");
}
__device__ __forceinline__ void mb_wait(uint32_t b, uint32_t par) {
  asm volatile("{\n.reg .pred P;\nW%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra W%=;\n}" :: "r"(b), "r"(par) : "memory");
}
__device__ __forceinline__ bool mb_test(uint32_t b, uint32_t par) {
  uint32_t r;
  asm volatile("{\n.reg .pred P;\nmbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2;\nselp.u32 %0, 1, 0, P;\n}" : "=r"(r) : "r"(b), "r"(par) : "memory");
  return r;
}
__device__ __forceinline__ void tma_pull(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ uint64_t ld_acq_sys(const uint64_t* a) {
  uint64_t v; asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(a) : "memory"); return v;
}
#ifndef PUB_GPU
#define PUB_GPU 0
#endif
__device__ __forceinline__ uint64_t ld_vol64(const uint64_t* a) {
  uint64_t v; asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(a) : "memory"); return v;
}
__device__ __forceinline__ void st_rel_sys(uint64_t* a, uint64_t v) {
#if PROTO_LAMP  // hint tag: weak store, no fence (validation in smem carries correctness)
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" :: "l"(a), "l"(v) : "memory");
#elif PUB_GPU  // experiment only: gpu-scope release is not formally sufficient for peer readers
  asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"(a), "l"(v) : "memory");
#else
  asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(a), "l"(v) : "memory");
#endif
}
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
template <int N> __device__ __forceinline__ void reg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(N)); }
template <int N> __device__ __forceinline__ void reg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(N)); }
__device__ __forceinline__ bool sent_val(uint32_t w) { return (w & ~((w & 0x77777777u) + 0x77777777u) & 0x88888888u) != 0; }
__device__ __forceinline__ bool sent_sf(uint32_t w) { const uint32_t x = ~w; return ((x - 0x01010101u) & ~x & 0x80808080u) != 0; }
__device__ __forceinline__ uint32_t ld_vol(const void* a) { uint32_t v; asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v; }
// Waits (with backoff) until the 4 B word at record offset off of a peer record is no longer a sentinel.
__device__ __forceinline__ void spin_valid(const char* g, uint32_t off) {
  const uint32_t o = off % REC; unsigned ns = 64;
  for (;;) {
    const uint32_t v = ld_vol(g + off);
    const bool bad = o < VB ? sent_val(v) : o < RS_OFF ? sent_sf(v) : v == SENT_RS;
    if (!bad) return;
    __nanosleep(ns); ns = ns < 1024 ? 2 * ns : ns;
  }
}
// Hint: true when every row scale of an n-row stage at g is written (n independent loads, one round trip).
__device__ __forceinline__ bool rows_ready(const char* g, int n) {
  uint32_t v[8];
#pragma unroll
  for (int r = 0; r < 8; r++) if (r < n) v[r] = ld_vol(g + (size_t)r * REC + RS_OFF);
  bool ok = true;
#pragma unroll
  for (int r = 0; r < 8; r++) if (r < n) ok &= v[r] != SENT_RS;
  return ok;
}
__device__ __forceinline__ void wait_rows(const char* g, int n) {
  unsigned ns = 64;
  while (!rows_ready(g, n)) { __nanosleep(ns); ns = ns < 1024 ? 2 * ns : ns; }
}
// Fills one row record with sentinels (217 x 16 B); t indexes 16 B granules.
__device__ __forceinline__ void sent_fill16(char* rec, int t) {
  const uint4 v = t < 192 ? make_uint4(SENT_VAL32, SENT_VAL32, SENT_VAL32, SENT_VAL32)
                : t < 216 ? make_uint4(~0u, ~0u, ~0u, ~0u) : make_uint4(SENT_RS, 0u, 0u, 0u);
  ((uint4*)rec)[t] = v;
}
__device__ __forceinline__ bool ge_tag(uint64_t v, uint64_t want) { return (long long)(v - want) >= 0; }

// ---------------- codec on 8-value chunks (bit-identical to fp4c::enc_pair_f / dec_acc) ----------------
// Encode chunk ch (8 values) of a row; lanes ch^1 pair up into one 16-block. Returns E2M1 word, sets scale byte.
__device__ __forceinline__ uint32_t enc8(const float* f, float inv, uint32_t& s) {
  float bm = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) bm = fmaxf(bm, fabsf(f[i]));
  bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1));
  const float sv = __fdiv_rn(__fmul_rn(bm, inv), 6.f);
  s = (uint32_t)__nv_cvt_float_to_fp8(sv, __NV_SATFINITE, __NV_E4M3);
  const float sff = e4m3f(s);
  const float m = sff > 0.f ? __fdiv_rn(inv, sff) : 0.f;
  float g[8];
#pragma unroll
  for (int i = 0; i < 8; i++) g[i] = __fmul_rn(f[i], m);
  return e2m1x8(g);
}
// Packs the scale bytes of 4 consecutive 16-blocks (lanes l, l+2, l+4, l+6) into lane l (l % 8 == 0).
__device__ __forceinline__ uint32_t pack_sf(uint32_t s) {
  uint32_t w = s;
  w |= __shfl_down_sync(0xffffffffu, s, 2) << 8;
  w |= __shfl_down_sync(0xffffffffu, s, 4) << 16;
  w |= __shfl_down_sync(0xffffffffu, s, 6) << 24;
  return w;
}
__device__ __forceinline__ void dec8(float* acc, uint32_t w, float scl) {
  __half2 h[4]; e2m1x8_dec(w, h);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 t = __half22float2(h[i]);
    acc[2 * i] = __fmaf_rn(t.x, scl, acc[2 * i]);
    acc[2 * i + 1] = __fmaf_rn(t.y, scl, acc[2 * i + 1]);
  }
}
__device__ __forceinline__ void st_bf16x8(__nv_bfloat16* o, const float* a) {
  uint4 u; __nv_bfloat162* b = (__nv_bfloat162*)&u;
#pragma unroll
  for (int i = 0; i < 4; i++) b[i] = __floats2bfloat162_rn(a[2 * i], a[2 * i + 1]);
  *(uint4*)o = u;
}

// Two-shot work order: owned stages (reduce) run ahead of non-owned stages (gather) by `lag` stages.
struct Seq {
  int jo, jn, nmy, c, me, NR, lag;
  __device__ void init(int c_, int me_, int NR_, int nmy_, int lag_) {
    c = c_; me = me_; NR = NR_; nmy = nmy_; lag = lag_;
    jo = ((me - c) % NR + NR) % NR; jn = 0; if ((c + jn) % NR == me) jn++;
  }
  __device__ void next(int& j, bool& red) {
    const bool hasR = jo < nmy, hasN = jn < nmy;
    if (hasR && (!hasN || jo <= jn + lag)) { j = jo; red = true; jo += NR; }
    else { j = jn; red = false; jn++; if ((c + jn) % NR == me) jn++; }
  }
};

// ---------------- kernel ----------------
template <int NR, int MODE, bool TS>
__global__ void __launch_bounds__(NTHR, 1) pull_kernel(KP p) {
  extern __shared__ __align__(1024) char smem[];
  uint64_t* bars = (uint64_t*)smem;
  // barrier map: full[8] empty[8] pubF[NP] pubE[NP] rpubF[NP] rpubE[NP]; pubE needs publisher + fetch
  const uint32_t full0 = sa(bars), empty0 = sa(bars + 8), pubF0 = sa(bars + 16), pubE0 = sa(bars + 16 + NP);
  const uint32_t rpubF0 = sa(bars + 16 + 2 * NP), rpubE0 = sa(bars + 16 + 3 * NP);
  float* red = (float*)(smem + 1024);         // [2][8 rows][8 warps] two-shot row amax
  unsigned* s_epoch = (unsigned*)(smem + 1536);
  uint32_t* s_bad = (uint32_t*)(smem + 1600);       // [2] bad-source mask per check round
  uint32_t* s_first = (uint32_t*)(smem + 1664);     // [2][8] first sentinel offset per source
  const uint32_t rtry0 = sa(bars + 16 + 4 * NP);    // re-pull barrier
  char* ring = smem + HDR;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int c = blockIdx.x, G = gridDim.x, k = p.k;
  const int nmy = c < p.nstages ? (p.nstages - c + G - 1) / G : 0;
  const size_t slot_bytes = (size_t)NR * k * REC;
  int nown = 0;
  if (TS) { const int jo = ((p.me - c) % NR + NR) % NR; nown = jo < nmy ? (nmy - jo + NR - 1) / NR : 0; }

  if (tid == 0) {
    for (int i = 0; i < MAXSLOT; i++) { mb_init(full0 + 8 * i, 1); mb_init(empty0 + 8 * i, NDEC / 32); }
    for (int i = 0; i < NP; i++) { mb_init(pubF0 + 8 * i, NENC); mb_init(pubE0 + 8 * i, (LAMP && !HINT) ? 1 : 2); mb_init(rpubF0 + 8 * i, NDEC); mb_init(rpubE0 + 8 * i, 1); }
    mb_init(rtry0, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // PDL: everything above overlaps the previous kernel; window/epoch/x access only after this.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (tid == 0) {
    *s_epoch = p.epoch[c] + 1;
    // orders the previous calls' sentinel refills before this call's data stores (see DESIGN.md)
    if (LAMP && REFILL_FENCE == 1) asm volatile("fence.acq_rel.sys;" ::: "memory");
  }
  __syncthreads();
  const unsigned ep = *s_epoch;
  if (tid == 0) for (int i = c; i < MAXG; i += G) p.epoch[i] = ep;   // keep every slot = call count
  if (tid == 0 && c == 0) p.hist[ep & 3] = make_int4(p.rows, k, G, TS ? 1 : 0);
  asm volatile("griddepcontrol.launch_dependents;");
  const int par = ep % NBUF;  // buffer index of this call
  char* const self = p.win[p.me];
  uint64_t* const myP = (uint64_t*)(self + p.flag_off) + c;
  uint64_t* const myR = (uint64_t*)(self + p.flag_off) + MAXG + c;
  const uint64_t tag0 = (uint64_t)ep << TAGSH;

  if (warp < 4) {
    reg_dec<40>();  // 128x40 + 256x(ENC+DEC) must fit the 640x96 launch allocation
    if (warp == 0) {
      // ---- fetch warp: lane s pulls source s ----
      uint64_t seenP = 0, seenR = 0;
      int okP = -1;  // Lamport probe: highest own-lane part stage seen complete (hint only)
      int sd = 0;  // own stages consumed from pubF (lane 0)
      auto self_upto = [&](int j) {
        for (; sd <= j; sd++) { mb_wait(pubF0 + 8 * (sd % NP), (sd / NP) & 1); mb_arrive(pubE0 + 8 * (sd % NP)); }
      };
      Seq sq; sq.init(c, p.me, NR, nmy, p.lag);
      for (int i = 0; i < nmy; i++) {
        int j; bool rd;
        if (TS) sq.next(j, rd); else { j = i; rd = true; }
        const int slot = i % p.nslot; const uint32_t ph = (i / p.nslot) & 1;
        const int r0 = (c + G * j) * k, nrows = min(k, p.rows - r0);
        const uint32_t bytes = nrows * REC;
        const int owner = (c + j) % NR;
        const int nsrc = rd ? NR : 1;
        const bool needSelf = rd || MODE == FP4AR_COMPUTE;
        if (lane == 0) { mb_wait(empty0 + 8 * slot, ph ^ 1); mb_expect(full0 + 8 * slot, nsrc * bytes); if (needSelf) self_upto(j); }
        __syncwarp();
        const bool act = rd ? lane < NR : lane == owner;
        if (act) {
          const int src = MODE == FP4AR_COMPUTE ? p.me : lane;
          const bool useRes = TS && !rd && MODE != FP4AR_COMPUTE;
          const uint64_t want = tag0 | (uint64_t)(j + 1);
          const uint64_t* fl = (const uint64_t*)(p.win[src] + p.flag_off) + (useRes ? MAXG : 0) + c;
          uint64_t& seen = useRes ? seenR : seenP;
          if ((!LAMP || HINT) && src != p.me && !ge_tag(seen, want)) {
            unsigned ns = 32;
            if (HINT) { while (!ge_tag(seen = ld_vol64(fl), want)) { __nanosleep(ns); ns = ns < 512 ? 2 * ns : ns; } }
            else { do { seen = ld_acq_sys(fl); } while (!ge_tag(seen, want)); }
          }
          asm volatile("fence.proxy.async.global;" ::: "memory");
          const char* g = p.win[src] + (useRes ? p.res_off : 0) + (size_t)par * p.part_bytes + (size_t)r0 * REC;
          // heuristic only (correctness comes from the decoders' check): wait for the stage's last row scale before pulling
          if (LAMP && PROBE == 1 && src != p.me) {
            if (useRes) wait_rows(g, nrows);
            else if (j > okP) {
              // look ahead: one round trip usually clears several stages, since every encoder warp walks stages in order
              const int ja = min(j + PROBE_AHEAD, nmy - 1), ra = (c + G * ja) * k, na = min(k, p.rows - ra);
              if (ja > j && rows_ready(p.win[src] + (size_t)par * p.part_bytes + (size_t)ra * REC, na)) okP = ja;
              else { wait_rows(g, nrows); okP = j; }
            }
          }
          const uint32_t dst = sa(ring + slot * slot_bytes + (size_t)(rd ? lane : 0) * k * REC);
          tma_pull(dst, g, bytes, full0 + 8 * slot);
        }
        __syncwarp();
      }
      if (lane == 0) self_upto(nmy - 1);
    } else if ((!LAMP || HINT) && warp == 1 && lane == 0) {
      // ---- publisher: one st.release.sys per stage (partials) and per owned stage (results) ----
      int tP = 0, tR = 0;
      const int jo0 = ((p.me - c) % NR + NR) % NR;
      while (tP < nmy || tR < nown) {
        // coalesce: one release covers every stage completed since the last one
        int t0 = tP;
        while (tP < nmy && mb_test(pubF0 + 8 * (tP % NP), (tP / NP) & 1)) tP++;
        if (tP > t0) {
          st_rel_sys(myP, tag0 | (uint64_t)tP);
          for (int t = t0; t < tP; t++) mb_arrive(pubE0 + 8 * (t % NP));
        }
        if (TS) {
          t0 = tR;
          while (tR < nown && mb_test(rpubF0 + 8 * (tR % NP), (tR / NP) & 1)) tR++;
          if (tR > t0) {
            st_rel_sys(myR, tag0 | (uint64_t)(jo0 + NR * (tR - 1) + 1));
            for (int t = t0; t < tR; t++) mb_arrive(rpubE0 + 8 * (t % NP));
          }
        }
      }
    }
    return;
  }
  if (warp < 12) {
    // ---- encoders: one warp per row, single pass from registers, local stores only ----
    reg_inc<ENC_REGS>();
    const int ew = warp - 4;
#if ENC_FAST == 2
    // half-row encoder: warp pair (2q, 2q+1) shares a row; lane-interleaved 16 B loads, 48 data regs, no spills
    const int pr = ew >> 1, hf = ew & 1;
    float* s_am = (float*)(smem + 1792);  // [4 pairs][2 parity][2 halves]
    int rc = 0;
    for (int j = 0; j < nmy; j++) {
      const int r0 = (c + G * j) * k, nrows = min(k, p.rows - r0);
      for (int rr = (pr - j * k % 4 + 4) % 4; rr < nrows; rr += 4, rc++) {
        const int row = r0 + rr;
        char* rec = self + (size_t)par * p.part_bytes + (size_t)row * REC;
        if (MODE == FP4AR_XPORT || ABL_NOENC) {
#pragma unroll 4
          for (int i = 0; i < 12; i++) { const int ch = hf * 384 + lane + 32 * i; ((uint32_t*)rec)[ch] = canon_neg0(ep ^ (row * 768 + ch)); }
          if (lane < 16) { ((uint32_t*)(rec + VB))[hf * 48 + lane] = 0x38383838u; ((uint32_t*)(rec + VB))[hf * 48 + 16 + lane] = 0x38383838u; ((uint32_t*)(rec + VB))[hf * 48 + 32 + lane] = 0x38383838u; }
          if (hf == 0 && lane == 0) *(float*)(rec + RS_OFF) = 1.f;
          continue;
        }
        const uint4* xr = (const uint4*)(p.x + (size_t)row * H) + hf * 384;
        uint4 v[12]; float am = 0.f;
#pragma unroll
        for (int i = 0; i < 12; i++) v[i] = __ldg(xr + lane + 32 * i);
#pragma unroll
        for (int i = 0; i < 12; i++) am = amax16B(v[i], am);
        am = warp_amax(am);
        float* sa2 = s_am + pr * 4 + (rc & 1) * 2;
        if (lane == 0) sa2[hf] = am;
        bar_sync(4 + pr, 64);
        am = fmaxf(sa2[0], sa2[1]);
        float inv, rs; row_scales(am, inv, rs);
#pragma unroll
        for (int i = 0; i < 12; i++) {
          const int ch = hf * 384 + lane + 32 * i;
          float f[8]; const __nv_bfloat162* b = (const __nv_bfloat162*)&v[i];
#pragma unroll
          for (int q = 0; q < 4; q++) { float2 t = __bfloat1622float2(b[q]); f[2 * q] = t.x; f[2 * q + 1] = t.y; }
          float bm = 0.f;
#pragma unroll
          for (int q = 0; q < 8; q++) bm = fmaxf(bm, fabsf(f[q]));
          bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1));
          const uint32_t sc = (uint32_t)__nv_cvt_float_to_fp8(fp4f::div6_rn(__fmul_rn(bm, inv)), __NV_SATFINITE, __NV_E4M3);
          const float sff = e4m3f(sc), m = sff > 0.f ? __fdiv_rn(inv, sff) : 0.f;
#pragma unroll
          for (int q = 0; q < 8; q++) f[q] = __fmul_rn(f[q], m);
          ((uint32_t*)rec)[ch] = canon_neg0(e2m1x8(f));
          const uint32_t sw = pack_sf(sc);
          if ((lane & 7) == 0) *(uint32_t*)(rec + VB + ch / 2) = sw;
        }
        if (hf == 0 && lane == 0) *(float*)(rec + RS_OFF) = rs;
      }
#else
    for (int j = 0; j < nmy; j++) {
      const int r0 = (c + G * j) * k, nrows = min(k, p.rows - r0);
      for (int rr = (ew - j * k % 8 + 8) % 8; rr < nrows; rr += 8) {  // rows rotate over warps across stages
        const int row = r0 + rr;
        char* rec = self + (size_t)par * p.part_bytes + (size_t)row * REC;
        if (MODE == FP4AR_XPORT || ABL_NOENC) {
#pragma unroll 4
          for (int i = 0; i < 24; i++) ((uint32_t*)rec)[lane + 32 * i] = canon_neg0(ep ^ (row * 768 + lane + 32 * i));
          for (int i = 0; i < 3; i++) ((uint32_t*)(rec + VB))[lane + 32 * i] = 0x38383838u;
          if (lane == 0) *(float*)(rec + RS_OFF) = 1.f;
          continue;
        }
#if ENC_FAST
        fp4f::enc_row_reg<true, ENC_LDM>(p.x + (size_t)row * H, lane, fp4f::EncRow{(uint4*)rec, (uint16_t*)(rec + VB), (float*)(rec + RS_OFF)});
        continue;
#endif
        const uint4* xr = (const uint4*)(p.x + (size_t)row * H);
        uint4 v[24]; float am = 0.f;
#pragma unroll
        for (int i = 0; i < 12; i++) { v[2 * i] = __ldg(xr + 2 * (lane + 32 * i)); v[2 * i + 1] = __ldg(xr + 2 * (lane + 32 * i) + 1); }
#pragma unroll
        for (int i = 0; i < 24; i++) am = amax16B(v[i], am);
        am = warp_amax(am);
        float inv, rs; row_scales(am, inv, rs);
#pragma unroll
        for (int i = 0; i < 12; i++) {
          const int bk = lane + 32 * i;  // one 16-value block per lane
          float f[16]; const __nv_bfloat162* b = (const __nv_bfloat162*)&v[2 * i];
#pragma unroll
          for (int q = 0; q < 8; q++) { float2 t = __bfloat1622float2(b[q]); f[2 * q] = t.x; f[2 * q + 1] = t.y; }
          float bm = 0.f;
#pragma unroll
          for (int q = 0; q < 16; q++) bm = fmaxf(bm, fabsf(f[q]));
          const uint32_t sc = (uint32_t)__nv_cvt_float_to_fp8(__fdiv_rn(__fmul_rn(bm, inv), 6.f), __NV_SATFINITE, __NV_E4M3);
          const float sff = e4m3f(sc), m = sff > 0.f ? __fdiv_rn(inv, sff) : 0.f;
#pragma unroll
          for (int q = 0; q < 16; q++) f[q] = __fmul_rn(f[q], m);
          ((uint2*)rec)[bk] = make_uint2(canon_neg0(e2m1x8(f)), canon_neg0(e2m1x8(f + 8)));
          uint32_t sw = sc | (__shfl_down_sync(0xffffffffu, sc, 1) << 8) | (__shfl_down_sync(0xffffffffu, sc, 2) << 16) | (__shfl_down_sync(0xffffffffu, sc, 3) << 24);
          if ((lane & 3) == 0) *(uint32_t*)(rec + VB + bk) = sw;
        }
        if (lane == 0) *(float*)(rec + RS_OFF) = rs;
      }
#endif
      if (j >= NP) mb_wait(pubE0 + 8 * (j % NP), ((j / NP) & 1) ^ 1);
      mb_arrive(pubF0 + 8 * (j % NP));
    }
    if (LAMP && !ABL_NOREFILL) {
      // refill call ep-2's buffer (next used by call ep+2) with sentinels; this CTA takes a contiguous row share
      const int4 h = p.hist[(ep + 2) & 3];
      const int b2 = (ep + 2) % NBUF, et = tid - 128;
      char* pb = self + (size_t)b2 * p.part_bytes;
      const int per = (h.x + G - 1) / G, rlo = min(h.x, c * per), rhi = min(h.x, rlo + per);
      for (int t = et; t < (rhi - rlo) * 217; t += NENC) sent_fill16(pb + (size_t)(rlo + t / 217) * REC, t % 217);
      if (h.w) {
        const int nst2 = (h.x + h.y - 1) / h.y;
        char* rb = self + p.res_off + (size_t)b2 * p.part_bytes;
        for (int s2 = c; s2 < nst2; s2 += G) {
          if ((s2 % h.z + s2 / h.z) % NR != p.me) continue;
          const int a0 = s2 * h.y, n2 = min(h.y, h.x - a0);
          for (int t = et; t < n2 * 217; t += NENC) sent_fill16(rb + (size_t)(a0 + t / 217) * REC, t % 217);
        }
      }
      if (REFILL_FENCE == 2) {
        bar_sync(3, NENC);
        if (et == 0) asm volatile("fence.acq_rel.sys;" ::: "memory");
      }
    }
    return;
  }
  // ---- decode-reduce warps ----
  reg_inc<DEC_REGS>();
  const int dt = tid - DEC0, dw = dt >> 5;
  Seq sq; sq.init(c, p.me, NR, nmy, p.lag);
  int q = 0;  // owned-stage counter (two-shot)
  uint32_t chk = 0, rp = 0;  // check rounds, re-pull barrier parity
  for (int i = 0; i < nmy; i++) {
    int j; bool rd;
    if (TS) sq.next(j, rd); else { j = i; rd = true; }
    const int slot = i % p.nslot;
    const int r0 = (c + G * j) * k, nrows = min(k, p.rows - r0);
    const char* sb = ring + slot * slot_bytes;
    mb_wait(full0 + 8 * slot, (i / p.nslot) & 1);
    if (LAMP && MODE != FP4AR_COMPUTE) {
      // validate peer sources in smem; re-pull any source that still holds sentinels
      const int npos = (!TS || rd) ? NR : 1, owner = (c + j) % NR;
      for (;;) {
        const int cb = chk & 1;
        if (dt < 8) s_first[cb * 8 + dt] = ~0u;
        if (dt == 0) s_bad[cb] = 0;
        bar_sync(2, NDEC);
#pragma unroll
        for (int ps = 0; ps < NR; ps++) {
          const int src = (!TS || rd) ? ps : owner;
          if (ps >= npos || src == p.me) continue;
          const char* base = sb + (size_t)ps * k * REC;
          for (int rr = 0; rr < nrows; rr++) {
            const char* rb = base + (size_t)rr * REC;
            uint32_t off = ~0u;
#pragma unroll
            for (int u = 2; u >= 0; u--) if (sent_val(((const uint32_t*)rb)[u * 256 + dt])) off = rr * REC + 4 * (u * 256 + dt);
            if (dt < 96) { if (sent_sf(((const uint32_t*)(rb + VB))[dt])) off = min(off, (uint32_t)(rr * REC + VB + 4 * dt)); }
            else if (dt == 96) { if (*(const uint32_t*)(rb + RS_OFF) == SENT_RS) off = min(off, (uint32_t)(rr * REC + RS_OFF)); }
            if (off != ~0u) { atomicOr(&s_bad[cb], 1u << ps); atomicMin(&s_first[cb * 8 + ps], off); }
          }
        }
        bar_sync(2, NDEC);
        const uint32_t bad = s_bad[cb];
        chk++;
        if (!bad) break;
        if (dt == 0) {
          const char* gs[NR]; uint32_t tot = 0;
#pragma unroll
          for (int ps = 0; ps < NR; ps++) {
            const int src = (!TS || rd) ? ps : owner;
            gs[ps] = p.win[src] + ((TS && !rd) ? p.res_off : 0) + (size_t)par * p.part_bytes + (size_t)r0 * REC;
            if (bad >> ps & 1) { spin_valid(gs[ps], s_first[cb * 8 + ps]); tot += nrows * REC; }
          }
          mb_expect(rtry0, tot);
#pragma unroll
          for (int ps = 0; ps < NR; ps++) if (bad >> ps & 1) tma_pull(sa(sb + (size_t)ps * k * REC), gs[ps], nrows * REC, rtry0);
        }
        mb_wait(rtry0, rp); rp ^= 1;
      }
    }
    if (MODE != FP4AR_XPORT && !ABL_NODEC) {
      if (!TS || !rd) {
        // one-shot reduce (NR sources) or two-shot gather (1 source at offset 0)
        const int ns = TS ? 1 : NR;
        for (int rr = 0; rr < nrows; rr++) {
          float rsv[NR];
#pragma unroll
          for (int s = 0; s < NR; s++) if (s < ns) rsv[s] = *(const float*)(sb + (size_t)(s * k + rr) * REC + RS_OFF);
#pragma unroll
          for (int u = 0; u < 3; u++) {
            const int ch = u * 256 + dt;
            float acc[8];
#pragma unroll
            for (int e = 0; e < 8; e++) acc[e] = 0.f;
#pragma unroll
            for (int s = 0; s < NR; s++) {
              if (s < ns) {
                const char* rb = sb + (size_t)(s * k + rr) * REC;
                const uint32_t w = ((const uint32_t*)rb)[ch];
                const uint32_t sc = ((const uint8_t*)rb)[VB + ch / 2];
                dec8(acc, w, __fmul_rn(e4m3f(sc), rsv[s]));
              }
            }
            st_bf16x8(p.out + (size_t)(r0 + rr) * H + 8 * ch, acc);
          }
        }
      } else {
        // two-shot owner: sum NR sources, row amax, re-encode to own result region, decode own encoding
        float* rbuf = red + (q & 1) * 64;
        for (int rr = 0; rr < nrows; rr++) {
          float rsv[NR];
#pragma unroll
          for (int s = 0; s < NR; s++) rsv[s] = *(const float*)(sb + (size_t)(s * k + rr) * REC + RS_OFF);
          float am = 0.f;
#pragma unroll
          for (int u = 0; u < 3; u++) {
            const int ch = u * 256 + dt;
            float acc[8];
#pragma unroll
            for (int e = 0; e < 8; e++) acc[e] = 0.f;
#pragma unroll
            for (int s = 0; s < NR; s++) {
              const char* rb = sb + (size_t)(s * k + rr) * REC;
              dec8(acc, ((const uint32_t*)rb)[ch], __fmul_rn(e4m3f(((const uint8_t*)rb)[VB + ch / 2]), rsv[s]));
            }
#pragma unroll
            for (int e = 0; e < 8; e++) am = fmaxf(am, fabsf(acc[e]));
          }
          am = warp_amax(am);
          if (lane == 0) rbuf[rr * 8 + dw] = am;
        }
        bar_sync(1, NDEC);
        for (int rr = 0; rr < nrows; rr++) {
          float am = 0.f;
#pragma unroll
          for (int w8 = 0; w8 < 8; w8++) am = fmaxf(am, rbuf[rr * 8 + w8]);
          float inv, rs2; row_scales(am, inv, rs2);
          float rsv[NR];
#pragma unroll
          for (int s = 0; s < NR; s++) rsv[s] = *(const float*)(sb + (size_t)(s * k + rr) * REC + RS_OFF);
          char* rec = self + p.res_off + (size_t)par * p.part_bytes + (size_t)(r0 + rr) * REC;
#pragma unroll
          for (int u = 0; u < 3; u++) {
            const int ch = u * 256 + dt;
            float acc[8];
#pragma unroll
            for (int e = 0; e < 8; e++) acc[e] = 0.f;
#pragma unroll
            for (int s = 0; s < NR; s++) {
              const char* rb = sb + (size_t)(s * k + rr) * REC;
              dec8(acc, ((const uint32_t*)rb)[ch], __fmul_rn(e4m3f(((const uint8_t*)rb)[VB + ch / 2]), rsv[s]));
            }
            uint32_t sc; const uint32_t w = canon_neg0(enc8(acc, inv, sc));
            ((uint32_t*)rec)[ch] = w;
            const uint32_t sw = pack_sf(sc);
            if ((lane & 7) == 0) *(uint32_t*)(rec + VB + ch / 2) = sw;
            float o[8];
#pragma unroll
            for (int e = 0; e < 8; e++) o[e] = 0.f;
            dec8(o, w, __fmul_rn(e4m3f(sc), rs2));
            st_bf16x8(p.out + (size_t)(r0 + rr) * H + 8 * ch, o);
          }
          if (dt == 0) *(float*)(rec + RS_OFF) = rs2;
        }
      }
    } else if (TS && rd) {
      // XPORT owner: same result bytes, no codec
      for (int rr = 0; rr < nrows; rr++) {
        char* rec = self + p.res_off + (size_t)par * p.part_bytes + (size_t)(r0 + rr) * REC;
#pragma unroll
        for (int u = 0; u < 3; u++) { const int ch = u * 256 + dt; ((uint32_t*)rec)[ch] = canon_neg0(ep ^ ch); if ((lane & 7) == 0) *(uint32_t*)(rec + VB + ch / 2) = 0x38383838u; }
        if (dt == 0) *(float*)(rec + RS_OFF) = 1.f;
      }
    }
    __syncwarp();
    if (lane == 0) mb_arrive(empty0 + 8 * slot);
    if (TS && rd) {
      if (!LAMP || HINT) {
        if (q >= NP) mb_wait(rpubE0 + 8 * (q % NP), ((q / NP) & 1) ^ 1);
        mb_arrive(rpubF0 + 8 * (q % NP));
      }
      q++;
    }
  }
}

// ---------------- host ----------------
__global__ void prefill(char* w, size_t nrec) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nrec * 217; i += (size_t)gridDim.x * blockDim.x)
    sent_fill16(w + (i / 217) * REC, (int)(i % 217));
}
struct Priv { int4* hist; unsigned* epoch; int max_rows; size_t part_bytes, res_off, flag_off; };

static size_t layout(int max_rows, size_t& part, size_t& res, size_t& flag) {
  part = (size_t)max_rows * REC; res = NBUF * part; flag = 2 * NBUF * part;
  return (flag + 16 * MAXG + 4095) / 4096 * 4096;
}
static size_t wbytes(int max_rows, int, int) { size_t a, b, f; return layout(max_rows, a, b, f); }

template <int NR, int MODE, bool TS> static const void* kfn() { return (const void*)pull_kernel<NR, MODE, TS>; }
template <int NR> static const void* pick(int mode, bool ts) {
  if (ts) return mode == FP4AR_XPORT ? kfn<NR, FP4AR_XPORT, true>() : mode == FP4AR_COMPUTE ? kfn<NR, FP4AR_COMPUTE, true>() : kfn<NR, FP4AR_FUSED, true>();
  return mode == FP4AR_XPORT ? kfn<NR, FP4AR_XPORT, false>() : mode == FP4AR_COMPUTE ? kfn<NR, FP4AR_COMPUTE, false>() : kfn<NR, FP4AR_FUSED, false>();
}
static const void* pickN(int nr, int mode, bool ts) {
  return nr == 2 ? pick<2>(mode, ts) : nr == 4 ? pick<4>(mode, ts) : nr == 8 ? pick<8>(mode, ts) : nullptr;
}

static int init(Fp4ArCtx* ctx, int max_rows, int hidden) {
  if (hidden != H || (ctx->nranks != 2 && ctx->nranks != 4 && ctx->nranks != 8)) return -1;
  Priv* pv = new Priv; pv->max_rows = max_rows;
  size_t need = layout(max_rows, pv->part_bytes, pv->res_off, pv->flag_off);
  if (need > ctx->win_bytes) { delete pv; return -2; }
  if (cudaMalloc(&pv->epoch, MAXG * 4) != cudaSuccess) { delete pv; return -3; }
  cudaMemset(pv->epoch, 0, MAXG * 4);
  if (cudaMalloc(&pv->hist, 4 * sizeof(int4)) != cudaSuccess) { delete pv; return -3; }
  cudaMemset(pv->hist, 0, 4 * sizeof(int4));
  if (LAMP) prefill<<<1024, 256>>>(ctx->peer[ctx->rank], (size_t)2 * NBUF * max_rows);
  cudaMemset(ctx->peer[ctx->rank] + pv->flag_off, 0, 16 * MAXG);
  for (int m = 0; m < 3; m++) for (int t = 0; t < 2; t++)
    cudaFuncSetAttribute(pickN(ctx->nranks, m, t), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  // collective barrier so no peer polls flags before they are zeroed
  float* d; cudaMalloc(&d, 4); cudaMemset(d, 0, 4);
  ncclAllReduce(d, d, 1, ncclFloat, ncclSum, ctx->comm, 0);
  cudaDeviceSynchronize(); cudaFree(d);
  ctx->priv = pv;
  return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

// tune[0] ring slots, tune[1] rows/stage, tune[2] grid CTAs, tune[3] two-shot lag | 0x100 = no PDL
static int launch(Fp4ArCtx* ctx, const Fp4ArArgs* a, cudaStream_t s) {
  Priv* pv = (Priv*)ctx->priv; const int NR = ctx->nranks;
  if (a->hidden != H || a->rows > pv->max_rows || a->rows < NR) return -1;
  const bool ts = a->algo == FP4AR_TWOSHOT;
  int G0 = a->tune[2] > 0 ? std::min(a->tune[2], MAXG) : ctx->sm_count;
  const int kfit = (SMEM_MAX - HDR) / (2 * NR * REC);                 // >= 2 slots
  const int kmax = std::max(1, std::min(8 / (NR / 2), (SMEM_MAX - HDR) / (3 * NR * REC)) / 2);  // TP2: 4 (v11 sweep)
  int k = a->tune[1] > 0 ? std::min(a->tune[1], std::min(8, kfit)) : std::max(1, std::min(kmax, a->rows / (G0 * 4)));
  if (k < 1) return -2;
  int nst = (a->rows + k - 1) / k;
  if (nst < NR) { k = 1; nst = a->rows; }
  const int G = std::min(G0, nst);
  if (ts && G < NR) return -3;
  const size_t slot = (size_t)NR * k * REC;
  // default ring ~111 KB: leaves L1 for the encoders' L1-allocating loads (bigger rings measured slower at 16K)
  int nslot = a->tune[0] > 0 ? a->tune[0] : std::max(2, std::min(MAXSLOT, (int)(RING_DEF / slot)));
  nslot = std::min<int>(std::min(nslot, MAXSLOT), (SMEM_MAX - HDR) / slot);
  if (nslot < 2) return -4;
  KP p{};
  p.x = a->x; p.out = a->out;
  for (int r = 0; r < NR; r++) p.win[r] = ctx->peer[r];
  p.epoch = pv->epoch; p.hist = pv->hist; p.rows = a->rows; p.me = ctx->rank; p.k = k; p.nstages = nst; p.nslot = nslot;
  p.lag = (a->tune[3] & 0xff) ? (a->tune[3] & 0xff) : 2 * NR;
  p.part_bytes = pv->part_bytes; p.res_off = pv->res_off; p.flag_off = pv->flag_off;
  cudaLaunchConfig_t cfg{}; cudaLaunchAttribute at[1];
  cfg.gridDim = dim3(G); cfg.blockDim = dim3(NTHR); cfg.dynamicSmemBytes = HDR + nslot * slot; cfg.stream = s;
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = (a->tune[3] & 0x100) ? 0 : 1;
  void* args[] = {&p};
  return cudaLaunchKernelExC(&cfg, pickN(NR, a->mode, ts), args) == cudaSuccess ? 0 : -5;
}
static void fini(Fp4ArCtx* ctx) {
  Priv* pv = (Priv*)ctx->priv; if (!pv) return;
  cudaFree(pv->epoch); cudaFree(pv->hist); delete pv; ctx->priv = nullptr;
}
}  // namespace

#define P_STR2(x) #x
#define P_STR(x) P_STR2(x)
static const Fp4ArVariant kVariantP = {P_STR(FP4AR_NAME), true, wbytes, init, launch, fini};
const Fp4ArVariant* FP4AR_REGISTER() { return &kVariantP; }
