// fp4-allreduce: push TP2 one-shot Lamport AllReduce, variants L2 (BF16 in) / L2q (NVFP4 in); frozen build L-f1.
// Variants L (Lamport push, per-word sentinel poll) and LT (L + TMA local load with validity report).
// Design, wire layout and protocol: DESIGN.md. Build flags: -DLAMPORT_LT=1 (variant LT), -DFP4AR_LT_HW (sm_107a validity PTX).
// tune[0] = CTAs (0 -> SM count), tune[1] = decode warps per row ds in {1,2,3,6} (0 -> 3, or 6 when rows/CTA < 3),
// tune[2] = flags (bit0 no PDL, bit1 LT skip sw verify, bit2 sender TMA bulk store instead of st.v4, bit3 poll nanosleep, bit4 no output staging, bit5 reset with st.v4 instead of TMA (L2), bit7 disables L2 2-warp row teams (default on)),
// tune[3] = LT rx stages per decoder warp (0 -> 2 if it fits, else 1).
#include "fp4ar_api.h"
#include "fp4_codec.cuh"
#include "codec_fast.cuh"
#include <cstdio>
#include <algorithm>

#ifndef LDBG_NOENC
#define LDBG_NOENC 0
#endif
#ifndef LDBG_NODEC
#define LDBG_NODEC 0
#endif
#if defined(FP4AR_LT_HW) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1070
#define LT_HW 1
#endif
#ifndef LDBG_NORESET
#define LDBG_NORESET 0
#endif
#ifndef LDBG_NOWORK
#define LDBG_NOWORK 0
#endif
#ifndef LAMPORT_LT
#define LAMPORT_LT 0
#endif

namespace {
using namespace fp4c;

constexpr int MB = 16;                 // meta word: row scale x4
constexpr int ROWB = VB + SB + MB;     // 3472 wire bytes per row
constexpr int NE = 4;                  // encoder warps (one warpgroup)
constexpr int ND = 8;                  // decoder warps (two warpgroups)
#ifndef LNEWG
#define LNEWG 2
#endif
constexpr int NEWG = LNEWG;            // encoder warpgroups (one-shot: alternate rows; two-shot: WG 0 only)
constexpr int NT = 128 * (1 + NEWG) + 32 * ND;
constexpr int W_LOAD = 0, W_SEND = 1, W_RESET = 2, W_FETCH = 3, W_ENC0 = 4, W_DEC0 = 4 + 4 * NEWG;
constexpr int REG_DEC = NEWG == 1 ? 168 : 120;  // decoder setmaxnreg; role sum must fit the launch allocation (NT x launch regs)
constexpr int NSIN = 4, NSENC_MAX = 16, NSUM = 2;
constexpr int INB = H * 2;             // 12288 B BF16 row
constexpr int SUMB = H * 4;            // 24576 B FP32 row
constexpr int PAGE = 4096;             // sentinel pattern page per region
enum { F_NOPDL = 1, F_TRUST = 2, F_TMAST = 4, F_SLEEP = 8, F_NOSTAGE = 16, F_RSTV4 = 32 };
enum { K_OS = 0, K_TSA = 1, K_TSB = 2 };
enum { SRC_G = 0, SRC_SV = 1, SRC_ST = 2 };  // global poll / smem verified / smem trusted

struct LP {
  const __nv_bfloat16* x; __nv_bfloat16* out;
  int R, me, G, ds, spw, flags, nsenc, ostg, split2;
  const uint8_t* xq;                   // pre-quantized input rows of 3456 B (INPUT-FP4.md) or nullptr
  float gs[4];                         // per-rank static scales (Q)  // ostg: per-warp output staging bytes (0 = direct stores)
  int cap, capo;                       // slot row capacity: one-shot / two-shot RS (ceil(cap/NR))
  char* peer[4];
  size_t osOff, osBuf, tsOff, tsBuf, tsAg, qOff, qBuf;  // qOff/qBuf: Q one-shot receive buffers (own layout)
  unsigned* ctrl;                      // [0] call, [1] done, [2] prev rows, [3] prev algo + 1 (0 = none)
};

// Slot array: V[nslot][cap][VB], S[nslot][cap][SB], M[nslot][cap][MB].
struct SA {
  size_t base; int nslot, cap;
  __device__ size_t v(int s, int r) const { return base + ((size_t)s * cap + r) * VB; }
  __device__ size_t sc(int s, int r) const { return base + (size_t)nslot * cap * VB + ((size_t)s * cap + r) * SB; }
  __device__ size_t m(int s, int r) const { return base + (size_t)nslot * cap * (VB + SB) + ((size_t)s * cap + r) * MB; }
};

// ---------------- PTX helpers ----------------
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint64_t* b, uint32_t n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sa(b)), "r"(n)); }
__device__ __forceinline__ void mb_init_v1(uint64_t* b) {
#if defined(LT_HW)
  asm volatile("mbarrier.init.layout::v1.shared.b64 [%0], 1;" ::"r"(sa(b)) : "memory");
#else
  mb_init(b, 1);
#endif
}
__device__ __forceinline__ void mb_arrive(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(sa(b)) : "memory"); }
__device__ __forceinline__ void mb_arrive_n(uint64_t* b, uint32_t n) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" ::"r"(sa(b)), "r"(n) : "memory"); }
__device__ __forceinline__ void mb_expect(uint64_t* b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa(b)), "r"(tx) : "memory"); }
__device__ __forceinline__ bool mb_try(uint64_t* b, uint32_t par) {
  uint32_t ok;
  asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(sa(b)), "r"(par) : "memory");
  return ok;
}
__device__ __forceinline__ bool mb_test(uint64_t* b, uint32_t par) {
  uint32_t ok;
  asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(sa(b)), "r"(par) : "memory");
  return ok;
}
__device__ __forceinline__ void mb_wait(uint64_t* b, uint32_t par) { while (!mb_try(b, par)) {} }
// Test-wait that also returns the TMA validity report (true = some loaded element was a sentinel).
__device__ __forceinline__ bool mb_test_report(uint64_t* b, uint32_t par, bool& bad) {
#if defined(LT_HW)
  uint32_t ok, rep;
  asm volatile("{\n.reg .pred p, q;\nmbarrier.test_wait.parity.phase_type::primary.acquire.cta.shared::cta.b64 p|q, [%2], %3;\n"
               "selp.u32 %0, 1, 0, p;\nselp.u32 %1, 1, 0, q;\n}" : "=r"(ok), "=r"(rep) : "r"(sa(b)), "r"(par) : "memory");
  bad = rep != 0; return ok;
#else
  bad = false; return mb_test(b, par);
#endif
}
__device__ __forceinline__ void tma_ld(void* dst, const void* src, uint32_t n, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sa(dst)), "l"(src), "r"(n), "r"(sa(b)) : "memory");
}
// Validity-reporting loads (Rubin, PTX ISA 9.4); plain loads otherwise (decoders' sw verify keeps it correct).
template <int KIND>
__device__ __forceinline__ void tma_ld_valid(void* dst, const void* src, uint32_t n, uint64_t* b) {
#if defined(LT_HW)
  if (KIND == 0) asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_16bytes::8 [%0], [%1], %2, [%3];" ::"r"(sa(dst)), "l"(src), "r"(n), "r"(sa(b)) : "memory");
  else if (KIND == 1) asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_element::ff [%0], [%1], %2, [%3];" ::"r"(sa(dst)), "l"(src), "r"(n), "r"(sa(b)) : "memory");
  else asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_16bytes::80000000 [%0], [%1], %2, [%3];" ::"r"(sa(dst)), "l"(src), "r"(n), "r"(sa(b)) : "memory");
#else
  tma_ld(dst, src, n, b);
#endif
}
__device__ __forceinline__ void tma_st(void* gdst, const void* src, uint32_t n) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(gdst), "r"(sa(src)), "r"(n) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ void bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ uint32_t ldv32(const void* p) { uint32_t v; asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }
__device__ __forceinline__ uint32_t ldv16(const void* p) { uint16_t v; asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(v) : "l"(p)); return v; }
__device__ __forceinline__ uint4 ldv128(const void* p) {
  uint4 r; asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p)); return r;
}
__device__ __forceinline__ void stg_v4(void* p, uint4 v) { asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)); }
// Nonzero iff some nibble of w is 0x8 (E2M1 -0 = sentinel).
__device__ __forceinline__ uint32_t negz(uint32_t w) { return w & ~((w & 0x77777777u) + 0x77777777u) & 0x88888888u; }
__device__ __forceinline__ bool bad_q(uint4 q) { return (negz(q.x) | negz(q.y) | negz(q.z) | negz(q.w)) != 0; }
__device__ __forceinline__ bool bad_s(uint32_t c) { return (c & 0xFFu) == 0xFFu || (c >> 8) == 0xFFu; }
__device__ __forceinline__ int sumoff(int g, int h) { return (g >> 7) * 4096 + h * 2048 + (g & 127) * 16; }
// rn(x / 6) for x >= 0, bit-identical to __fdiv_rn(x, 6.f) (codec lane, exhaustively verified).
__device__ __forceinline__ float div6_rn(float x) {
  const float r6 = 0x1.555556p-3f;
  const float q0 = __fmul_rn(x, r6);
  const float q1 = __fmaf_rn(__fmaf_rn(-q0, 6.f, x), r6, q0);
  return x == __int_as_float(0x7f800000) ? x : q1;
}

// Per-CTA work lists (identical on every rank).
struct Work {
  int r0, nrows, nB, rb0;
  __device__ void init(const LP& p, int NR, int ALGO) {
    const int b = blockIdx.x;
    r0 = (int)((long long)b * p.R / p.G); nrows = (int)((long long)(b + 1) * p.R / p.G) - r0;
    nB = 0; rb0 = 0;
    if (ALGO == FP4AR_TWOSHOT) {
      rb0 = r0 + ((p.me - r0 % NR) + NR) % NR;
      nB = rb0 < r0 + nrows ? (r0 + nrows - 1 - rb0) / NR + 1 : 0;
    }
  }
};

// ---------------- encoder: one row from smem (BF16 or FP32 sum) -> encoded stage (wire layout) ----------------
// Thread e (0..127) owns 8-value groups g = e + 128 i; a lane pair is one 16-value block, a quad one 16 B value word.
template <int MODE, bool F32>
__device__ __forceinline__ void encode_row(const char* src, char* st, float* red, int e, int bar) {
  const int lane = e & 31, w = e >> 5;
  float f[6][8];
  float am = 0.f;
  if (MODE != FP4AR_XPORT && !LDBG_NOENC) {
#pragma unroll
    for (int i = 0; i < 6; i++) {
      const int g = e + 128 * i;
      if (F32) {
        const float4 a = *(const float4*)(src + sumoff(g, 0)), b = *(const float4*)(src + sumoff(g, 1));
        f[i][0] = a.x; f[i][1] = a.y; f[i][2] = a.z; f[i][3] = a.w; f[i][4] = b.x; f[i][5] = b.y; f[i][6] = b.z; f[i][7] = b.w;
      } else {
        const uint4 t = *(const uint4*)(src + g * 16);
        const uint32_t u[4] = {t.x, t.y, t.z, t.w};
#pragma unroll
        for (int k = 0; k < 4; k++) { f[i][2 * k] = __uint_as_float(u[k] << 16); f[i][2 * k + 1] = __uint_as_float(u[k] & 0xffff0000u); }
      }
#pragma unroll
      for (int k = 0; k < 8; k++) am = fmaxf(am, fabsf(f[i][k]));
    }
    am = warp_amax(am);
    if (lane == 0) red[w] = am;
  }
  named_bar(bar, NE * 32);  // also marks the source stage as fully read
  float inv = 0.f, rs = 1.f;
  if (MODE != FP4AR_XPORT && !LDBG_NOENC) { am = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3])); row_scales(am, inv, rs); }
#pragma unroll
  for (int i = 0; i < 6; i++) {
    const int g = e + 128 * i;
    uint32_t q, s;
    if (MODE != FP4AR_XPORT && !LDBG_NOENC) {
      float bm = 0.f;
#pragma unroll
      for (int k = 0; k < 8; k++) bm = fmaxf(bm, fabsf(f[i][k]));
      bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1));  // 16-value block = lane pair
      const float sv = div6_rn(__fmul_rn(bm, inv));
      s = (uint32_t)__nv_cvt_float_to_fp8(sv, __NV_SATFINITE, __NV_E4M3) & 0x7Fu;
      const float sff = e4m3f(s);
      const float m = sff > 0.f ? __fdiv_rn(inv, sff) : 0.f;
      float gq[8];
#pragma unroll
      for (int k = 0; k < 8; k++) gq[k] = __fmul_rn(f[i][k], m);
      q = canon_neg0(e2m1x8(gq));
    } else { q = 0x11111111u; s = 0x38u; }
    ((uint32_t*)st)[g] = q;
    if (!(lane & 1)) ((uint8_t*)(st + VB))[g >> 1] = (uint8_t)s;
  }
  if (e < 4) ((float*)(st + VB + SB))[e] = rs;
}

// ---------------- decoder: one unit (row part) from NS sources, lane = one 32-value pair ----------------
// Strided source set: source s at v0 + s*sv (+16 p), s0 + s*ss (+2 p), m0 + s*sm. B = where data is read (global or smem rx),
// G = global copy for polling; source `me` (if >= 0) comes from trusted smem O. bkind: SRC_G, SRC_SV or SRC_ST.
struct Src { const char *v0, *s0, *m0; size_t sv, ss, sm;
  __device__ const char* v(int s) const { return v0 + s * sv; }
  __device__ const char* sc(int s) const { return s0 + s * ss; }
  __device__ const char* m(int s) const { return m0 + s * sm; } };
template <int NS, int MODE, int OUTK>
__device__ __forceinline__ void decode_unit(const Src B, const Src G, const Src O, int me, int bkind,
                                            int p0, int nit, int lane, bool sleep, __nv_bfloat16* orow, char* sumst, char* ost = nullptr) {
  constexpr int UB = NS >= 4 ? 3 : 6;
  float rs[NS];
#pragma unroll
  for (int s = 0; s < NS; s++) {
    const bool own = s == me;
    const int k = own ? SRC_ST : bkind;
    uint32_t b = k == SRC_G ? ldv32(B.m(s)) : *(const uint32_t*)(own ? O.m0 : B.m(s));
    if (MODE != FP4AR_COMPUTE && k != SRC_ST) while (b == SENT_RS) { if (sleep) __nanosleep(64); b = ldv32(G.m(s)); }
    rs[s] = __uint_as_float(b);
  }
#pragma unroll 1
  for (int b0 = 0; b0 < nit; b0 += UB) {
    uint4 v[UB][NS]; uint32_t c[UB][NS];
#pragma unroll
    for (int i = 0; i < UB; i++) {
      if (b0 + i >= nit) break;
      const int pp = p0 + (b0 + i) * 32 + lane;
#pragma unroll
      for (int s = 0; s < NS; s++) {
        const bool own = s == me;
        if (!own && bkind == SRC_G) { v[i][s] = ldv128(B.v(s) + 16 * pp); c[i][s] = ldv16(B.sc(s) + 2 * pp); }
        else { const char* pv = own ? O.v0 : B.v(s); const char* ps = own ? O.s0 : B.sc(s);
               v[i][s] = *(const uint4*)(pv + 16 * pp); c[i][s] = *(const uint16_t*)(ps + 2 * pp); }
      }
    }
    if (MODE != FP4AR_COMPUTE && bkind != SRC_ST) {
#pragma unroll
      for (int i = 0; i < UB; i++) {
        if (b0 + i >= nit) break;
        const int pp = p0 + (b0 + i) * 32 + lane;
#pragma unroll
        for (int s = 0; s < NS; s++) {
          if (s == me) continue;
          while (bad_q(v[i][s])) { if (sleep) __nanosleep(64); v[i][s] = ldv128(G.v(s) + 16 * pp); }
          while (bad_s(c[i][s])) { if (sleep) __nanosleep(64); c[i][s] = ldv16(G.sc(s) + 2 * pp); }
        }
      }
    }
    if (MODE == FP4AR_XPORT || LDBG_NODEC) continue;
#pragma unroll
    for (int i = 0; i < UB; i++) {
      if (b0 + i >= nit) break;
      const int pp = p0 + (b0 + i) * 32 + lane;
      if (OUTK == 0 && ost) {
        // staged: word kk = (k + lane/2) & 3 so 8 lanes of a phase hit 8 distinct 16 B bank groups
        float sc0[NS], sc1[NS];
#pragma unroll
        for (int s = 0; s < NS; s++) { sc0[s] = __fmul_rn(e4m3f(c[i][s] & 0xff), rs[s]); sc1[s] = __fmul_rn(e4m3f(c[i][s] >> 8), rs[s]); }
#pragma unroll
        for (int k = 0; k < 4; k++) {
          const int kk = (k + (lane >> 1)) & 3;
          float a8[8];
#pragma unroll
          for (int t = 0; t < 8; t++) a8[t] = 0.f;
#pragma unroll
          for (int s = 0; s < NS; s++) {
            const uint32_t w = kk == 0 ? v[i][s].x : kk == 1 ? v[i][s].y : kk == 2 ? v[i][s].z : v[i][s].w;
            const float sc = kk < 2 ? sc0[s] : sc1[s];
            __half2 h[4]; e2m1x8_dec(w, h);
#pragma unroll
            for (int t = 0; t < 4; t++) { const float2 f2 = __half22float2(h[t]); a8[2 * t] = __fmaf_rn(f2.x, sc, a8[2 * t]); a8[2 * t + 1] = __fmaf_rn(f2.y, sc, a8[2 * t + 1]); }
          }
          uint4 u; __nv_bfloat162* bb = (__nv_bfloat162*)&u;
#pragma unroll
          for (int t = 0; t < 4; t++) bb[t] = __floats2bfloat162_rn(a8[2 * t], a8[2 * t + 1]);
          *(uint4*)(ost + (pp - p0) * 64 + kk * 16) = u;
        }
        continue;
      }
      float acc[32];
      zero32(acc);
#pragma unroll
      for (int s = 0; s < NS; s++) dec_acc(acc, v[i][s], (uint16_t)c[i][s], rs[s]);
      if (OUTK == 0) {
        uint4 u[4]; __nv_bfloat162* bb = (__nv_bfloat162*)u;
#pragma unroll
        for (int k = 0; k < 16; k++) bb[k] = __floats2bfloat162_rn(acc[2 * k], acc[2 * k + 1]);
#pragma unroll
        for (int k = 0; k < 4; k++) stg_v4(orow + 32 * pp + 8 * k, u[k]);
      } else {
#pragma unroll
        for (int k = 0; k < 4; k++) {
          *(float4*)(sumst + sumoff(4 * pp + k, 0)) = make_float4(acc[8 * k], acc[8 * k + 1], acc[8 * k + 2], acc[8 * k + 3]);
          *(float4*)(sumst + sumoff(4 * pp + k, 1)) = make_float4(acc[8 * k + 4], acc[8 * k + 5], acc[8 * k + 6], acc[8 * k + 7]);
        }
      }
    }
  }
}


// Whole-row decode for L2: per lane pairs lane + 32 j; poll each peer word, decode with fp4f::dec_acc_fast, staged stores.
// Re-polls all not-yet-valid words of a group together (parallel loads), never one round trip per word.
#ifndef DEC_PIPE
#define DEC_PIPE 0
#endif
template <int NS, int MODE, int JN = 6>
__device__ __forceinline__ void decode_row2(const Src G, const Src O, int me, int lane, __nv_bfloat16* out, uint4* stg, int jb = 0, const float* rsq = nullptr) {
  constexpr int U = JN == 3 ? 3 : NS >= 4 ? 2 : 3;
  float rs[NS];
#pragma unroll
  for (int s = 0; s < NS; s++) {
    if (rsq) { rs[s] = rsq[s]; continue; }
    uint32_t b = s == me ? *(const uint32_t*)O.m0 : ldv32(G.m(s));
    if (MODE != FP4AR_COMPUTE && s != me) while (b == SENT_RS) b = ldv32(G.m(s));
    rs[s] = __uint_as_float(b);
  }
#pragma unroll 1
  for (int j0 = jb; j0 < jb + JN; j0 += U) {
    uint4 v[U][NS]; uint32_t c[U][NS];
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int pp = lane + 32 * (j0 + u);
#pragma unroll
      for (int s = 0; s < NS; s++) {
        if (s == me) { v[u][s] = ((const uint4*)O.v0)[pp]; c[u][s] = ((const uint16_t*)O.s0)[pp]; }
        else { v[u][s] = ldv128(G.v(s) + 16 * pp); c[u][s] = ldv16(G.sc(s) + 2 * pp); }
      }
    }
    if (MODE != FP4AR_COMPUTE) {
      while (true) {
        bool ok = true;
#pragma unroll
        for (int u = 0; u < U; u++)
#pragma unroll
          for (int s = 0; s < NS; s++) if (s != me) ok &= !bad_q(v[u][s]) && !bad_s(c[u][s]);
        if (ok) break;
#pragma unroll
        for (int u = 0; u < U; u++) {
          const int pp = lane + 32 * (j0 + u);
#pragma unroll
          for (int s = 0; s < NS; s++) {
            if (s == me) continue;
            if (bad_q(v[u][s])) v[u][s] = ldv128(G.v(s) + 16 * pp);
            if (bad_s(c[u][s])) c[u][s] = ldv16(G.sc(s) + 2 * pp);
          }
        }
      }
    }
    if (MODE == FP4AR_XPORT) continue;
#pragma unroll
    for (int u = 0; u < U; u++) {
      float acc[32]; zero32(acc);
#pragma unroll
      for (int s = 0; s < NS; s++) fp4f::dec_acc_fast(acc, v[u][s], c[u][s], rs[s]);
      uint4 o4[4]; fp4f::pack_bf16x32(acc, o4);
      fp4f::store_block_staged(stg, lane, o4, (uint4*)out + 128 * (j0 + u));
    }
  }
}

// ---------------- the kernel ----------------
template <int NR, int MODE, int ALGO, bool LT>
__global__ void __launch_bounds__(NT, 1) lamport_kernel(LP p) {
  extern __shared__ __align__(1024) char sm[];
  __shared__ __align__(8) uint64_t in_full[NSIN], in_empty[NSIN], enc_full[NSENC_MAX], enc_empty[NSENC_MAX], sum_full[NSUM];
  __shared__ __align__(8) uint64_t rx_full[ND * 2], rx_empty[ND * 2], tb[ND * 2][NR];
  __shared__ int enc_desc[NSENC_MAX];
  __shared__ float red[NEWG][2][NE];
  __shared__ volatile int sum_consumed;
  __shared__ int choice;
  __shared__ unsigned s_call, s_prevR, s_prevA;

  constexpr bool OS = ALGO == FP4AR_ONESHOT;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ds = p.ds, spw = p.spw, NSENC = p.nsenc;
  const bool trust = p.flags & F_TRUST, sleep = p.flags & F_SLEEP;
  char* const s_in = sm;
  char* const s_sum = s_in + NSIN * INB;
  char* const s_enc = s_sum + (OS ? 0 : NSUM * SUMB);
  char* const s_pat = s_enc + NSENC * ROWB + 16;
  char* const s_rx = s_pat + 3 * PAGE;
  const int segV = VB / ds, segS = SB / ds, segB = segV + segS + MB;
  const int rxStride = NR * segB;
  char* const s_ost = s_rx + (LT ? ND * spw * rxStride : 0);  // per decoder warp: 2 x ostg

  // ---- prologue (overlaps the previous kernel under PDL) ----
  if (tid == 0) {
    for (int i = 0; i < NSIN; i++) { mb_init(&in_full[i], 1); mb_init(&in_empty[i], NE * 32); }
    // one-shot: decoders read the own contribution from the stage, so they also free it
    for (int i = 0; i < NSENC; i++) { mb_init(&enc_full[i], NE * 32); mb_init(&enc_empty[i], OS ? 1 + ds * 32 : 1); }
    for (int i = 0; i < NSUM; i++) mb_init(&sum_full[i], ds * 32);
    if (LT) for (int i = 0; i < ND * 2; i++) { mb_init(&rx_full[i], NR); mb_init(&rx_empty[i], 32); for (int s = 0; s < NR; s++) mb_init_v1(&tb[i][s]); }
    sum_consumed = 0;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == W_RESET) {
    for (int i = lane; i < PAGE / 4; i += 32) {
      ((uint32_t*)s_pat)[i] = SENT_VAL32; ((uint32_t*)(s_pat + PAGE))[i] = 0xFFFFFFFFu; ((uint32_t*)(s_pat + 2 * PAGE))[i] = SENT_RS;
    }
    fence_async_smem();
  }
  __syncthreads();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (tid == 0) { const unsigned* cc = p.ctrl + 4 * blockIdx.x; s_call = cc[0]; s_prevR = cc[2]; s_prevA = cc[3]; }
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const unsigned call = s_call; const int buf = call % 3, pbuf = (call + 2) % 3;
  Work wk; wk.init(p, NR, ALGO);
  const int nA = wk.nrows, nB = wk.nB, nE = nA + nB;
  const SA os{p.osOff + (size_t)buf * p.osBuf, NR, p.cap};
  const SA rsA{p.tsOff + (size_t)buf * p.tsBuf, NR, p.capo};
  const SA ag{p.tsOff + (size_t)buf * p.tsBuf + p.tsAg, 1, p.cap};
  char* const self = p.peer[p.me];
  const int nU = OS ? nA * ds : (nB + nA) * ds;

  if (warp < 4) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (warp == W_LOAD) {
      // ---- loader: BF16 rows -> input ring (TMA) ----
      if (lane == 0 && MODE != FP4AR_XPORT) {
        for (int i = 0; i < nA; i++) {
          const int st = i % NSIN;
          mb_wait(&in_empty[st], ((i / NSIN) & 1) ^ 1);
          mb_expect(&in_full[st], INB);
          tma_ld(s_in + st * INB, p.x + (size_t)(wk.r0 + i) * H, INB, &in_full[st]);
        }
      }
    } else if (warp == W_SEND) {
      // ---- sender: encoded stage -> peers' receive buffers (one-shot: peers only; own copy stays in smem) ----
      const bool stv4 = !(p.flags & F_TMAST);
      for (int i = 0; i < nE; i++) {
        const int st = i % NSENC;
        mb_wait(&enc_full[st], (i / NSENC) & 1);
        const int d = enc_desc[st], kind = d & 3, row = d >> 2;
        const char* src = s_enc + st * ROWB;
        int ndst = kind == K_TSA ? 1 : kind == K_OS ? NR - 1 : NR;
        if (MODE == FP4AR_COMPUTE) ndst = OS ? 0 : 1;
        for (int k = 0; k < ndst; k++) {
          const int r = MODE == FP4AR_COMPUTE ? p.me : (kind == K_TSA ? row % NR : (p.me + 1 + k) % NR);
          char* base = p.peer[r];
          size_t ov, osf, om;
          if (kind == K_OS) { ov = os.v(p.me, row); osf = os.sc(p.me, row); om = os.m(p.me, row); }
          else if (kind == K_TSA) { ov = rsA.v(p.me, row / NR); osf = rsA.sc(p.me, row / NR); om = rsA.m(p.me, row / NR); }
          else { ov = ag.v(0, row); osf = ag.sc(0, row); om = ag.m(0, row); }
          if (!stv4) {
            if (lane == 0) { tma_st(base + ov, src, VB); tma_st(base + osf, src + VB, SB); tma_st(base + om, src + VB + SB, MB); }
          } else {
#pragma unroll
            for (int o = lane; o < VB / 16; o += 32) stg_v4(base + ov + 16 * o, ((const uint4*)src)[o]);
            if (lane < SB / 16) stg_v4(base + osf + 16 * lane, ((const uint4*)(src + VB))[lane]);
            if (lane == 0) stg_v4(base + om, *(const uint4*)(src + VB + SB));
          }
        }
        if (!stv4) {
          if (lane == 0) { bulk_commit(); bulk_wait_read<1>(); if (i > 0) mb_arrive(&enc_empty[(i - 1) % NSENC]); }
        } else {
          __syncwarp();
          if (lane == 0) mb_arrive(&enc_empty[st]);
        }
      }
      if (!stv4 && lane == 0) { bulk_wait_read<0>(); if (nE > 0) mb_arrive(&enc_empty[(nE - 1) % NSENC]); bulk_wait_all(); }
    } else if (warp == W_RESET) {
      // ---- re-arm the receive buffer used by the previous call (sentinel fill, TMA stores from pattern pages) ----
      if (s_prevA != 0) {
        const int pR = (int)s_prevR, pA = (int)s_prevA - 1;
        size_t rb[3][NR + 1]; size_t rl[3][NR + 1]; int nreg = 0;
        if (pA == FP4AR_ONESHOT) {
          const SA q{p.osOff + (size_t)pbuf * p.osBuf, NR, p.cap};
          for (int s = 0; s < NR; s++) {  // own slot is never written in one-shot
            const size_t n = s == p.me ? 0 : (size_t)pR;
            rb[0][s] = q.v(s, 0); rb[1][s] = q.sc(s, 0); rb[2][s] = q.m(s, 0); rl[0][s] = n * VB; rl[1][s] = n * SB; rl[2][s] = n * MB;
          }
          nreg = NR;
        } else {
          const SA q{p.tsOff + (size_t)pbuf * p.tsBuf, NR, p.capo};
          const SA a{p.tsOff + (size_t)pbuf * p.tsBuf + p.tsAg, 1, p.cap};
          const int pRo = (pR + NR - 1) / NR;
          for (int s = 0; s < NR; s++) { rb[0][s] = q.v(s, 0); rb[1][s] = q.sc(s, 0); rb[2][s] = q.m(s, 0); rl[0][s] = (size_t)pRo * VB; rl[1][s] = (size_t)pRo * SB; rl[2][s] = (size_t)pRo * MB; }
          rb[0][NR] = a.v(0, 0); rb[1][NR] = a.sc(0, 0); rb[2][NR] = a.m(0, 0);
          rl[0][NR] = (size_t)pR * VB; rl[1][NR] = (size_t)pR * SB; rl[2][NR] = (size_t)pR * MB;
          nreg = NR + 1;
        }
        int j = 0;
        for (int t = 0; t < 3; t++)
          for (int s = 0; s < nreg; s++) {
            const size_t u = rl[t][s] / 16, lo = u * blockIdx.x / p.G * 16, hi = u * (blockIdx.x + 1) / p.G * 16;
            for (size_t o = lo; o < hi; o += PAGE, j++)
              if ((j & 31) == lane) tma_st(self + rb[t][s] + o, s_pat + t * PAGE, (uint32_t)min((size_t)PAGE, hi - o));
          }
        bulk_commit(); bulk_wait_all();
      }
    } else if (LT && warp == W_FETCH) {
      // ---- LT fetch: TMA local receive buffer -> per-decoder-warp rx stages; re-issue while a sentinel is reported ----
      // Lane s owns source s. A lane with nothing to load (own slot in one-shot) still arrives, after the stage is free.
      const int nstg = ND * spw;
      uint32_t ph = 0;  // tb phase bit per stage
      int issued = 0, done = 0;
      auto unit = [&](int u, int& stg, int& j, int& nsrc, const char*& v, const char*& s, const char*& m, int& ss) {
        const int w = u % ND; j = u / ND; stg = w * spw + j % spw;
        const int part = u % ds; int row; const char* vb; const char* sb; const char* mbp;
        if (OS) { row = wk.r0 + u / ds; nsrc = NR; vb = self + os.v(0, row); sb = self + os.sc(0, row); mbp = self + os.m(0, row); ss = 0; }
        else if (u < nB * ds) { row = wk.rb0 + (u / ds) * NR; nsrc = NR; vb = self + rsA.v(0, row / NR); sb = self + rsA.sc(0, row / NR); mbp = self + rsA.m(0, row / NR); ss = 1; }
        else { row = wk.r0 + (u - nB * ds) / ds; nsrc = 1; vb = self + ag.v(0, row); sb = self + ag.sc(0, row); mbp = self + ag.m(0, row); ss = 2; }
        v = vb + part * segV; s = sb + part * segS; m = mbp;
      };
      const bool idle_own = OS;  // own contribution comes from the encoded stage
      auto issue = [&](int u) {
        int stg, j, nsrc, ss; const char *v, *s, *m; unit(u, stg, j, nsrc, v, s, m, ss);
        if (lane >= nsrc || (idle_own && lane == p.me)) return;
        const int cap = ss == 1 ? p.capo : p.cap;
        const int src = MODE == FP4AR_COMPUTE ? (ss == 2 ? 0 : p.me) : lane;
        const size_t dv = (size_t)src * cap * VB, dsb = (size_t)src * cap * SB, dm = (size_t)src * cap * MB;
        char* d = s_rx + stg * rxStride + lane * segB;
        mb_expect(&tb[stg][lane], segB);
        tma_ld_valid<0>(d, v + dv, segV, &tb[stg][lane]);
        tma_ld_valid<1>(d + segV, s + dsb, segS, &tb[stg][lane]);
        tma_ld_valid<2>(d + segV + segS, m + dm, MB, &tb[stg][lane]);
      };
      while (done < nU) {
        if (issued < nU && issued - done < nstg) {
          int stg, j, nsrc, ss; const char *v, *s, *m; unit(issued, stg, j, nsrc, v, s, m, ss);
          if (mb_test(&rx_empty[stg], ((j / spw) & 1) ^ 1)) { issue(issued); issued++; continue; }
        }
        if (done < issued) {
          int stg, j, nsrc, ss; const char *v, *s, *m; unit(done, stg, j, nsrc, v, s, m, ss);
          if (lane < nsrc) {
            if (!(idle_own && lane == p.me)) {
              bool bad;
              if (!mb_test_report(&tb[stg][lane], (ph >> stg) & 1, bad)) continue;
              ph ^= 1u << stg;
              if (bad && MODE != FP4AR_COMPUTE) { if (sleep) __nanosleep(64); issue(done); continue; }
            }
            mb_arrive_n(&rx_full[stg], nsrc == 1 ? NR : 1);
          }
          done++;
        }
      }
    }
  } else if (warp < W_DEC0) {
    if (NEWG == 1) asm volatile("setmaxnreg.dec.sync.aligned.u32 120;");
    // ---- encoders: A rows (BF16 input ring) and, two-shot, B rows (FP32 sums) ----
    const int wg = (warp - W_ENC0) >> 2, e = tid - W_ENC0 * 32 - wg * 128;
    const int istep = OS ? NEWG : 1;
    int ia = OS ? wg : 0, ib = 0;
    for (int i = OS ? wg : (wg == 0 ? 0 : nE); i < nE; i += istep) {
      bool doB = false;
      if (!OS) {
        if (e == 0) choice = ib < nB && (ia == nA || mb_test(&sum_full[ib % NSUM], (ib / NSUM) & 1));
        named_bar(1 + NEWG, NE * 32);
        doB = choice;
      }
      const int st = i % NSENC;
      mb_wait(&enc_empty[st], ((i / NSENC) & 1) ^ 1);
      char* es = s_enc + st * ROWB;
      float* rd = red[wg][(i / istep) & 1];
      if (!doB) {
        const int s = ia % NSIN;
        if (MODE != FP4AR_XPORT) mb_wait(&in_full[s], (ia / NSIN) & 1);
        encode_row<MODE, false>(s_in + s * INB, es, rd, e, 1 + wg);
        if (MODE != FP4AR_XPORT) mb_arrive(&in_empty[s]);
        if (e == 0) enc_desc[st] = ((wk.r0 + ia) << 2) | (OS ? K_OS : K_TSA);
        ia += istep;
      } else {
        const int s = ib % NSUM;
        mb_wait(&sum_full[s], (ib / NSUM) & 1);
        encode_row<MODE, true>(s_sum + s * SUMB, es, rd, e, 1);
        if (e == 0) { __threadfence_block(); sum_consumed = ib + 1; enc_desc[st] = ((wk.rb0 + ib * NR) << 2) | K_TSB; }
        ib++;
      }
      fence_async_smem();
      mb_arrive(&enc_full[st]);
    }
  } else {
    if (REG_DEC == 168) asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
    else asm volatile("setmaxnreg.inc.sync.aligned.u32 120;");
    // ---- decoders: units (row part) round-robin over decoder warps ----
    const int dw = warp - W_DEC0;
    const int pper = PAIRS / ds, nit = 6 / ds;
    const int vk = LT ? (trust ? SRC_ST : SRC_SV) : SRC_G;
    int j = 0;
    for (int u = dw; u < nU; u += ND, j++) {
      const int part = u % ds, p0 = part * pper;
      const char* rx = nullptr; const int stg = dw * spw + j % spw;
      // output staging (double buffer per warp); B units write FP32 sums instead
      char* ob = nullptr; __nv_bfloat16* ostRow = nullptr;
      if (p.ostg && MODE != FP4AR_XPORT && !LDBG_NODEC && (OS || u >= nB * ds)) {
        ob = s_ost + (dw * 2 + (j & 1)) * p.ostg;
        ostRow = p.out + (size_t)(OS ? wk.r0 + u / ds : wk.r0 + (u - nB * ds) / ds) * H + p0 * 32;
        if (lane == 0) bulk_wait_read<1>();
        __syncwarp();
      }
      if (LT) { mb_wait(&rx_full[stg], (j / spw) & 1); rx = s_rx + stg * rxStride; }  // stage holds only this part
      if (OS || u < nB * ds) {
        const int row = OS ? wk.r0 + u / ds : wk.rb0 + (u / ds) * NR;
        const int lr = OS ? row : row / NR;
        const SA& q = OS ? os : rsA;
        const int s0 = MODE == FP4AR_COMPUTE ? p.me : 0;
        Src G{self + q.v(s0, lr), self + q.sc(s0, lr), self + q.m(s0, lr), 0, 0, 0};
        if (MODE != FP4AR_COMPUTE) { G.sv = (size_t)q.cap * VB; G.ss = (size_t)q.cap * SB; G.sm = (size_t)q.cap * MB; }
        Src B = G;
        if (LT) B = Src{rx - 16 * p0, rx + segV - 2 * p0, rx + segV + segS, (size_t)segB, (size_t)segB, (size_t)segB};
        if (OS) {  // own contribution straight from the encoded stage
          const int i = u / ds, est = i % NSENC;
          const char* b = s_enc + est * ROWB;
          const Src O{b, b + VB, b + VB + SB, 0, 0, 0};
          mb_wait(&enc_full[est], (i / NSENC) & 1);
          decode_unit<NR, MODE, 0>(B, G, O, p.me, vk, p0, nit, lane, sleep, p.out + (size_t)row * H, nullptr, ob);
          mb_arrive(&enc_empty[est]);
        } else {
          const int k = u / ds, s = k % NSUM;
          while (sum_consumed < k - NSUM + 1) {}
          decode_unit<NR, MODE, 1>(B, G, G, -1, vk, p0, nit, lane, sleep, nullptr, s_sum + s * SUMB);
          mb_arrive(&sum_full[s]);
        }
      } else {
        const int row = wk.r0 + (u - nB * ds) / ds;
        const Src G{self + ag.v(0, row), self + ag.sc(0, row), self + ag.m(0, row), 0, 0, 0};
        Src B = G;
        if (LT) B = Src{rx - 16 * p0, rx + segV - 2 * p0, rx + segV + segS, 0, 0, 0};
        decode_unit<1, MODE, 0>(B, G, G, -1, vk, p0, nit, lane, sleep, p.out + (size_t)row * H, nullptr, ob);
      }
      if (LT) mb_arrive(&rx_empty[stg]);
      if (ob) {  // staged output row part -> global with one bulk store
        fence_async_smem(); __syncwarp();
        if (lane == 0) { tma_st(ostRow, ob, p.ostg); bulk_commit(); }
      }
    }
    if (lane == 0) bulk_wait_all();
  }
  // per-CTA call state (every kernel of a context uses the same grid), visible to the next kernel at completion
  if (tid == 0) { unsigned* cc = p.ctrl + 4 * blockIdx.x; cc[2] = p.R; cc[3] = ALGO + 1; cc[0] = call + 1; }
}


// ================= L2: one-shot, warp-per-row pipeline on the codec lane's row functions =================
// Each worker warp owns rows r0 + w + NW2*k of its CTA's chunk: TMA x -> smem, fp4f::enc_row_smem into an own-copy ring
// (LAG2+1 rows), push to peers with st.v4, then (LAG2 rows later) poll + decode-sum with smem-staged stores.
#ifndef NW2
#define NW2 10
#endif
#ifndef LAG2
#define LAG2 1
#endif
constexpr int ENCB2 = ROWB;
constexpr int QROW = VB + SB;  // pre-quantized input / Q wire row (no row scale)                                    // own encoded row (wire layout)
constexpr int WSM2 = INB + (LAG2 + 1) * ENCB2 + 2048;          // per worker warp: x row, own ring, store staging
constexpr size_t SMEM2 = 3 * PAGE + 64 + (size_t)NW2 * WSM2;

// Re-arm this CTA's share of the receive buffer used by the previous call (TMA bulk stores of sentinel pages).
template <int NR>
__device__ __forceinline__ void reset_prev(const LP& p, char* self, const char* s_pat, int lane, int pA, int pR, int pbuf) {
  int j = 0;
  auto range = [&](size_t base, size_t bytes, int t) {
    const size_t u = bytes / 16, lo = u * blockIdx.x / p.G * 16, hi = u * (blockIdx.x + 1) / p.G * 16;
    if (p.flags & F_RSTV4) {
      const uint32_t w = t == 0 ? SENT_VAL32 : t == 1 ? 0xFFFFFFFFu : SENT_RS;
      for (size_t o = lo + 16 * lane; o < hi; o += 512) stg_v4(self + base + o, make_uint4(w, w, w, w));
      return;
    }
    for (size_t o = lo; o < hi; o += PAGE, j++)
      if ((j & 31) == lane) tma_st(self + base + o, s_pat + t * PAGE, (uint32_t)min((size_t)PAGE, hi - o));
  };
  if (pA == 2) {  // Q one-shot: interleaved 3456 B rows; values -> 0x88, scales -> 0xFF
    const size_t qslot = (size_t)p.cap * QROW;
    for (int s = 0; s < NR; s++) {
      if (s == p.me) continue;
      const size_t base = p.qOff + (size_t)pbuf * p.qBuf + (size_t)s * qslot, u = (size_t)pR * QROW / 16;
      const size_t lo = u * blockIdx.x / p.G, hi = u * (blockIdx.x + 1) / p.G;
      for (size_t g = lo + lane; g < hi; g += 32) {
        const uint32_t w = (g % (QROW / 16)) < VB / 16 ? SENT_VAL32 : 0xFFFFFFFFu;
        stg_v4(self + base + 16 * g, make_uint4(w, w, w, w));
      }
    }
    return;
  }
  if (pA == FP4AR_ONESHOT) {
    const SA q{p.osOff + (size_t)pbuf * p.osBuf, NR, p.cap};
    for (int s = 0; s < NR; s++) if (s != p.me) { range(q.v(s, 0), (size_t)pR * VB, 0); range(q.sc(s, 0), (size_t)pR * SB, 1); range(q.m(s, 0), (size_t)pR * MB, 2); }
  } else {
    const SA q{p.tsOff + (size_t)pbuf * p.tsBuf, NR, p.capo};
    const SA a{p.tsOff + (size_t)pbuf * p.tsBuf + p.tsAg, 1, p.cap};
    const int pRo = (pR + NR - 1) / NR;
    for (int s = 0; s < NR; s++) { range(q.v(s, 0), (size_t)pRo * VB, 0); range(q.sc(s, 0), (size_t)pRo * SB, 1); range(q.m(s, 0), (size_t)pRo * MB, 2); }
    range(a.v(0, 0), (size_t)pR * VB, 0); range(a.sc(0, 0), (size_t)pR * SB, 1); range(a.m(0, 0), (size_t)pR * MB, 2);
  }
  bulk_commit(); bulk_wait_all();
}


// Half-row encode for 2-warp teams (small row counts): pairs lane + 32 j, j in [3h, 3h+3); same math as fp4f::enc_row_smem.
// Row amax is exchanged through s_am and a 64-thread named barrier.
__device__ __forceinline__ float enc_half_smem(const __nv_bfloat16* sx, int lane, int h, char* ob, float* s_am, int bar) {
  const uint4* s = (const uint4*)sx;
  const int f = (lane >> 1) & 3;
  uint32_t bmp[3]; float am = 0.f;
#pragma unroll
  for (int jj = 0; jj < 3; jj++) {
    const uint4* p = s + 4 * (lane + 32 * (3 * h + jj));
    __nv_bfloat162 a = fp4f::hzero2(), b = fp4f::hzero2();
    a = fp4f::habsmax8(fp4f::lds128(p + (0 ^ f)), a); a = fp4f::habsmax8(fp4f::lds128(p + (1 ^ f)), a);
    b = fp4f::habsmax8(fp4f::lds128(p + (2 ^ f)), b); b = fp4f::habsmax8(fp4f::lds128(p + (3 ^ f)), b);
    const uint32_t ua = *(uint32_t*)&a, ub = *(uint32_t*)&b;
    const uint32_t xa = max(ua & 0xffffu, ua >> 16), xb = max(ub & 0xffffu, ub >> 16);
    bmp[jj] = (f & 2) ? (xb | (xa << 16)) : (xa | (xb << 16));
    am = fmaxf(am, __uint_as_float(max(xa, xb) << 16));
  }
  am = warp_amax(am);
  if (lane == 0) s_am[h] = am;
  named_bar(bar, 64);
  am = fmaxf(s_am[0], s_am[1]);
  float inv, rs; row_scales(am, inv, rs);
#pragma unroll
  for (int jj = 0; jj < 3; jj++) {
    const int j = 3 * h + jj;
    const uint4* p = s + 4 * (lane + 32 * j);
    const float bm0 = __uint_as_float(bmp[jj] << 16), bm1 = __uint_as_float(bmp[jj] & 0xffff0000u);
    uint32_t sf2; float m0, m1; fp4f::pair_scales(bm0, bm1, inv, sf2, m0, m1);
    uint32_t w[4];
#pragma unroll
    for (int k = 0; k < 4; k++) w[k] = fp4f::enc_chunk_bf16<true>(fp4f::lds128(p + (k ^ f)), ((k ^ f) & 2) ? m1 : m0);
    uint32_t t0 = (f & 1) ? w[1] : w[0], t1 = (f & 1) ? w[0] : w[1], t2 = (f & 1) ? w[3] : w[2], t3 = (f & 1) ? w[2] : w[3];
    ((uint4*)ob)[lane + 32 * j] = (f & 2) ? make_uint4(t2, t3, t0, t1) : make_uint4(t0, t1, t2, t3);
    ((uint16_t*)(ob + VB))[lane + 32 * j] = (uint16_t)sf2;
  }
  return rs;
}

template <int NR, int MODE>
__global__ void __launch_bounds__(32 * (NW2 + 1), 1) lamport2_kernel(LP p) {
  extern __shared__ __align__(1024) char sm[];
  __shared__ __align__(8) uint64_t xb_full[NW2];
  __shared__ unsigned s_call, s_prevR, s_prevA;
  __shared__ float s_am[NW2 / 2][2];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  char* const s_pat = sm;
  char* const wsm = sm + 3 * PAGE + 64 + (warp < NW2 ? warp : 0) * WSM2;
  char* const xb = wsm; char* const own = wsm + INB; uint4* const stg = (uint4*)(own + (LAG2 + 1) * ENCB2);
  if (threadIdx.x == 0) { for (int i = 0; i < NW2; i++) mb_init(&xb_full[i], 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  if (warp == NW2) {
    for (int i = lane; i < PAGE / 4; i += 32) {
      ((uint32_t*)s_pat)[i] = SENT_VAL32; ((uint32_t*)(s_pat + PAGE))[i] = 0xFFFFFFFFu; ((uint32_t*)(s_pat + 2 * PAGE))[i] = SENT_RS;
    }
    fence_async_smem();
  }
  __syncthreads();
  const int r0 = (int)((long long)blockIdx.x * p.R / p.G), nrows = (int)((long long)(blockIdx.x + 1) * p.R / p.G) - r0;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (warp < NW2 && lane == 0) {  // first input row TMA before the call-state read (it does not depend on it)
    const bool team = p.xq || p.split2;
    const int t = team ? warp >> 1 : warp, nt = team ? NW2 / 2 : NW2;
    if ((!team || (warp & 1) == 0) && nrows > t) {
      char* dst = sm + 3 * PAGE + 64 + warp * WSM2;
      if (p.xq) { mb_expect(&xb_full[warp], QROW); tma_ld(dst, p.xq + (size_t)(r0 + t) * QROW, QROW, &xb_full[warp]); }
      else if (MODE != FP4AR_XPORT) { mb_expect(&xb_full[warp], INB); tma_ld(dst, p.x + (size_t)(r0 + t) * H, INB, &xb_full[warp]); }
    }
    (void)nt;
  }
  if (threadIdx.x == 0) { const unsigned* cc = p.ctrl + 4 * blockIdx.x; s_call = cc[0]; s_prevR = cc[2]; s_prevA = cc[3]; }
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const unsigned call = s_call; const int buf = call % 3, pbuf = (call + 2) % 3;
  char* const self = p.peer[p.me];
  const SA os{p.osOff + (size_t)buf * p.osBuf, NR, p.cap};
#ifdef LDBG_PRINT
  if (threadIdx.x == 0 && blockIdx.x < 2) printf("L2 blk %d me %d call %u R %d G %d nrows %d split2 %d xq %p flags %d cap %d\n", blockIdx.x, p.me, call, p.R, p.G, nrows, p.split2, p.xq, p.flags, p.cap);
#endif
  if (warp == NW2) {
    if (s_prevA != 0 && !LDBG_NORESET) reset_prev<NR>(p, self, s_pat, lane, (int)s_prevA - 1, (int)s_prevR, pbuf);
  } else if (p.xq) {
    // ---- pre-quantized input (L2q): rows of 3456 B (values + block scales), static per-rank gs; no encode, no row scale on
    // the wire. Receive slots hold interleaved 3456 B rows. 2-warp teams; the leader TMA-loads row k+1 into a 3-row ring while
    // row k is pushed (one TMA store per peer) and row k-1 is decoded (own row from the ring).
    constexpr int NTM = NW2 / 2, NQ = 3;
    const int t = warp >> 1, h = warp & 1, L = 2 * t;
    char* const ring = sm + 3 * PAGE + 64 + L * WSM2;
    uint4* const qstg = (uint4*)(sm + 3 * PAGE + 64 + warp * WSM2 + INB + (LAG2 + 1) * ENCB2);
    const int n = nrows > t ? (nrows - t + NTM - 1) / NTM : 0;
    const bool tmast = !(p.flags & F_TMAST);
    const size_t qslot = (size_t)p.cap * QROW;  // receive slot stride
    char* const qbuf = self + p.qOff + (size_t)buf * p.qBuf;
    auto rowk = [&](int k) { return r0 + t + NTM * k; };
    auto load = [&](int k) {
      mb_expect(&xb_full[L], QROW);
      tma_ld(ring + (k % NQ) * ENCB2, p.xq + (size_t)rowk(k) * QROW, QROW, &xb_full[L]);
    };
    for (int k = 0; k < n + LAG2; k++) {
      if (k < n) {
        char* ob = ring + (k % NQ) * ENCB2;
        mb_wait(&xb_full[L], k & 1);
        if (h == 0 && MODE != FP4AR_COMPUTE) {
          const size_t off = p.qOff + (size_t)buf * p.qBuf + (size_t)p.me * qslot + (size_t)rowk(k) * QROW;
          if (tmast) {
            if (lane == 0) for (int d = 1; d < NR; d++) tma_st(p.peer[(p.me + d) % NR] + off, ob, QROW);
          } else {
            for (int d = 1; d < NR; d++)
              for (int o = lane; o < QROW / 16; o += 32) stg_v4(p.peer[(p.me + d) % NR] + off + 16 * o, ((const uint4*)ob)[o]);
          }
        }
        if (h == 0 && lane == 0) bulk_commit();
        named_bar(1 + t, 64);  // both warps past phase k and done with row k-2's slot
        if (h == 0 && lane == 0 && k + 1 < n) { bulk_wait_read<2>(); load(k + 1); }
      }
      const int kd = k - LAG2;
      if (kd >= 0 && kd < n) {
        const int row = rowk(kd);
        const char* ob = ring + (kd % NQ) * ENCB2;
        const int s0 = MODE == FP4AR_COMPUTE ? p.me : 0;
        const size_t st_ = MODE == FP4AR_COMPUTE ? 0 : qslot;
        const char* g0 = qbuf + (size_t)s0 * qslot + (size_t)row * QROW;
        const Src G{g0, g0 + VB, nullptr, st_, st_, 0};
        const Src O{ob, ob + VB, nullptr, 0, 0, 0};
        decode_row2<NR, MODE, 3>(G, O, p.me, lane, p.out + (size_t)row * H, qstg, 3 * h, p.gs);
      }
    }
    if (lane == 0) bulk_wait_all();
  } else if (p.split2) {
    // ---- 2-warp teams: each row is encoded and decoded by two warps (half the pairs each) ----
    constexpr int NTM = NW2 / 2;
    const int t = warp >> 1, h = warp & 1, L = 2 * t;
    char* const lxb = sm + 3 * PAGE + 64 + L * WSM2; char* const lown = lxb + INB;
    const int n = nrows > t ? (nrows - t + NTM - 1) / NTM : 0;
    const bool tmast = !(p.flags & F_TMAST);
    auto rowk = [&](int k) { return r0 + t + NTM * k; };
    for (int k = 0; k < n + LAG2; k++) {
      if (k < n) {
        char* ob = lown + (k % (LAG2 + 1)) * ENCB2;
        if (h == 0 && k > LAG2 && lane == 0) bulk_wait_read<LAG2>();  // leader's bulk stores of this slot done reading
        float rs = 1.f;
        if (MODE != FP4AR_XPORT) {
          mb_wait(&xb_full[L], k & 1);
          rs = enc_half_smem((const __nv_bfloat16*)lxb, lane, h, ob, s_am[t], 1 + t);
        } else {
          named_bar(1 + t, 64);
#pragma unroll
          for (int jj = 0; jj < 3; jj++) ((uint4*)ob)[lane + 32 * (3 * h + jj)] = make_uint4(0x11111111u, 0x11111111u, 0x11111111u, 0x11111111u);
          if (lane < 12) ((uint4*)(ob + VB))[12 * h + lane] = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
        }
        if (h == 0 && lane < 4) ((float*)(ob + VB + SB))[lane] = rs;
        fence_async_smem();
        named_bar(1 + t, 64);  // both halves written, x row fully read
        if (h == 0 && lane == 0) {
          if (k + 1 < n && MODE != FP4AR_XPORT) { mb_expect(&xb_full[L], INB); tma_ld(lxb, p.x + (size_t)rowk(k + 1) * H, INB, &xb_full[L]); }
          if (MODE != FP4AR_COMPUTE) {
            const int row = rowk(k);
            for (int d = 1; d < NR; d++) {
              char* base = p.peer[(p.me + d) % NR];
              if (tmast) { tma_st(base + os.v(p.me, row), ob, VB); tma_st(base + os.sc(p.me, row), ob + VB, SB); tma_st(base + os.m(p.me, row), ob + VB + SB, MB); }
            }
          }
          bulk_commit();
        }
        if (MODE != FP4AR_COMPUTE && !tmast) {
          const int row = rowk(k);
          for (int d = 1; d < NR; d++) {
            char* base = p.peer[(p.me + d) % NR];
#pragma unroll
            for (int jj = 0; jj < 3; jj++) { const int pp = lane + 32 * (3 * h + jj); stg_v4(base + os.v(p.me, row) + 16 * pp, ((const uint4*)ob)[pp]); }
            if (lane < 12) stg_v4(base + os.sc(p.me, row) + 16 * (12 * h + lane), ((const uint4*)(ob + VB))[12 * h + lane]);
            if (h == 0 && lane == 0) stg_v4(base + os.m(p.me, row), *(const uint4*)(ob + VB + SB));
          }
        }
      }
      const int kd = k - LAG2;
      if (kd >= 0 && kd < n) {
        const int row = rowk(kd);
        const char* ob = lown + (kd % (LAG2 + 1)) * ENCB2;
        const int s0 = MODE == FP4AR_COMPUTE ? p.me : 0;
        const Src G{self + os.v(s0, row), self + os.sc(s0, row), self + os.m(s0, row),
                    MODE == FP4AR_COMPUTE ? 0 : (size_t)p.cap * VB, MODE == FP4AR_COMPUTE ? 0 : (size_t)p.cap * SB, MODE == FP4AR_COMPUTE ? 0 : (size_t)p.cap * MB};
        const Src O{ob, ob + VB, ob + VB + SB, 0, 0, 0};
        decode_row2<NR, MODE, 3>(G, O, p.me, lane, p.out + (size_t)row * H, stg, 3 * h);
      }
    }
    if (lane == 0) bulk_wait_all();
  } else {
    const int n = LDBG_NOWORK ? 0 : nrows > warp ? (nrows - warp + NW2 - 1) / NW2 : 0;
    const bool tmast = !(p.flags & F_TMAST);  // L2: TMA bulk-store egress by default; bit2 selects st.v4
    auto rowk = [&](int k) { return r0 + warp + NW2 * k; };
    for (int k = 0; k < n + LAG2; k++) {
      if (k < n) {
        char* ob = own + (k % (LAG2 + 1)) * ENCB2;
        if (tmast && k > LAG2) { if (lane == 0) bulk_wait_read<LAG2>(); __syncwarp(); }  // ring slot's bulk stores done reading
        float rs = 1.f;
        if (MODE != FP4AR_XPORT) {
          mb_wait(&xb_full[warp], k & 1);
          rs = fp4f::enc_row_smem<true>((const __nv_bfloat16*)xb, lane, fp4f::EncRow{(uint4*)ob, (uint16_t*)(ob + VB), nullptr});
          fence_async_smem(); __syncwarp();
          if (k + 1 < n && lane == 0) { mb_expect(&xb_full[warp], INB); tma_ld(xb, p.x + (size_t)rowk(k + 1) * H, INB, &xb_full[warp]); }
        } else {
#pragma unroll
          for (int j = 0; j < 6; j++) ((uint4*)ob)[lane + 32 * j] = make_uint4(0x11111111u, 0x11111111u, 0x11111111u, 0x11111111u);
          if (lane < 24) ((uint4*)(ob + VB))[lane] = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
        }
        if (lane < 4) ((float*)(ob + VB + SB))[lane] = rs;
        if (tmast) fence_async_smem();
        __syncwarp();
        if (MODE != FP4AR_COMPUTE && tmast) {  // push via TMA bulk stores (one commit group per row)
          if (lane == 0) {
            const int row = rowk(k);
            for (int d = 1; d < NR; d++) {
              char* base = p.peer[(p.me + d) % NR];
              tma_st(base + os.v(p.me, row), ob, VB); tma_st(base + os.sc(p.me, row), ob + VB, SB); tma_st(base + os.m(p.me, row), ob + VB + SB, MB);
            }
          }
        }
        if (lane == 0 && tmast) bulk_commit();
        if (MODE != FP4AR_COMPUTE && !tmast) {  // push own row to every peer (weak 16 B stores, no fence, no flag)
          const int row = rowk(k);
#pragma unroll
          for (int d = 1; d < NR; d++) {
            char* base = p.peer[(p.me + d) % NR];
            char* dv = base + os.v(p.me, row);
#pragma unroll
            for (int j = 0; j < 6; j++) stg_v4(dv + 16 * (lane + 32 * j), ((const uint4*)ob)[lane + 32 * j]);
            if (lane < 24) stg_v4(base + os.sc(p.me, row) + 16 * lane, ((const uint4*)(ob + VB))[lane]);
            if (lane == 0) stg_v4(base + os.m(p.me, row), *(const uint4*)(ob + VB + SB));
          }
        }
      }
      const int kd = k - LAG2;  // decode row k - LAG2 after pushing row k
      if (kd >= 0 && kd < n) {
        const int row = rowk(kd);
        const char* ob = own + (kd % (LAG2 + 1)) * ENCB2;
        const int s0 = MODE == FP4AR_COMPUTE ? p.me : 0;
        const Src G{self + os.v(s0, row), self + os.sc(s0, row), self + os.m(s0, row),
                    MODE == FP4AR_COMPUTE ? 0 : (size_t)p.cap * VB, MODE == FP4AR_COMPUTE ? 0 : (size_t)p.cap * SB, MODE == FP4AR_COMPUTE ? 0 : (size_t)p.cap * MB};
        const Src O{ob, ob + VB, ob + VB + SB, 0, 0, 0};
        decode_row2<NR, MODE>(G, O, p.me, lane, p.out + (size_t)row * H, stg);
      }
    }
    if (lane == 0) bulk_wait_all();
  }
  if (threadIdx.x == 0) { unsigned* cc = p.ctrl + 4 * blockIdx.x; cc[2] = p.R; cc[3] = (p.xq ? 2 : FP4AR_ONESHOT) + 1; cc[0] = call + 1; }
}

// ---------------- host ----------------
constexpr size_t CTRL_BYTES = 16 * 4096;  // per-CTA call state [4 x u32] for up to 4096 CTAs
struct Priv { unsigned* ctrl; int max_rows, cap, capo; size_t osOff, osBuf, tsOff, tsBuf, tsAg, inOff, qOff, qBuf; };

size_t layout(int max_rows, int nranks, Priv* q) {
  const int cap = max_rows, capo = (max_rows + nranks - 1) / nranks;
  auto al = [](size_t x) { return (x + 4095) / 4096 * 4096; };
  const size_t osBuf = al((size_t)nranks * cap * ROWB), tsAg = (size_t)nranks * capo * ROWB, tsBuf = al(tsAg + (size_t)cap * ROWB);
  const size_t qBuf = al((size_t)nranks * cap * QROW), inOff = 3 * osBuf + 3 * tsBuf;
  const size_t qOff = inOff + al((size_t)max_rows * QROW);
  if (q) { q->cap = cap; q->capo = capo; q->osOff = 0; q->osBuf = osBuf; q->tsOff = 3 * osBuf; q->tsBuf = tsBuf; q->tsAg = tsAg; q->inOff = inOff; q->qOff = qOff; q->qBuf = qBuf; }
  return (qOff + 3 * qBuf + (1 << 21) - 1) / (1 << 21) * (1 << 21);  // os x3, ts x3, input slot (window mode), Q x3
}
size_t win_bytes(int max_rows, int hidden, int nranks) { (void)hidden; return layout(max_rows, nranks, nullptr); }

__global__ void fill_sentinel(char* base, int nslot, int cap) {
  const size_t nv = (size_t)nslot * cap * VB / 4, ns = (size_t)nslot * cap * SB / 4, nm = (size_t)nslot * cap * MB / 4;
  uint32_t* w = (uint32_t*)base;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nv + ns + nm; i += (size_t)gridDim.x * blockDim.x)
    w[i] = i < nv ? SENT_VAL32 : i < nv + ns ? 0xFFFFFFFFu : SENT_RS;
}

__global__ void fill_sentinel_q(char* base, size_t rows) {
  uint32_t* w = (uint32_t*)base;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < rows * QROW / 4; i += (size_t)gridDim.x * blockDim.x)
    w[i] = (i % (QROW / 4)) < VB / 4 ? SENT_VAL32 : 0xFFFFFFFFu;
}
template <int NR, int MODE, int ALGO, bool LT> const void* kfn() { return (const void*)lamport_kernel<NR, MODE, ALGO, LT>; }
template <int NR, bool LT> const void* pick(int mode, int algo) {
  if (algo == FP4AR_ONESHOT) return mode == 0 ? kfn<NR, 0, 0, LT>() : mode == 1 ? kfn<NR, 1, 0, LT>() : kfn<NR, 2, 0, LT>();
  return mode == 0 ? kfn<NR, 0, 1, LT>() : mode == 1 ? kfn<NR, 1, 1, LT>() : kfn<NR, 2, 1, LT>();
}
#ifndef LAMPORT_V2
#define LAMPORT_V2 0
#endif
template <int NR> const void* pick2(int mode) { return mode == 0 ? (const void*)lamport2_kernel<NR, 0> : mode == 1 ? (const void*)lamport2_kernel<NR, 1> : (const void*)lamport2_kernel<NR, 2>; }
bool use_v2(int algo) { return LAMPORT_V2 && !LAMPORT_LT && algo == FP4AR_ONESHOT; }
const void* pickk(int nr, int mode, int algo) {
  if (use_v2(algo)) return nr == 2 ? pick2<2>(mode) : pick2<4>(mode); return nr == 2 ? pick<2, LAMPORT_LT>(mode, algo) : pick<4, LAMPORT_LT>(mode, algo); }
int nsenc_for(int algo) { return algo == FP4AR_ONESHOT ? NSENC_MAX : 4; }  // one-shot ring must be a multiple of ND
size_t smem_bytes(int nr, int algo, int ds, int spw, int ostg = 0) {
  size_t b = NSIN * INB + (algo == FP4AR_TWOSHOT ? NSUM * SUMB : 0) + (size_t)nsenc_for(algo) * ROWB + 16 + 3 * PAGE;
  if (LAMPORT_LT) b += (size_t)ND * spw * nr * (VB / ds + SB / ds + MB);
  b += (size_t)ND * 2 * ostg;
  return b;
}
constexpr size_t SMEM_MAX = 225 * 1024;  // leaves room for static smem

int init_impl(Fp4ArCtx* c, int max_rows, int hidden) {
  if (hidden != H || (c->nranks != 2 && c->nranks != 4)) { fprintf(stderr, "lamport: need hidden=6144, nranks 2|4\n"); return -1; }
  Priv* q = new Priv{}; q->max_rows = max_rows;
  layout(max_rows, c->nranks, q);
  if (cudaMalloc(&q->ctrl, CTRL_BYTES) != cudaSuccess) return -1;
  cudaMemset(q->ctrl, 0, CTRL_BYTES);
  char* self = c->peer[c->rank];
  for (int b = 0; b < 3; b++) {
    fill_sentinel<<<1024, 256>>>(self + q->osOff + b * q->osBuf, c->nranks, q->cap);
    fill_sentinel<<<1024, 256>>>(self + q->tsOff + b * q->tsBuf, c->nranks, q->capo);
    fill_sentinel<<<1024, 256>>>(self + q->tsOff + b * q->tsBuf + q->tsAg, 1, q->cap);
    fill_sentinel_q<<<1024, 256>>>(self + q->qOff + b * q->qBuf, (size_t)c->nranks * q->cap);
  }
  for (int m = 0; m < 3; m++)
    for (int a = 0; a < 2; a++)
      cudaFuncSetAttribute(pickk(c->nranks, m, a), cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_MAX);
  if (cudaDeviceSynchronize() != cudaSuccess) return -1;
  // all ranks' buffers armed before anyone pushes
  float* t; cudaMalloc(&t, 4);
  if (ncclAllReduce(t, t, 1, ncclFloat, ncclSum, c->comm, 0) != ncclSuccess) return -1;
  cudaDeviceSynchronize(); cudaFree(t);
  c->priv = q;
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int launch_impl(Fp4ArCtx* c, const Fp4ArArgs* a, cudaStream_t s) {
  Priv* q = (Priv*)c->priv;
  if (a->rows > q->max_rows || a->hidden != H) return -1;
  LP p{};
  p.x = a->x; p.out = a->out; p.xq = a->xq;
  for (int r = 0; r < 4; r++) p.gs[r] = a->gs[r];
  if (a->xq && !(LAMPORT_V2 && a->algo == FP4AR_ONESHOT)) return -1;  // pre-quantized input: L2q one-shot only
  p.R = a->rows; p.me = c->rank; p.flags = a->tune[2];
  p.G = a->tune[0] > 0 ? std::min(a->tune[0], c->sm_count) : c->sm_count;
  int ds = a->tune[1];
  if (ds != 1 && ds != 2 && ds != 3 && ds != 6) {
    const int rpc = std::max(1, (a->rows + p.G - 1) / p.G);
    ds = 3 * rpc >= ND ? 3 : 6;  // ds=1 measured 2x slower fused on GB200
  }
  int spw = a->tune[3] > 0 ? std::min(a->tune[3], 2) : 2;
  while (LAMPORT_LT && smem_bytes(c->nranks, a->algo, ds, spw) > SMEM_MAX) {
    if (spw > 1) spw--; else if (ds < 6) ds = ds == 1 ? 2 : ds == 2 ? 3 : 6; else return -1;
  }
  if (smem_bytes(c->nranks, a->algo, ds, spw) > SMEM_MAX) return -1;
  int ostg = (p.flags & F_NOSTAGE) || ds < 2 ? 0 : INB / ds;
  if (smem_bytes(c->nranks, a->algo, ds, spw, ostg) > SMEM_MAX) ostg = 0;
  p.ostg = ostg;
  {  // L2: 2-warp teams when a CTA has at most one row per team (tune c bit6 forces on, bit7 off)
    const int rpc = (a->rows + p.G - 1) / p.G;
    p.split2 = (p.flags & 128) ? 0 : 1; (void)rpc;  // teams won at every size on VR200 (it11)
  }
  p.ds = ds; p.spw = spw; p.nsenc = nsenc_for(a->algo);
  p.cap = q->cap; p.capo = q->capo;
  for (int r = 0; r < c->nranks; r++) p.peer[r] = c->peer[r];
  p.osOff = q->osOff; p.osBuf = q->osBuf; p.tsOff = q->tsOff; p.tsBuf = q->tsBuf; p.tsAg = q->tsAg; p.ctrl = q->ctrl; p.qOff = q->qOff; p.qBuf = q->qBuf;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(p.G); cfg.blockDim = dim3(NT); cfg.stream = s;
  cfg.dynamicSmemBytes = smem_bytes(c->nranks, a->algo, ds, spw, ostg);
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = (p.flags & F_NOPDL) ? 0 : 1;
  void* args[] = {&p};
  if (use_v2(a->algo)) { cfg.blockDim = dim3(32 * (NW2 + 1)); cfg.dynamicSmemBytes = SMEM2; }
  return cudaLaunchKernelExC(&cfg, pickk(c->nranks, a->mode, a->algo), args) == cudaSuccess ? 0 : -1;
}

uint8_t* input_slot_impl(Fp4ArCtx* c, int rows) { Priv* q = (Priv*)c->priv; return rows <= q->max_rows ? (uint8_t*)c->peer[c->rank] + q->inOff : nullptr; }
void fini_impl(Fp4ArCtx* c) { Priv* q = (Priv*)c->priv; if (q) { cudaFree(q->ctrl); delete q; c->priv = nullptr; } }
}  // namespace

#define LSTR2(a) #a
#define LSTR(a) LSTR2(a)
#ifdef FP4AR_NAME
static const Fp4ArVariant kVar = {LSTR(FP4AR_NAME), true, win_bytes, init_impl, launch_impl, fini_impl, LAMPORT_V2 ? input_slot_impl : nullptr};
const Fp4ArVariant* FP4AR_REGISTER() { return &kVar; }
#else
static const Fp4ArVariant kVar = {LAMPORT_LT ? "LT" : "L", true, win_bytes, init_impl, launch_impl, fini_impl};
#if LAMPORT_LT
const Fp4ArVariant* fp4ar_register_LT() { return &kVar; }
#else
const Fp4ArVariant* fp4ar_register_L() { return &kVar; }
#endif
#endif
