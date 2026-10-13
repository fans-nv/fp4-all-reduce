// fp4-allreduce: fused TP2 push AR + residual + Gemma RMSNorm (BF16/NVFP4/MXFP8 out, NVFP4 in); frozen build LN-t10.
// Lane NORM: L2 (lane L it10 one-shot push, warp-per-row) + fused residual add + Gemma RMSNorm epilogue, BF16 or NVFP4 normed out.
// Copied from kernels/lamport/snap/it10/lamport.cu (helpers, L2 kernel, host); spec and log: NOTES.md.
#include "fp4ar_api.h"
#include "fp4_codec.cuh"
#include "codec_fast.cuh"
#include "lnorm_epi.cuh"
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
  int R, me, G, ds, spw, flags, nsenc, ostg;  // ostg: per-warp output staging bytes (0 = direct stores)
  int cap, capo;                       // slot row capacity: one-shot / two-shot RS (ceil(cap/NR))
  char* peer[4];
  size_t osOff, osBuf, tsOff, tsBuf, tsAg;
  unsigned* ctrl;                      // [0] call, [1] done, [2] prev rows, [3] prev algo + 1 (0 = none)
  lne::NormArgs nm;                    // fused-norm epilogue (lane NORM)
  const uint8_t* xq; float gs[4];      // pre-quantized input (INPUT-FP4.md): 3456 B rows + per-rank static scales
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

struct Src { const char *v0, *s0, *m0; size_t sv, ss, sm;
  __device__ const char* v(int s) const { return v0 + s * sv; }
  __device__ const char* sc(int s) const { return s0 + s * ss; }
  __device__ const char* m(int s) const { return m0 + s * sm; } };

#ifndef DEC_PIPE
#define DEC_PIPE 0
#endif
template <int NS, int MODE>
__device__ __forceinline__ void decode_row2(const Src G, const Src O, int me, int lane, __nv_bfloat16* out, uint4* stg) {
  constexpr int U = NS >= 4 ? 2 : 3;
  float rs[NS];
#pragma unroll
  for (int s = 0; s < NS; s++) {
    uint32_t b = s == me ? *(const uint32_t*)O.m0 : ldv32(G.m(s));
    if (MODE != FP4AR_COMPUTE && s != me) while (b == SENT_RS) b = ldv32(G.m(s));
    rs[s] = __uint_as_float(b);
  }
#pragma unroll 1
  for (int j0 = 0; j0 < 6; j0 += U) {
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

// Fused-norm decode of one row: pass 1 polls + decode-sums the pairs (as decode_row2), adds the residual in FP32,
// accumulates the FP32 sum of squares, keeps bf16(z) in registers (6 pairs x 64 B per lane) and stores it as the new
// residual (staged). Pass 2: rinv, y = bf16(z) * rinv * (wbias + w) -> BF16 (staged) or NVFP4 (vLLM quant math).
#ifndef LNU
#define LNU 2
#endif
#ifndef LN_ABL
#define LN_ABL 0  // ablation bits: 1 no residual load, 2 no residual store, 4 no pass 2 (debug only, gate fails)
#endif
#ifndef LN_NTMA
#define LN_NTMA 0  // LN_TMAR: normed BF16 row also written in place into rb and TMA-stored
#endif
#ifndef LN_P1NAT
#define LN_P1NAT 0  // pass 1 rb access in natural order (conflicting) instead of XOR order + selects
#endif
#ifndef LN_PERM
#define LN_PERM 1  // epi 1 pass 2 in XOR chunk order (no selects)
#endif
#ifndef LN_EARLY
#define LN_EARLY 1
#endif
#ifndef LN_RCO
#define LN_RCO 1  // residual loaded coalesced (16 B per lane contiguous) and transposed in smem
#endif
#ifndef LN_RALL
#define LN_RALL 0
#endif
#ifndef LN_WSM
#define LN_WSM 1  // norm weight row in smem (the sentinel-page area; reset then uses st.v4, flag bit5)
#endif
#ifndef LN_PF
#define LN_PF 1   // L2 bulk prefetch of the residual row one pipeline step ahead
#endif
#ifndef LNW
#define LNW 3
#endif
template <int NS, int MODE, int EPI>
__device__ __forceinline__ void decode_row_norm(const Src G, const Src O, int me, int lane, const LP& p, int row, uint4* stg, const uint4* wsm) {
  constexpr int U = LNU;
  const uint4* rin = (const uint4*)(p.nm.res_in + (size_t)row * H);
  uint4 r[6][4];  // LN_RALL: whole residual row in flight at once; else per pair group (before that group's poll)
  auto ldres = [&](int j) {
#pragma unroll
    for (int k = 0; k < 4; k++) r[j][k] = LN_ABL & 1 ? make_uint4(0, 0, 0, 0) : fp4f::ld_nc_na(LN_RCO ? rin + 128 * j + lane + 32 * k : rin + 4 * (lane + 32 * j) + k);
  };
  if (LN_RALL) {
#pragma unroll
    for (int j = 0; j < 6; j++) ldres(j);
  }
  float rs[NS];
#pragma unroll
  for (int s = 0; s < NS; s++) {
    uint32_t b = s == me ? *(const uint32_t*)O.m0 : ldv32(G.m(s));
    if (MODE != FP4AR_COMPUTE && s != me) while (b == SENT_RS) b = ldv32(G.m(s));
    rs[s] = __uint_as_float(b);
  }
  uint4* rout = (uint4*)(p.nm.res_out + (size_t)row * H);
  uint4 hold[6][4];
  float2 ss2 = make_float2(0.f, 0.f);
#pragma unroll
  for (int j0 = 0; j0 < 6; j0 += U) {
    if (!LN_RALL) {
#pragma unroll
      for (int u = 0; u < U; u++) ldres(j0 + u);
    }
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
    if (LN_RCO) {  // coalesced residual block -> lane-owned pairs through the staging buffer (inverse of store_block_staged)
      const int f = (lane >> 1) & 3;
#pragma unroll
      for (int u = 0; u < U; u++) {
#pragma unroll
        for (int k = 0; k < 4; k++) { const int c = lane + 32 * k, l = c >> 2; stg[4 * l + ((c & 3) ^ ((l >> 1) & 3))] = r[j0 + u][k]; }
        __syncwarp();
#pragma unroll
        for (int k = 0; k < 4; k++) r[j0 + u][k] = stg[4 * lane + (k ^ f)];
        __syncwarp();
      }
    }
#pragma unroll
    for (int u = 0; u < U; u++) {
      float acc[32]; zero32(acc);
#pragma unroll
      for (int s = 0; s < NS; s++) fp4f::dec_acc_fast(acc, v[u][s], c[u][s], rs[s]);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const uint32_t ru[4] = {r[j0 + u][k].x, r[j0 + u][k].y, r[j0 + u][k].z, r[j0 + u][k].w};
#pragma unroll
        for (int i = 0; i < 4; i++) {  // packed: ss2.x sums even elements, ss2.y odd ones
          const float2 z = lne::add2(make_float2(acc[8 * k + 2 * i], acc[8 * k + 2 * i + 1]), lne::bf2(ru[i]));
          acc[8 * k + 2 * i] = z.x; acc[8 * k + 2 * i + 1] = z.y; ss2 = lne::fma2(z, z, ss2);
        }
      }
      fp4f::pack_bf16x32(acc, hold[j0 + u]);
      if (!(LN_ABL & 2)) fp4f::store_block_staged(stg, lane, hold[j0 + u], rout + 128 * (j0 + u));
    }
  }
  if (MODE == FP4AR_XPORT || (LN_ABL & 4)) return;
  float ss = __fadd_rn(ss2.x, ss2.y);
#pragma unroll
  for (int o = 16; o; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
  const float rinv = lne::norm_rinv(ss, p.nm.eps);
  const uint4* wp = (const uint4*)p.nm.w;
#pragma unroll
  for (int j0 = 0; j0 < 6; j0 += LNW) {
  uint4 wg[LNW][4];
  const int f = (lane >> 1) & 3;
#pragma unroll
  for (int u = 0; u < LNW; u++) {
#pragma unroll
    for (int k = 0; k < 4; k++)  // wg[u][k] = chunk k^f (smem XOR order, conflict-free), un-swizzled with selects
      wg[u][k] = LN_WSM ? fp4f::lds128(wsm + 4 * (lane + 32 * (j0 + u)) + (k ^ f)) : __ldg(wp + 4 * (lane + 32 * (j0 + u)) + (k ^ f));
    const uint4 t0 = (f & 1) ? wg[u][1] : wg[u][0], t1 = (f & 1) ? wg[u][0] : wg[u][1], t2 = (f & 1) ? wg[u][3] : wg[u][2], t3 = (f & 1) ? wg[u][2] : wg[u][3];
    wg[u][0] = (f & 2) ? t2 : t0; wg[u][1] = (f & 2) ? t3 : t1; wg[u][2] = (f & 2) ? t0 : t2; wg[u][3] = (f & 2) ? t1 : t3;
  }
#pragma unroll
  for (int u = 0; u < LNW; u++) {
    const int j = j0 + u, pp = lane + 32 * j;
    float y[32]; lne::norm_pair2(hold[j], wg[u], rinv, p.nm.wbias, y);
    if (EPI == 1) {
      uint4 o4[4]; fp4f::pack_bf16x32(y, o4);
      fp4f::store_block_staged(stg, lane, o4, (uint4*)(p.out + (size_t)row * H) + 128 * j);
    } else if (EPI == 3) {
      uint4 q[2]; const uint32_t e8 = lne::quant_pair_mxfp8(y, q);
      uint8_t* qr = p.nm.outq + (size_t)row * lne::XROW;
      ((uint4*)qr)[2 * pp] = q[0]; ((uint4*)qr)[2 * pp + 1] = q[1]; qr[H + pp] = (uint8_t)e8;
    } else {
      uint4 q; const uint32_t sf = lne::quant_pair_vllm(y, p.nm.qscale, q);
      uint8_t* qr = p.nm.outq + (size_t)row * lne::QROW;
      ((uint4*)qr)[pp] = q; ((uint16_t*)(qr + VB))[pp] = (uint16_t)sf;
    }
  }
  }
}

// out[k] = in[k ^ f] (XOR chunk permutation, self-inverse) with selects only (no dynamic register indexing).
__device__ __forceinline__ void xsel(const uint4* in, int f, uint4* out) {
  const uint4 t0 = (f & 1) ? in[1] : in[0], t1 = (f & 1) ? in[0] : in[1], t2 = (f & 1) ? in[3] : in[2], t3 = (f & 1) ? in[2] : in[3];
  out[0] = (f & 2) ? t2 : t0; out[1] = (f & 2) ? t3 : t1; out[2] = (f & 2) ? t0 : t2; out[3] = (f & 2) ? t1 : t3;
}
// LN_TMAR path: the residual row was TMA-loaded into rb (12 KB smem, issued one pipeline step ahead). Pass 1 writes
// bf16(z) back into rb (XOR-swizzled 16 B chunks), then lane 0 TMA-stores rb as res_out; pass 2 reads x = bf16(z) from rb.
template <int NS, int MODE, int EPI>
__device__ __forceinline__ void decode_row_norm_t(const Src G, const Src O, int me, int lane, const LP& p, int row, uint4* stg,
                                                  const uint4* wsm, uint4* rb, uint64_t* rbar, uint32_t par) {
  constexpr int U = LNU;
  const int f = (lane >> 1) & 3;
  float rs[NS];
#pragma unroll
  for (int s = 0; s < NS; s++) {
    uint32_t b = s == me ? *(const uint32_t*)O.m0 : ldv32(G.m(s));
    if (MODE != FP4AR_COMPUTE && s != me) while (b == SENT_RS) b = ldv32(G.m(s));
    rs[s] = __uint_as_float(b);
  }
  float2 ss2 = make_float2(0.f, 0.f);
  bool rwait = MODE != FP4AR_XPORT;
#pragma unroll 1
  for (int j0 = 0; j0 < 6; j0 += U) {
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
    if (rwait) { mb_wait(rbar, par); rwait = false; }
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int pp = lane + 32 * (j0 + u);
      float acc[32]; zero32(acc);
#pragma unroll
      for (int s = 0; s < NS; s++) fp4f::dec_acc_fast(acc, v[u][s], c[u][s], rs[s]);
      uint4 t[4], r[4];
      if (LN_P1NAT) {
#pragma unroll
        for (int k = 0; k < 4; k++) r[k] = fp4f::lds128(rb + 4 * pp + k);  // natural order: bank conflicts, no selects
      } else {
#pragma unroll
        for (int k = 0; k < 4; k++) t[k] = fp4f::lds128(rb + 4 * pp + (k ^ f));
        xsel(t, f, r);
      }
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const uint32_t ru[4] = {r[k].x, r[k].y, r[k].z, r[k].w};
#pragma unroll
        for (int i = 0; i < 4; i++) {  // packed: ss2.x sums even elements, ss2.y odd ones
          const float2 z = lne::add2(make_float2(acc[8 * k + 2 * i], acc[8 * k + 2 * i + 1]), lne::bf2(ru[i]));
          acc[8 * k + 2 * i] = z.x; acc[8 * k + 2 * i + 1] = z.y; ss2 = lne::fma2(z, z, ss2);
        }
      }
      uint4 o4[4]; fp4f::pack_bf16x32(acc, o4);
      if (LN_P1NAT) {
#pragma unroll
        for (int k = 0; k < 4; k++) rb[4 * pp + k] = o4[k];
      } else {
        xsel(o4, f, t);
#pragma unroll
        for (int k = 0; k < 4; k++) rb[4 * pp + (k ^ f)] = t[k];  // natural layout written in XOR order
      }
    }
  }
  if (MODE == FP4AR_XPORT) return;
  fence_async_smem(); __syncwarp();
  if (lane == 0) { tma_st(p.nm.res_out + (size_t)row * H, rb, INB); bulk_commit(); }
  float ss = __fadd_rn(ss2.x, ss2.y);
#pragma unroll
  for (int o = 16; o; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
  const float rinv = lne::norm_rinv(ss, p.nm.eps);
  if (EPI == 1 && LN_NTMA) { if (lane == 0) bulk_wait_read<0>(); __syncwarp(); }  // rb is rewritten with the normed row
#pragma unroll 2
  for (int j = 0; j < 6; j++) {
    const int pp = lane + 32 * j;
    uint4 t[4], x[4], w[4];
    if (EPI == 1 && !LN_NTMA && LN_PERM) {  // elementwise: keep chunks in XOR order (x and w alike), no un-swizzle
#pragma unroll
      for (int k = 0; k < 4; k++) { x[k] = fp4f::lds128(rb + 4 * pp + (k ^ f)); w[k] = fp4f::lds128(wsm + 4 * pp + (k ^ f)); }
      float y[32]; lne::norm_pair2(x, w, rinv, p.nm.wbias, y);
      uint4 o4[4]; fp4f::pack_bf16x32(y, o4);
#pragma unroll
      for (int k = 0; k < 4; k++) stg[4 * lane + (k ^ f)] = o4[k];  // o4[k] = chunk k^f: staging holds chunks in natural order
      __syncwarp();
      uint4* out = (uint4*)(p.out + (size_t)row * H) + 128 * j;
#pragma unroll
      for (int k = 0; k < 4; k++) out[lane + 32 * k] = stg[lane + 32 * k];
      __syncwarp();
      continue;
    }
#pragma unroll
    for (int k = 0; k < 4; k++) t[k] = fp4f::lds128(rb + 4 * pp + (k ^ f));
    xsel(t, f, x);
#pragma unroll
    for (int k = 0; k < 4; k++) t[k] = fp4f::lds128(wsm + 4 * pp + (k ^ f));
    xsel(t, f, w);
    float y[32]; lne::norm_pair2(x, w, rinv, p.nm.wbias, y);
    if (EPI == 1 && LN_NTMA) {
      uint4 o4[4]; fp4f::pack_bf16x32(y, o4); xsel(o4, f, t);
#pragma unroll
      for (int k = 0; k < 4; k++) rb[4 * pp + (k ^ f)] = t[k];
    } else if (EPI == 1) {
      uint4 o4[4]; fp4f::pack_bf16x32(y, o4);
      fp4f::store_block_staged(stg, lane, o4, (uint4*)(p.out + (size_t)row * H) + 128 * j);
    } else if (EPI == 3) {
      uint4 q[2]; const uint32_t e8 = lne::quant_pair_mxfp8(y, q);
      uint8_t* qr = p.nm.outq + (size_t)row * lne::XROW;
      ((uint4*)qr)[2 * pp] = q[0]; ((uint4*)qr)[2 * pp + 1] = q[1]; qr[H + pp] = (uint8_t)e8;
    } else {
      uint4 q; const uint32_t sf = lne::quant_pair_vllm(y, p.nm.qscale, q);
      uint8_t* qr = p.nm.outq + (size_t)row * lne::QROW;
      ((uint4*)qr)[pp] = q; ((uint16_t*)(qr + VB))[pp] = (uint16_t)sf;
    }
  }
  if (EPI == 1 && LN_NTMA) {
    fence_async_smem(); __syncwarp();
    if (lane == 0) { tma_st(p.out + (size_t)row * H, rb, INB); bulk_commit(); }
  }
}

// ================= L2: one-shot, warp-per-row pipeline on the codec lane's row functions =================
// Each worker warp owns rows r0 + w + NW2*k of its CTA's chunk: TMA x -> smem, fp4f::enc_row_smem into an own-copy ring
// (LAG2+1 rows), push to peers with st.v4, then (LAG2 rows later) poll + decode-sum with smem-staged stores.
#ifndef LN_TMAR
#define LN_TMAR 1  // residual row TMA-loaded into smem one step ahead; res_out TMA-stored from it
#endif
#ifndef LN_ALIAS
#define LN_ALIAS 1  // BF16 input: residual row TMA-loaded into the x buffer once x is encoded (10 workers instead of 6)
#endif
// Worker warps and per-warp smem per instantiation: 6 workers when a separate 12 KB residual buffer is needed.
__host__ __device__ constexpr bool sep_rb(int epi, bool q) { return epi > 0 && LN_TMAR && (q || !LN_ALIAS); }
__host__ __device__ constexpr int nwk(int epi, bool q) { return sep_rb(epi, q) ? 6 : 10; }
#ifndef LAG2
#define LAG2 1
#endif
constexpr int ENCB2 = ROWB;                                    // own encoded row (wire layout)
__host__ __device__ constexpr int wsmk(int epi, bool q) { return INB + (LAG2 + 1) * ENCB2 + 2048 + (sep_rb(epi, q) ? INB : 0); }  // x row, own ring, staging, [residual]
__host__ __device__ constexpr size_t smemk(int epi, bool q) { return 3 * PAGE + 64 + (size_t)nwk(epi, q) * wsmk(epi, q); }
constexpr size_t SMEM_MAXK = 3 * PAGE + 64 + (size_t)6 * (INB + (LAG2 + 1) * ENCB2 + 2048 + INB);

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

template <int NR, int MODE, int EPI, bool QIN>
__global__ void __launch_bounds__(32 * (nwk(EPI, QIN) + 1), 1) lnorm_kernel(LP p) {
  constexpr int NW2 = nwk(EPI, QIN), WSM2 = wsmk(EPI, QIN);
  constexpr bool AL = EPI > 0 && LN_TMAR && !sep_rb(EPI, QIN) && MODE != FP4AR_XPORT;  // residual aliases the x buffer
  extern __shared__ __align__(1024) char sm[];
  __shared__ __align__(8) uint64_t xb_full[NW2], rb_full[NW2];
  __shared__ unsigned s_call, s_prevR, s_prevA;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  char* const s_pat = sm;
  char* const wsm = sm + 3 * PAGE + 64 + (warp < NW2 ? warp : 0) * WSM2;
  char* const xb = wsm; char* const own = wsm + INB; uint4* const stg = (uint4*)(own + (LAG2 + 1) * ENCB2);
  uint4* const rb = AL ? (uint4*)xb : stg + 128;  // LN_TMAR residual row (12 KB)
  if (threadIdx.x == 0) { for (int i = 0; i < NW2; i++) { mb_init(&xb_full[i], 1); mb_init(&rb_full[i], 1); } asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  if (EPI > 0 && LN_WSM) {  // static weight row: safe to read before griddepcontrol.wait
    for (int i = threadIdx.x; i < H / 8; i += blockDim.x) ((uint4*)s_pat)[i] = __ldg((const uint4*)p.nm.w + i);
  } else if (warp == NW2) {
    for (int i = lane; i < PAGE / 4; i += 32) {
      ((uint32_t*)s_pat)[i] = SENT_VAL32; ((uint32_t*)(s_pat + PAGE))[i] = 0xFFFFFFFFu; ((uint32_t*)(s_pat + 2 * PAGE))[i] = SENT_RS;
    }
    fence_async_smem();
  }
  __syncthreads();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int r0 = (int)((long long)blockIdx.x * p.R / p.G), nrows = (int)((long long)(blockIdx.x + 1) * p.R / p.G) - r0;
  const int n = warp >= NW2 || LDBG_NOWORK ? 0 : nrows > warp ? (nrows - warp + NW2 - 1) / NW2 : 0;
  auto rowk = [&](int k) { return r0 + warp + NW2 * k; };
  constexpr bool TR = EPI > 0 && LN_TMAR && MODE != FP4AR_XPORT;
  auto ldr = [&](int k) { mb_expect(&rb_full[warp], INB); tma_ld(rb, p.nm.res_in + (size_t)rowk(k) * H, INB, &rb_full[warp]); };
  auto qslot = [&](int k) { return xb + (k % 3) * ENCB2; };  // QIN: 3-record ring in the (unused) x area
  if (LN_EARLY && n > 0 && lane == 0) {  // first input TMAs right after griddepcontrol.wait (lane L f1), before the call-state read
    if (TR && !AL) ldr(0);
    if (MODE != FP4AR_XPORT) {
      if (QIN) { mb_expect(&xb_full[warp], lne::QROW); tma_ld(qslot(0), p.xq + (size_t)rowk(0) * lne::QROW, lne::QROW, &xb_full[warp]); }
      else { mb_expect(&xb_full[warp], INB); tma_ld(xb, p.x + (size_t)rowk(0) * H, INB, &xb_full[warp]); }
    }
  }
  if (threadIdx.x == 0) { s_call = p.ctrl[0]; s_prevR = p.ctrl[2]; s_prevA = p.ctrl[3]; }
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const unsigned call = s_call; const int buf = call % 3, pbuf = (call + 2) % 3;
  char* const self = p.peer[p.me];
  const SA os{p.osOff + (size_t)buf * p.osBuf, NR, p.cap};
  if (warp == NW2) {
    if (s_prevA != 0 && !LDBG_NORESET) reset_prev<NR>(p, self, s_pat, lane, (int)s_prevA - 1, (int)s_prevR, pbuf);
  } else {
    const bool tmast = !(p.flags & F_TMAST);  // L2: TMA bulk-store egress by default; bit2 selects st.v4
    if (!LN_EARLY && TR && !AL && n > 0 && lane == 0) ldr(0);
    if (!LN_EARLY && n > 0 && lane == 0 && MODE != FP4AR_XPORT) {
      if (QIN) { mb_expect(&xb_full[warp], lne::QROW); tma_ld(qslot(0), p.xq + (size_t)rowk(0) * lne::QROW, lne::QROW, &xb_full[warp]); }
      else { mb_expect(&xb_full[warp], INB); tma_ld(xb, p.x + (size_t)rowk(0) * H, INB, &xb_full[warp]); }
    }
    for (int k = 0; k < n + LAG2; k++) {
      if (k < n) {
        char* ob = QIN ? qslot(k) : own + (k % (LAG2 + 1)) * ENCB2;
        if (EPI > 0 && LN_PF && !(LN_ABL & 1) && lane == 0)
          asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p.nm.res_in + (size_t)rowk(k) * H), "r"(INB) : "memory");
        if (!QIN && tmast && k > LAG2) { if (lane == 0) bulk_wait_read<LAG2>(); __syncwarp(); }  // ring slot's bulk stores done reading
        float rs = 1.f;
        if (QIN && MODE != FP4AR_XPORT) {
          mb_wait(&xb_full[warp], k & 1);
          rs = p.gs[p.me];
          if (k + 1 < n && lane == 0) {  // slot (k+1)%3 held row k-2: decoded, and its bulk group is <= k-2 (wait_read<1>)
            bulk_wait_read<1>(); mb_expect(&xb_full[warp], lne::QROW); tma_ld(qslot(k + 1), p.xq + (size_t)rowk(k + 1) * lne::QROW, lne::QROW, &xb_full[warp]);
          }
        } else if (MODE != FP4AR_XPORT) {
          mb_wait(&xb_full[warp], k & 1);
          rs = fp4f::enc_row_smem<true>((const __nv_bfloat16*)xb, lane, fp4f::EncRow{(uint4*)ob, (uint16_t*)(ob + VB), nullptr});
          fence_async_smem(); __syncwarp();
          if (!AL && k + 1 < n && lane == 0) { mb_expect(&xb_full[warp], INB); tma_ld(xb, p.x + (size_t)rowk(k + 1) * H, INB, &xb_full[warp]); }
        } else {
          if (QIN && tmast && k > 1) { if (lane == 0) bulk_wait_read<1>(); __syncwarp(); }  // slot k%3 last pushed as row k-3
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
      if (AL && kd >= 0 && kd < n && lane == 0) ldr(kd);  // x(k) is encoded: the buffer takes residual(kd)
      if (kd >= 0 && kd < n) {
        const int row = rowk(kd);
        const char* ob = QIN ? qslot(kd) : own + (kd % (LAG2 + 1)) * ENCB2;
        const int s0 = MODE == FP4AR_COMPUTE ? p.me : 0;
        const Src G{self + os.v(s0, row), self + os.sc(s0, row), self + os.m(s0, row),
                    MODE == FP4AR_COMPUTE ? 0 : (size_t)p.cap * VB, MODE == FP4AR_COMPUTE ? 0 : (size_t)p.cap * SB, MODE == FP4AR_COMPUTE ? 0 : (size_t)p.cap * MB};
        const Src O{ob, ob + VB, ob + VB + SB, 0, 0, 0};
        if (EPI == 0) decode_row2<NR, MODE>(G, O, p.me, lane, p.out + (size_t)row * H, stg);
        else if (LN_TMAR) {
          decode_row_norm_t<NR, MODE, EPI>(G, O, p.me, lane, p, row, stg, (const uint4*)s_pat, rb, &rb_full[warp], kd & 1);
          __syncwarp();
          if (TR && !AL && kd + 1 < n && lane == 0) { bulk_wait_read<0>(); ldr(kd + 1); }  // rb free: pass 2 done, res_out store read
        }
        else decode_row_norm<NR, MODE, EPI>(G, O, p.me, lane, p, row, stg, (const uint4*)s_pat);
      }
      if (AL && k + 1 < n) {  // buffer free (pass 2 read it, its res_out / normed stores have read it): next x row
        fence_async_smem(); __syncwarp();
        if (lane == 0) { bulk_wait_read<0>(); mb_expect(&xb_full[warp], INB); tma_ld(xb, p.x + (size_t)rowk(k + 1) * H, INB, &xb_full[warp]); }
      }
    }
    if (lane == 0) bulk_wait_all();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const unsigned old = atomicAdd(&p.ctrl[1], 1u);
    if (old == (unsigned)p.G - 1) { p.ctrl[1] = 0; p.ctrl[2] = p.R; p.ctrl[3] = FP4AR_ONESHOT + 1; __threadfence(); p.ctrl[0] = call + 1; __threadfence(); }
  }
}


// ---------------- host ----------------
struct Priv { unsigned* ctrl; int max_rows, cap, capo; size_t osOff, osBuf, tsOff, tsBuf, tsAg; };

size_t layout(int max_rows, int nranks, Priv* q) {
  const int cap = max_rows, capo = (max_rows + nranks - 1) / nranks;
  auto al = [](size_t x) { return (x + 4095) / 4096 * 4096; };
  const size_t osBuf = al((size_t)nranks * cap * ROWB);
  if (q) { q->cap = cap; q->capo = capo; q->osOff = 0; q->osBuf = osBuf; q->tsOff = 3 * osBuf; q->tsBuf = 0; q->tsAg = 0; }
  return (3 * osBuf + (1 << 21) - 1) / (1 << 21) * (1 << 21);
}

__global__ void fill_sentinel(char* base, int nslot, int cap) {
  const size_t nv = (size_t)nslot * cap * VB / 4, ns = (size_t)nslot * cap * SB / 4, nm = (size_t)nslot * cap * MB / 4;
  uint32_t* w = (uint32_t*)base;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nv + ns + nm; i += (size_t)gridDim.x * blockDim.x)
    w[i] = i < nv ? SENT_VAL32 : i < nv + ns ? 0xFFFFFFFFu : SENT_RS;
}

template <int NR, int EPI, bool Q> const void* pk(int mode) {
  return mode == 0 ? (const void*)lnorm_kernel<NR, 0, EPI, Q> : mode == 1 ? (const void*)lnorm_kernel<NR, 1, EPI, Q> : (const void*)lnorm_kernel<NR, 2, EPI, Q>;
}
template <int NR, bool Q> const void* pke(int mode, int epi) { return epi == 0 ? pk<NR, 0, Q>(mode) : epi == 1 ? pk<NR, 1, Q>(mode) : epi == 2 ? pk<NR, 2, Q>(mode) : pk<NR, 3, Q>(mode); }
// TP4 builds without the QIN path (TP2 is the target).
const void* pickk(int nr, int mode, int epi, bool q = false) { return nr == 2 ? (q ? pke<2, true>(mode, epi) : pke<2, false>(mode, epi)) : (q ? nullptr : pke<4, false>(mode, epi)); }
}  // namespace

size_t lnorm_window_bytes(int max_rows, int hidden, int nranks) { (void)hidden; return layout(max_rows, nranks, nullptr); }

int lnorm_init(Fp4ArCtx* c, int max_rows, int hidden) {
  if (hidden != H || (c->nranks != 2 && c->nranks != 4)) { fprintf(stderr, "lnorm: need hidden=6144, nranks 2|4\n"); return -1; }
  Priv* q = new Priv{}; q->max_rows = max_rows;
  layout(max_rows, c->nranks, q);
  if (cudaMalloc(&q->ctrl, 16) != cudaSuccess) return -1;
  cudaMemset(q->ctrl, 0, 16);
  char* self = c->peer[c->rank];
  for (int b = 0; b < 3; b++) fill_sentinel<<<1024, 256>>>(self + q->osOff + b * q->osBuf, c->nranks, q->cap);
  for (int m = 0; m < 3; m++)
    for (int e = 0; e < 4; e++)
      for (int qq = 0; qq < 2; qq++) if (pickk(c->nranks, m, e, qq)) cudaFuncSetAttribute(pickk(c->nranks, m, e, qq), cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemk(e, qq));
  if (cudaDeviceSynchronize() != cudaSuccess) return -1;
  float* t; cudaMalloc(&t, 4);
  if (ncclAllReduce(t, t, 1, ncclFloat, ncclSum, c->comm, 0) != ncclSuccess) return -1;
  cudaDeviceSynchronize(); cudaFree(t);
  c->priv = q;
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// a->out = BF16 out (epi 0: AR sum; epi 1: normed); n->outq for epi 2. One-shot only.
int lnorm_launch(Fp4ArCtx* c, const Fp4ArArgs* a, const lne::NormArgs* n, cudaStream_t s) {
  Priv* q = (Priv*)c->priv;
  if (a->rows > q->max_rows || a->hidden != H || n->epi < 0 || n->epi > 3) return -1;
  LP p{};
  if (a->xq && (c->nranks != 2 || a->xq_in_window)) return -1;
  p.xq = a->xq; for (int r = 0; r < c->nranks; r++) p.gs[r] = a->gs[r];
  p.x = a->x; p.out = a->out; p.R = a->rows; p.me = c->rank; p.flags = a->tune[2]; p.nm = *n;
  if (n->epi > 0 && LN_WSM) p.flags |= F_RSTV4;  // sentinel pages replaced by the weight row
  p.G = a->tune[0] > 0 ? std::min(a->tune[0], c->sm_count) : c->sm_count;
  p.cap = q->cap; p.capo = q->capo;
  for (int r = 0; r < c->nranks; r++) p.peer[r] = c->peer[r];
  p.osOff = q->osOff; p.osBuf = q->osBuf; p.tsOff = q->tsOff; p.tsBuf = q->tsBuf; p.tsAg = q->tsAg; p.ctrl = q->ctrl;
  cudaLaunchConfig_t cfg{};
  const bool qq = a->xq != nullptr;
  cfg.gridDim = dim3(p.G); cfg.blockDim = dim3(32 * (nwk(n->epi, qq) + 1)); cfg.stream = s; cfg.dynamicSmemBytes = smemk(n->epi, qq);
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = (p.flags & F_NOPDL) ? 0 : 1;
  void* args[] = {&p};
  return cudaLaunchKernelExC(&cfg, pickk(c->nranks, a->mode, n->epi, a->xq != nullptr), args) == cudaSuccess ? 0 : -1;
}

void lnorm_fini(Fp4ArCtx* c) { Priv* q = (Priv*)c->priv; if (q) { cudaFree(q->ctrl); delete q; c->priv = nullptr; } }
