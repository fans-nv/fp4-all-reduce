// fp4-allreduce: push TP4 two-shot Lamport AllReduce, variants L4 (BF16 in) / L4q (NVFP4 in); frozen build tp4push-f1.
// Variants L4 / L4q: TP4 Lamport push (derived from kernels/lamport/snap/it10 L2). Design + log: NOTES.md (same dir).
// One persistent CTA per SM: NW worker warps (warp per row) + 1 reset warp. BF16 input (a->xq == nullptr) or NVFP4 input
// (a->xq: 3456 B rows, static per-rank gs; no encode in the push phase). Egress = TMA bulk stores; no flags, no sys fences.
// One-shot: TMA x -> smem, encode, push the row to 3 peers, decode-sum 4 sources (own from smem) one row later.
// Two-shot: owner(row) = row % NR, slot index row / NR. A: encode + push each row to its owner's RS slot (self via local
// store). B: owner polls 4 RS sources, decode-sum FP32 (pass 1 = poll + amax), re-encode (pass 2, L2-hot re-read), out =
// decode of re-encoding, push the encoded sum to 3 peers' AG slot. C: decode the AG row of every non-owned row.
// tune[0] = CTAs (0 -> SM count), tune[2] flags: bit0 no PDL.
#include "fp4ar_api.h"
#include "fp4_codec.cuh"
#include "codec_fast.cuh"
#include <cstdio>
#include <algorithm>

#ifndef TS_OLD
#define TS_OLD 1  // 1 = phase-ordered two-shot (default; the quad pipeline, TS_OLD=0, measured slower on GB200)
#endif
#ifndef L4_WS
#define L4_WS 0  // two-shot warp specialization (measured slower on GB200)
#endif
#ifndef BCQ_MIN
#define BCQ_MIN 64
#endif
#ifndef TEAM_MAX
#define TEAM_MAX 40
#endif
#ifndef WS_MIN
#define WS_MIN 0
#endif
#ifndef OU
#define OU 1  // owner pass-2 pairs loaded per batch
#endif
#ifndef NW4
#define NW4 10
#endif

namespace {
using namespace fp4c;

constexpr int MB = 16;                 // row scale x4
constexpr int ROWB = VB + SB + MB;     // 3472 wire bytes per row
constexpr int QROW = VB + SB;          // NVFP4 input row (3456 B)
constexpr int INB = H * 2;
constexpr int PAGE = 4096;
constexpr int NW = NW4;
constexpr int ENCB = ROWB;
constexpr int WSM = INB + 2 * ENCB + 2048;  // per worker warp: x row (Q: 3-slot ring), 2-slot own ring, store staging
constexpr size_t SMEM = 3 * PAGE + 64 + (size_t)NW * WSM;
enum { F_NOPDL = 1, F_NOTEAM = 2, F_BCQ = 4 };

struct LP {
  const __nv_bfloat16* x; __nv_bfloat16* out; const uint8_t* xq;
  float gs[4];
  int R, me, G, flags, cap, capo, nwa, nwb, nst;
  char* peer[4];
  size_t osOff, osBuf, tsOff, tsBuf, tsAg;
  unsigned* ctrl;  // [0] call, [1] done, [2] prev rows, [3] prev algo + 1 (0 = none)
};

// Slot array: V[nslot][cap][VB], S[nslot][cap][SB], M[nslot][cap][MB].
struct SA {
  size_t base; int nslot, cap;
  __device__ size_t v(int s, int r) const { return base + ((size_t)s * cap + r) * VB; }
  __device__ size_t sc(int s, int r) const { return base + (size_t)nslot * cap * VB + ((size_t)s * cap + r) * SB; }
  __device__ size_t m(int s, int r) const { return base + (size_t)nslot * cap * (VB + SB) + ((size_t)s * cap + r) * MB; }
};
struct Src { const char *v0, *s0, *m0; size_t sv, ss, sm;
  __device__ const char* v(int s) const { return v0 + s * sv; }
  __device__ const char* sc(int s) const { return s0 + s * ss; }
  __device__ const char* m(int s) const { return m0 + s * sm; } };

// ---------------- PTX helpers ----------------
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint64_t* b, uint32_t n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sa(b)), "r"(n)); }
__device__ __forceinline__ void mb_expect(uint64_t* b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa(b)), "r"(tx) : "memory"); }
__device__ __forceinline__ bool mb_try(uint64_t* b, uint32_t par) {
  uint32_t ok;
  asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(sa(b)), "r"(par) : "memory");
  return ok;
}
__device__ __forceinline__ void mb_wait(uint64_t* b, uint32_t par) { while (!mb_try(b, par)) {} }
__device__ __forceinline__ void tma_ld(void* dst, const void* src, uint32_t n, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sa(dst)), "l"(src), "r"(n), "r"(sa(b)) : "memory");
}
__device__ __forceinline__ void tma_st(void* gdst, const void* src, uint32_t n) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(gdst), "r"(sa(src)), "r"(n) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ void bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ uint32_t ldv32(const void* p) { uint32_t v; asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }
__device__ __forceinline__ uint32_t ldv16(const void* p) { uint16_t v; asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(v) : "l"(p)); return v; }
__device__ __forceinline__ uint4 ldv128(const void* p) {
  uint4 r; asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p)); return r;
}
__device__ __forceinline__ uint32_t negz(uint32_t w) { return w & ~((w & 0x77777777u) + 0x77777777u) & 0x88888888u; }
__device__ __forceinline__ bool bad_q(uint4 q) { return (negz(q.x) | negz(q.y) | negz(q.z) | negz(q.w)) != 0; }
__device__ __forceinline__ bool bad_s(uint32_t c) { return (c & 0xFFu) == 0xFFu || (c >> 8) == 0xFFu; }

// Row scale of each source: Q -> static gs; else the M word (own from smem O when s == me), polled until valid.
template <int NS, int MODE, bool Q>
__device__ __forceinline__ void src_scales(const Src& G, const Src& O, int me, const float* gs, float* rs) {
#pragma unroll
  for (int s = 0; s < NS; s++) {
    if (Q) { rs[s] = gs[s]; continue; }
    uint32_t b = s == me ? *(const uint32_t*)O.m0 : ldv32(G.m(s));
    if (MODE != FP4AR_COMPUTE && s != me) while (b == SENT_RS) b = ldv32(G.m(s));
    rs[s] = __uint_as_float(b);
  }
}
// Load U pairs (j0..j0+U-1) of NS sources; poll peers until no sentinel remains (all bad words re-polled together).
template <int NS, int U, int MODE>
__device__ __forceinline__ void load_poll(const Src& G, const Src& O, int me, int lane, int j0, uint4 (&v)[U][NS], uint32_t (&c)[U][NS]) {
#pragma unroll
  for (int u = 0; u < U; u++) {
    const int pp = lane + 32 * (j0 + u);
#pragma unroll
    for (int s = 0; s < NS; s++) {
      if (s == me) { v[u][s] = ((const uint4*)O.v0)[pp]; c[u][s] = ((const uint16_t*)O.s0)[pp]; }
      else { v[u][s] = ldv128(G.v(s) + 16 * pp); c[u][s] = ldv16(G.sc(s) + 2 * pp); }
    }
  }
  if (MODE == FP4AR_COMPUTE) return;
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

// Poll + decode-sum NS sources (rank order) -> BF16 row with smem-staged coalesced stores.
template <int NS, int MODE, bool Q>
__device__ __forceinline__ void decode_row(const Src G, const Src O, int me, int lane, __nv_bfloat16* out, uint4* stg, const float* gs) {
  constexpr int U = NS >= 4 ? 2 : 3;
  float rs[NS]; src_scales<NS, MODE, Q>(G, O, me, gs, rs);
#pragma unroll 1
  for (int j0 = 0; j0 < 6; j0 += U) {
    uint4 v[U][NS]; uint32_t c[U][NS];
    load_poll<NS, U, MODE>(G, O, me, lane, j0, v, c);
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

// Two-shot owner: pass 1 polls + decode-sums for the row amax, pass 2 re-reads (ld.cg, L2-hot), re-encodes into ob (wire
// layout, CANON) and writes out = decode of the re-encoding. All NS sources come from G (own RS slot included).
template <int NS, int MODE, bool Q>
__device__ __forceinline__ void owner_row(const Src G, int lane, const float* gs, char* ob, __nv_bfloat16* out, uint4* stg) {
  constexpr int U = 2;
  const Src O{nullptr, nullptr, nullptr, 0, 0, 0};
  float rs[NS]; src_scales<NS, MODE, Q>(G, O, -1, gs, rs);
  float am = 0.f;
#pragma unroll 1
  for (int j0 = 0; j0 < 6; j0 += U) {
    uint4 v[U][NS]; uint32_t c[U][NS];
    load_poll<NS, U, MODE>(G, O, -1, lane, j0, v, c);
    if (MODE == FP4AR_XPORT) continue;
#pragma unroll
    for (int u = 0; u < U; u++) {
      float acc[32]; zero32(acc);
#pragma unroll
      for (int s = 0; s < NS; s++) fp4f::dec_acc_fast(acc, v[u][s], c[u][s], rs[s]);
      am = amax32(acc, am);
    }
  }
  float rs2 = 1.f;
  if (MODE == FP4AR_XPORT) {
#pragma unroll
    for (int j = 0; j < 6; j++) ((uint4*)ob)[lane + 32 * j] = make_uint4(0x11111111u, 0x11111111u, 0x11111111u, 0x11111111u);
    if (lane < 24) ((uint4*)(ob + VB))[lane] = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
  } else {
    float inv; row_scales(warp_amax(am), inv, rs2);
#pragma unroll 1
    for (int j0 = 0; j0 < 6; j0 += OU) {
      uint4 v[OU][NS]; uint32_t c[OU][NS];
#pragma unroll
      for (int u = 0; u < OU; u++)
#pragma unroll
        for (int s = 0; s < NS; s++) { const int pp = lane + 32 * (j0 + u); v[u][s] = __ldcg((const uint4*)G.v(s) + pp); c[u][s] = __ldcg((const uint16_t*)G.sc(s) + pp); }
#pragma unroll
      for (int u = 0; u < OU; u++) {
        const int pp = lane + 32 * (j0 + u); float acc[32]; zero32(acc);
#pragma unroll
        for (int s = 0; s < NS; s++) fp4f::dec_acc_fast(acc, v[u][s], c[u][s], rs[s]);
        uint4 q; const uint32_t sf = fp4f::enc_pair_f_fast<true>(acc, inv, q);
        ((uint4*)ob)[pp] = q; ((uint16_t*)(ob + VB))[pp] = (uint16_t)sf;
        zero32(acc); fp4f::dec_acc_fast(acc, q, sf, rs2);
        uint4 o4[4]; fp4f::pack_bf16x32(acc, o4);
        fp4f::store_block_staged(stg, lane, o4, (uint4*)out + 128 * (j0 + u));
      }
    }
  }
  if (lane < 4) ((float*)(ob + VB + SB))[lane] = rs2;
}

// Two-shot owner, 2-warp team, single pass: warp h decodes pairs 3h..3h+2 of all NS sources (poll) and keeps the FP32 sums in
// registers; row amax via s_am (double-buffered by item parity) + 64-thread named barrier; re-encode into ob, out = decode.
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
template <int NS, int MODE, bool Q>
__device__ __forceinline__ void owner_half(const Src G, int lane, int h, const float* gs, char* ob, __nv_bfloat16* out, uint4* stg,
                                           float* s_am, int bar) {
  const Src O{nullptr, nullptr, nullptr, 0, 0, 0};
  float rs[NS]; src_scales<NS, MODE, Q>(G, O, -1, gs, rs);
  float acc[3][32];
#pragma unroll
  for (int jj = 0; jj < 3; jj++) {
    uint4 v[1][NS]; uint32_t c[1][NS];
    load_poll<NS, 1, MODE>(G, O, -1, lane, 3 * h + jj, v, c);
    zero32(acc[jj]);
    if (MODE != FP4AR_XPORT) {
#pragma unroll
      for (int s = 0; s < NS; s++) fp4f::dec_acc_fast(acc[jj], v[0][s], c[0][s], rs[s]);
    }
  }
  float am = 0.f;
#pragma unroll
  for (int jj = 0; jj < 3; jj++) am = amax32(acc[jj], am);
  am = warp_amax(am);
  if (lane == 0) s_am[h] = am;
  named_bar(bar, 64);
  am = fmaxf(s_am[0], s_am[1]);
  float rs2 = 1.f;
  if (MODE == FP4AR_XPORT) {
#pragma unroll
    for (int jj = 0; jj < 3; jj++) ((uint4*)ob)[lane + 32 * (3 * h + jj)] = make_uint4(0x11111111u, 0x11111111u, 0x11111111u, 0x11111111u);
    if (lane < 12) ((uint4*)(ob + VB))[12 * h + lane] = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
  } else {
    float inv; row_scales(am, inv, rs2);
#pragma unroll
    for (int jj = 0; jj < 3; jj++) {
      const int pp = lane + 32 * (3 * h + jj);
      uint4 q; const uint32_t sf = fp4f::enc_pair_f_fast<true>(acc[jj], inv, q);
      ((uint4*)ob)[pp] = q; ((uint16_t*)(ob + VB))[pp] = (uint16_t)sf;
      float d[32]; zero32(d); fp4f::dec_acc_fast(d, q, sf, rs2);
      uint4 o4[4]; fp4f::pack_bf16x32(d, o4);
      fp4f::store_block_staged(stg, lane, o4, (uint4*)out + 128 * (3 * h + jj));
    }
  }
  if (h == 0 && lane < 4) ((float*)(ob + VB + SB))[lane] = rs2;
}

// Re-arm this CTA's share of the receive buffers written in the previous call (TMA bulk stores of sentinel pages).
template <int NR>
__device__ __forceinline__ void reset_prev(const LP& p, char* self, const char* s_pat, int lane, int pA, int pR, int pbuf) {
  int j = 0;
  auto range = [&](size_t base, size_t bytes, int t) {
    const size_t u = bytes / 16, lo = u * blockIdx.x / p.G * 16, hi = u * (blockIdx.x + 1) / p.G * 16;
    for (size_t o = lo; o < hi; o += PAGE, j++)
      if ((j & 31) == lane) tma_st(self + base + o, s_pat + t * PAGE, (uint32_t)min((size_t)PAGE, hi - o));
  };
  auto slots = [&](const SA& q, int rows, int skip) {
    for (int s = 0; s < NR; s++) if (s != skip) { range(q.v(s, 0), (size_t)rows * VB, 0); range(q.sc(s, 0), (size_t)rows * SB, 1); range(q.m(s, 0), (size_t)rows * MB, 2); }
  };
  if (pA == FP4AR_ONESHOT) slots(SA{p.osOff + (size_t)pbuf * p.osBuf, NR, p.cap}, pR, p.me);
  else {
    const int pRo = (pR + NR - 1) / NR;
    slots(SA{p.tsOff + (size_t)pbuf * p.tsBuf, NR, p.capo}, pRo, -1);           // RS (own slot is written locally)
    slots(SA{p.tsOff + (size_t)pbuf * p.tsBuf + p.tsAg, NR, p.capo}, pRo, p.me); // AG (own slot never written)
  }
  bulk_commit(); bulk_wait_all();
}

template <int NR, int MODE, int ALGO, bool Q, int BV>
__global__ void __launch_bounds__(32 * (NW + 1), 1) l4_kernel(const __grid_constant__ LP p) {
  extern __shared__ __align__(1024) char sm[];
  __shared__ __align__(8) uint64_t xb_full[2 * NW];
  __shared__ unsigned s_call, s_prevR, s_prevA;
  __shared__ float s_am[NW / 2][2][2];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  char* const s_pat = sm;
  char* const wsm = sm + 3 * PAGE + 64 + (warp < NW ? warp : 0) * WSM;
  char* const xb = wsm; char* const own = wsm + INB; uint4* const stg = (uint4*)(own + 2 * ENCB);
  if (threadIdx.x == 0) { for (int i = 0; i < 2 * NW; i++) mb_init(&xb_full[i], 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  if (warp == NW) {
    for (int i = lane; i < PAGE / 4; i += 32) {
      ((uint32_t*)s_pat)[i] = SENT_VAL32; ((uint32_t*)(s_pat + PAGE))[i] = 0xFFFFFFFFu; ((uint32_t*)(s_pat + 2 * PAGE))[i] = SENT_RS;
    }
    fence_async_smem();
  }
  __syncthreads();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (threadIdx.x == 0) { s_call = p.ctrl[0]; s_prevR = p.ctrl[2]; s_prevA = p.ctrl[3]; }
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const unsigned call = s_call; const int buf = call % 3, pbuf = (call + 2) % 3;
  char* const self = p.peer[p.me];
  const int QA = ALGO == FP4AR_TWOSHOT ? NR : 1;  // two-shot chunks are quad aligned (R % 4 == 0)
  const int r0 = QA * (int)((long long)blockIdx.x * (p.R / QA) / p.G), nrows = QA * (int)((long long)(blockIdx.x + 1) * (p.R / QA) / p.G) - r0;
  if (warp == NW) {
    if (s_prevA != 0) reset_prev<NR>(p, self, s_pat, lane, (int)s_prevA - 1, (int)s_prevR, pbuf);
  } else if (ALGO == FP4AR_TWOSHOT && !TS_OLD) {
    // ---- two-shot, quad pipeline: warp w owns quads q = qa + w + NW k (rows 4q..4q+3, one per owner). Step k does
    // A(quad k): load/encode the 4 rows, push each to its owner's RS slot; B(quad k-1): owner row; C(quad k-2): 3 AG rows.
    // Every wait targets an item that every rank issues at an earlier step, so no cycle.
    const int Qn = p.R / NR, qa = (int)((long long)blockIdx.x * Qn / p.G), nq = (int)((long long)(blockIdx.x + 1) * Qn / p.G) - qa;
    const int n = nq > warp ? (nq - warp + NW - 1) / NW : 0;
    auto quad = [&](int k) { return qa + warp + NW * k; };
    auto rowj = [&](int j) { return NR * quad(j / NR) + (j % NR); };  // j-th A row of this warp
    auto bar = [&](int j) { return &xb_full[2 * warp + (j & 1)]; };
    auto qslot = [&](int j) -> char* { return xb + (j % 3) * ENCB; };
    auto load = [&](int j) {
      if (Q) { mb_expect(bar(j), QROW); tma_ld(qslot(j), p.xq + (size_t)rowj(j) * QROW, QROW, bar(j)); }
      else { mb_expect(bar(j), INB); tma_ld(xb, p.x + (size_t)rowj(j) * H, INB, bar(j)); }
    };
    const SA rsa{p.tsOff + (size_t)buf * p.tsBuf, NR, p.capo};
    const SA aga{p.tsOff + (size_t)buf * p.tsBuf + p.tsAg, NR, p.capo};
    const size_t z = MODE == FP4AR_COMPUTE ? 0 : 1;
    const int nA = NR * n;
    if (nA > 0 && lane == 0 && (Q || MODE != FP4AR_XPORT)) load(0);
    int e = 0;  // items written into the 2-slot own ring (BF16 A rows and B rows; Q: B rows only)
    auto oslot = [&]() -> char* {
      char* ob = own + (e & 1) * ENCB;
      if (e >= 2) { if (lane == 0) bulk_wait_read<1>(); __syncwarp(); }
      e++;
      return ob;
    };
    for (int k = 0; k < n + 2; k++) {
      if (k < n) {
#pragma unroll 1
        for (int i = 0; i < NR; i++) {
          const int j = NR * k + i, row = rowj(j);
          char* ob;
          if (Q) {
            ob = qslot(j);
            if (j >= 2) { if (lane == 0) bulk_wait_read<1>(); __syncwarp(); }
            if (lane == 0 && j + 1 < nA) { fence_async_smem(); load(j + 1); }
            mb_wait(bar(j), (j >> 1) & 1);
          } else {
            ob = oslot();
            float rs = 1.f;
            if (MODE != FP4AR_XPORT) {
              mb_wait(bar(j), (j >> 1) & 1);
              rs = fp4f::enc_row_smem<true>((const __nv_bfloat16*)xb, lane, fp4f::EncRow{(uint4*)ob, (uint16_t*)(ob + VB), nullptr});
              __syncwarp();
              if (j + 1 < nA && lane == 0) { fence_async_smem(); load(j + 1); }
            } else {
#pragma unroll
              for (int t = 0; t < 6; t++) ((uint4*)ob)[lane + 32 * t] = make_uint4(0x11111111u, 0x11111111u, 0x11111111u, 0x11111111u);
              if (lane < 24) ((uint4*)(ob + VB))[lane] = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
            }
            if (lane < 4) ((float*)(ob + VB + SB))[lane] = rs;
          }
          fence_async_smem();
          __syncwarp();
          if (lane == 0) {
            if (MODE != FP4AR_COMPUTE) {
              char* base = p.peer[row % NR]; const int ix = row / NR;
              tma_st(base + rsa.v(p.me, ix), ob, VB); tma_st(base + rsa.sc(p.me, ix), ob + VB, SB);
              if (!Q) tma_st(base + rsa.m(p.me, ix), ob + VB + SB, MB);
            }
            bulk_commit();
          }
        }
      }
      if (k >= 1 && k - 1 < n) {  // B: my row of quad k-1
        const int ix = quad(k - 1), row = NR * ix + p.me;
        char* ob = oslot();
        const int s0 = MODE == FP4AR_COMPUTE ? p.me : 0;
        const Src G{self + rsa.v(s0, ix), self + rsa.sc(s0, ix), self + rsa.m(s0, ix), z * p.capo * VB, z * p.capo * SB, z * p.capo * MB};
        owner_row<NR, MODE, Q>(G, lane, p.gs, ob, p.out + (size_t)row * H, stg);
        fence_async_smem();
        __syncwarp();
        if (lane == 0) {
          if (MODE != FP4AR_COMPUTE)
            for (int d = 1; d < NR; d++) {
              char* base = p.peer[(p.me + d) % NR];
              tma_st(base + aga.v(p.me, ix), ob, VB); tma_st(base + aga.sc(p.me, ix), ob + VB, SB); tma_st(base + aga.m(p.me, ix), ob + VB + SB, MB);
            }
          bulk_commit();
        }
      }
      if (k >= 2 && k - 2 < n) {  // C: the 3 other rows of quad k-2
        const int ix = quad(k - 2);
#pragma unroll 1
        for (int d = 1; d < NR; d++) {
          const int o = (p.me + d) % NR, row = NR * ix + o;
          const int s0 = MODE == FP4AR_COMPUTE ? p.me : o;
          const Src G{self + aga.v(s0, ix), self + aga.sc(s0, ix), self + aga.m(s0, ix), 0, 0, 0};
          decode_row<1, MODE, false>(G, G, -1, lane, p.out + (size_t)row * H, stg, nullptr);
        }
      }
    }
    if (lane == 0) bulk_wait_all();
  } else if (ALGO == FP4AR_TWOSHOT && BV == 3) {
    // ---- two-shot staged: the chunk's quads are split into p.nst segments; per segment every warp does its A rows, then
    // B/C by quad for that segment. Waits only target items of the same or an earlier segment, which every rank issues first.
    const SA rsa{p.tsOff + (size_t)buf * p.tsBuf, NR, p.capo};
    const SA aga{p.tsOff + (size_t)buf * p.tsBuf + p.tsAg, NR, p.capo};
    const size_t z = MODE == FP4AR_COMPUTE ? 0 : 1;
    const bool loads = Q || MODE != FP4AR_XPORT;
    auto slot = [&](int g) -> char* { return Q ? xb + (g % 3) * ENCB : own + (g & 1) * ENCB; };
    auto bar = [&](int g) { return &xb_full[2 * warp + (g & 1)]; };
    auto load = [&](int g, int row) {
      if (Q) { mb_expect(bar(g), QROW); tma_ld(slot(g), p.xq + (size_t)row * QROW, QROW, bar(g)); }
      else { mb_expect(bar(g), INB); tma_ld(xb, p.x + (size_t)row * H, INB, bar(g)); }
    };
    const int q0 = r0 / NR, nq = nrows / NR;
    int kg = 0;
    for (int st = 0; st < p.nst; st++) {
      const int qs = q0 + nq * st / p.nst, qe = q0 + nq * (st + 1) / p.nst, rs0 = qs * NR, nr = (qe - qs) * NR;
      const int n = nr > warp ? (nr - warp + NW - 1) / NW : 0;
      auto rowk = [&](int k) { return rs0 + warp + NW * k; };
      if (lane == 0) bulk_wait_read<0>();
      __syncwarp();
      if (n > 0 && lane == 0 && loads) { fence_async_smem(); load(kg, rowk(0)); }
      for (int k = 0; k < n; k++) {
        const int g = kg + k, row = rowk(k);
        char* ob = slot(g);
        if (k >= 2) { if (lane == 0) bulk_wait_read<1>(); __syncwarp(); }
        if (Q) {
          if (lane == 0 && k + 1 < n) { fence_async_smem(); load(g + 1, rowk(k + 1)); }
          mb_wait(bar(g), (g >> 1) & 1);
        } else {
          float rs = 1.f;
          if (MODE != FP4AR_XPORT) {
            mb_wait(bar(g), (g >> 1) & 1);
            rs = fp4f::enc_row_smem<true>((const __nv_bfloat16*)xb, lane, fp4f::EncRow{(uint4*)ob, (uint16_t*)(ob + VB), nullptr});
            __syncwarp();
            if (k + 1 < n && lane == 0) { fence_async_smem(); load(g + 1, rowk(k + 1)); }
          } else {
#pragma unroll
            for (int j = 0; j < 6; j++) ((uint4*)ob)[lane + 32 * j] = make_uint4(0x11111111u, 0x11111111u, 0x11111111u, 0x11111111u);
            if (lane < 24) ((uint4*)(ob + VB))[lane] = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
          }
          if (lane < 4) ((float*)(ob + VB + SB))[lane] = rs;
        }
        fence_async_smem();
        __syncwarp();
        if (lane == 0) {
          if (MODE != FP4AR_COMPUTE) {
            char* base = p.peer[row % NR]; const int ix = row / NR;
            tma_st(base + rsa.v(p.me, ix), ob, VB); tma_st(base + rsa.sc(p.me, ix), ob + VB, SB);
            if (!Q) tma_st(base + rsa.m(p.me, ix), ob + VB + SB, MB);
          }
          bulk_commit();
        }
      }
      kg += n;
      if (lane == 0) bulk_wait_read<0>();
      __syncwarp();
      int kb = 0;
      for (int ix = qs + (warp - nr % NW + NW) % NW; ix < qe; ix += NW, kb++) {
        {
          const int row = NR * ix + p.me;
          char* ob = xb + (kb & 1) * ENCB;
          if (kb >= 2) { if (lane == 0) bulk_wait_read<1>(); __syncwarp(); }
          const int s0 = MODE == FP4AR_COMPUTE ? p.me : 0;
          const Src G{self + rsa.v(s0, ix), self + rsa.sc(s0, ix), self + rsa.m(s0, ix), z * p.capo * VB, z * p.capo * SB, z * p.capo * MB};
          owner_row<NR, MODE, Q>(G, lane, p.gs, ob, p.out + (size_t)row * H, stg);
          fence_async_smem();
          __syncwarp();
          if (lane == 0) {
            if (MODE != FP4AR_COMPUTE)
              for (int d = 1; d < NR; d++) {
                char* base = p.peer[(p.me + d) % NR];
                tma_st(base + aga.v(p.me, ix), ob, VB); tma_st(base + aga.sc(p.me, ix), ob + VB, SB); tma_st(base + aga.m(p.me, ix), ob + VB + SB, MB);
              }
            bulk_commit();
          }
        }
#pragma unroll 1
        for (int d = 1; d < NR; d++) {
          const int o = (p.me + d) % NR, row = NR * ix + o;
          const int s0 = MODE == FP4AR_COMPUTE ? p.me : o;
          const Src G{self + aga.v(s0, ix), self + aga.sc(s0, ix), self + aga.m(s0, ix), 0, 0, 0};
          decode_row<1, MODE, false>(G, G, -1, lane, p.out + (size_t)row * H, stg, nullptr);
        }
      }
    }
    if (lane == 0) bulk_wait_all();
  } else {
    // ---- phase A (both algos): rows r0 + warp + NW k: load (+ encode) into a ring slot, push ----
    // two-shot warp specialization (p.nwa > 0): warps [0,nwa) do A, [nwa,nwa+nwb) B, the rest C; else every warp does all phases
    const bool ws = L4_WS && ALGO == FP4AR_TWOSHOT && p.nwa > 0;
    const int NWA = ws ? p.nwa : NW;
    const int n = warp < NWA && nrows > warp ? (nrows - warp + NWA - 1) / NWA : 0;
    auto rowk = [&](int k) { return r0 + warp + NWA * k; };
    auto slot = [&](int k) -> char* { return Q ? xb + (k % 3) * ENCB : own + (k & 1) * ENCB; };
    auto bar = [&](int k) { return &xb_full[2 * warp + (k & 1)]; };
    auto load = [&](int k) {
      if (Q) { mb_expect(bar(k), QROW); tma_ld(slot(k), p.xq + (size_t)rowk(k) * QROW, QROW, bar(k)); }
      else { mb_expect(bar(k), INB); tma_ld(xb, p.x + (size_t)rowk(k) * H, INB, bar(k)); }
    };
    const bool loads = Q || MODE != FP4AR_XPORT;
    const SA os{p.osOff + (size_t)buf * p.osBuf, NR, p.cap};
    const SA rsa{p.tsOff + (size_t)buf * p.tsBuf, NR, p.capo};
    const SA aga{p.tsOff + (size_t)buf * p.tsBuf + p.tsAg, NR, p.capo};
    if (n > 0 && lane == 0 && loads) load(0);
    constexpr int LAG = ALGO == FP4AR_ONESHOT ? 1 : 0;
    for (int k = 0; k < n + LAG; k++) {
      if (k < n) {
        char* ob = slot(k);
        if (k >= 2) { if (lane == 0) bulk_wait_read<1>(); __syncwarp(); }  // slot(k)'s previous user (row k-2 or k-3) done
        if (Q) {
          if (lane == 0 && k + 1 < n) { fence_async_smem(); load(k + 1); }  // slot(k+1) = slot(k-2): decoded, store read done
          mb_wait(bar(k), (k >> 1) & 1);
        } else {
          float rs = 1.f;
          if (MODE != FP4AR_XPORT) {
            mb_wait(bar(k), (k >> 1) & 1);
            rs = fp4f::enc_row_smem<true>((const __nv_bfloat16*)xb, lane, fp4f::EncRow{(uint4*)ob, (uint16_t*)(ob + VB), nullptr});
            __syncwarp();
            if (k + 1 < n && lane == 0) { fence_async_smem(); load(k + 1); }
          } else {
#pragma unroll
            for (int j = 0; j < 6; j++) ((uint4*)ob)[lane + 32 * j] = make_uint4(0x11111111u, 0x11111111u, 0x11111111u, 0x11111111u);
            if (lane < 24) ((uint4*)(ob + VB))[lane] = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
          }
          if (lane < 4) ((float*)(ob + VB + SB))[lane] = rs;
        }
        fence_async_smem();
        __syncwarp();
        if (lane == 0) {
          const int row = rowk(k);
          if (MODE != FP4AR_COMPUTE) {
            if (ALGO == FP4AR_ONESHOT) {
              for (int d = 1; d < NR; d++) {
                char* base = p.peer[(p.me + d) % NR];
                tma_st(base + os.v(p.me, row), ob, VB); tma_st(base + os.sc(p.me, row), ob + VB, SB);
                if (!Q) tma_st(base + os.m(p.me, row), ob + VB + SB, MB);
              }
            } else {
              char* base = p.peer[row % NR]; const int ix = row / NR;
              tma_st(base + rsa.v(p.me, ix), ob, VB); tma_st(base + rsa.sc(p.me, ix), ob + VB, SB);
              if (!Q) tma_st(base + rsa.m(p.me, ix), ob + VB + SB, MB);
            }
          }
          bulk_commit();
        }
      }
      if (ALGO == FP4AR_ONESHOT) {
        const int kd = k - LAG;
        if (kd >= 0 && kd < n) {
          const int row = rowk(kd);
          const char* ob = slot(kd);
          const int s0 = MODE == FP4AR_COMPUTE ? p.me : 0;
          const size_t z = MODE == FP4AR_COMPUTE ? 0 : 1;
          const Src G{self + os.v(s0, row), self + os.sc(s0, row), self + os.m(s0, row), z * p.cap * VB, z * p.cap * SB, z * p.cap * MB};
          const Src O{ob, ob + VB, ob + VB + SB, 0, 0, 0};
          decode_row<NR, MODE, Q>(G, O, p.me, lane, p.out + (size_t)row * H, stg, p.gs);
        }
      }
    }
    if (ALGO == FP4AR_TWOSHOT) {
      // ---- phase B: rows owned by me in this CTA's chunk ----
      const int rb0 = r0 + ((p.me - r0 % NR) + NR) % NR;
      const int nB = rb0 < r0 + nrows ? (r0 + nrows - 1 - rb0) / NR + 1 : 0;
      if (lane == 0) bulk_wait_read<0>();
      __syncwarp();
      const size_t z = MODE == FP4AR_COMPUTE ? 0 : 1;
      int kb = 0;
      // B and C items start at the warps after the last A row, so at small row counts they land on idle warps
      const int nC = nrows - nB;
      int wB = (warp - nrows % NW + NW) % NW, wC = (warp - (nrows + nB) % NW + NW) % NW, sB = NW, sC = NW;
      if (ws) {
        sB = p.nwb; sC = NW - NWA - p.nwb; wB = warp - NWA; wC = warp - NWA - p.nwb;
        if (wB < 0 || wB >= sB) wB = nB;
        if (wC < 0) wC = nC;
      }
      if (BV == 2 && !ws) {
        // B/C by quad: warp w takes chunk quads w, w+NW, ... in increasing order: own row -> B, other 3 -> C. Every C waits on a
        // B that its owner's warp reaches after items of lower rows only, so no cycle.
        const int q0 = (r0 + NR - 1) / NR, q1 = (r0 + nrows) / NR;  // full quads inside the chunk (chunks are quad aligned)
        for (int qd = q0 + wB % NW; qd < q1; qd += NW, kb++) {
          const int ix = qd;
          {
            const int row = NR * ix + p.me;
            char* ob = xb + (kb & 1) * ENCB;
            if (kb >= 2) { if (lane == 0) bulk_wait_read<1>(); __syncwarp(); }
            const int s0 = MODE == FP4AR_COMPUTE ? p.me : 0;
            const Src G{self + rsa.v(s0, ix), self + rsa.sc(s0, ix), self + rsa.m(s0, ix), z * p.capo * VB, z * p.capo * SB, z * p.capo * MB};
            owner_row<NR, MODE, Q>(G, lane, p.gs, ob, p.out + (size_t)row * H, stg);
            fence_async_smem();
            __syncwarp();
            if (lane == 0) {
              if (MODE != FP4AR_COMPUTE)
                for (int d = 1; d < NR; d++) {
                  char* base = p.peer[(p.me + d) % NR];
                  tma_st(base + aga.v(p.me, ix), ob, VB); tma_st(base + aga.sc(p.me, ix), ob + VB, SB); tma_st(base + aga.m(p.me, ix), ob + VB + SB, MB);
                }
              bulk_commit();
            }
          }
#pragma unroll 1
          for (int d = 1; d < NR; d++) {
            const int o = (p.me + d) % NR, row = NR * ix + o;
            const int s0 = MODE == FP4AR_COMPUTE ? p.me : o;
            const Src G{self + aga.v(s0, ix), self + aga.sc(s0, ix), self + aga.m(s0, ix), 0, 0, 0};
            decode_row<1, MODE, false>(G, G, -1, lane, p.out + (size_t)row * H, stg, nullptr);
          }
        }
        wB = nB; wC = nC;
      } else if (BV == 1 && !ws) {
        constexpr int NT = NW / 2;
        const int t = warp >> 1, h = warp & 1;
        char* const lxb = sm + 3 * PAGE + 64 + 2 * t * WSM;
        for (int i = (t - ((nrows % NW) >> 1) + NT) % NT; i < nB; i += NT, kb++) {
          const int row = rb0 + NR * i, ix = row / NR;
          char* ob = lxb + (kb & 1) * ENCB;
          if (h == 0 && kb >= 2 && lane == 0) bulk_wait_read<1>();
          named_bar(1 + t, 64);  // ob free (leader's stores read it) before either half writes
          const int s0 = MODE == FP4AR_COMPUTE ? p.me : 0;
          const Src G{self + rsa.v(s0, ix), self + rsa.sc(s0, ix), self + rsa.m(s0, ix), z * p.capo * VB, z * p.capo * SB, z * p.capo * MB};
          owner_half<NR, MODE, Q>(G, lane, h, p.gs, ob, p.out + (size_t)row * H, stg, s_am[t][kb & 1], 1 + t);
          fence_async_smem();
          named_bar(1 + t, 64);
          if (h == 0 && lane == 0) {
            if (MODE != FP4AR_COMPUTE)
              for (int d = 1; d < NR; d++) {
                char* base = p.peer[(p.me + d) % NR];
                tma_st(base + aga.v(p.me, ix), ob, VB); tma_st(base + aga.sc(p.me, ix), ob + VB, SB); tma_st(base + aga.m(p.me, ix), ob + VB + SB, MB);
              }
            bulk_commit();
          }
        }
        wB = nB;
      }
      for (int i = wB; i < nB; i += sB, kb++) {
        const int row = rb0 + NR * i, ix = row / NR;
        char* ob = xb + (kb & 1) * ENCB;
        if (kb >= 2) { if (lane == 0) bulk_wait_read<1>(); __syncwarp(); }
        const int s0 = MODE == FP4AR_COMPUTE ? p.me : 0;
        const Src G{self + rsa.v(s0, ix), self + rsa.sc(s0, ix), self + rsa.m(s0, ix), z * p.capo * VB, z * p.capo * SB, z * p.capo * MB};
        owner_row<NR, MODE, Q>(G, lane, p.gs, ob, p.out + (size_t)row * H, stg);
        fence_async_smem();
        __syncwarp();
        if (lane == 0) {
          if (MODE != FP4AR_COMPUTE)
            for (int d = 1; d < NR; d++) {
              char* base = p.peer[(p.me + d) % NR];
              tma_st(base + aga.v(p.me, ix), ob, VB); tma_st(base + aga.sc(p.me, ix), ob + VB, SB); tma_st(base + aga.m(p.me, ix), ob + VB + SB, MB);
            }
          bulk_commit();
        }
      }
      // ---- phase C: non-owned rows (c-th non-owned position of the chunk) ----
      const int f = rb0 - r0;
      for (int c = wC; c < nC; c += sC) {
        const int blk = c / (NR - 1), w = c % (NR - 1);
        const int row = r0 + blk * NR + w + (w >= f ? 1 : 0), o = row % NR, ix = row / NR;
        const int s0 = MODE == FP4AR_COMPUTE ? p.me : o;
        const Src G{self + aga.v(s0, ix), self + aga.sc(s0, ix), self + aga.m(s0, ix), 0, 0, 0};
        decode_row<1, MODE, false>(G, G, -1, lane, p.out + (size_t)row * H, stg, nullptr);
      }
    }
    if (lane == 0) bulk_wait_all();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const unsigned old = atomicAdd(&p.ctrl[1], 1u);
    if (old == (unsigned)p.G - 1) { p.ctrl[1] = 0; p.ctrl[2] = p.R; p.ctrl[3] = ALGO + 1; __threadfence(); p.ctrl[0] = call + 1; __threadfence(); }
  }
}

// ---------------- host ----------------
struct Priv { unsigned* ctrl; int max_rows, cap, capo; size_t osOff, osBuf, tsOff, tsBuf, tsAg; };

size_t layout(int max_rows, int nranks, Priv* q) {
  const int cap = max_rows, capo = (max_rows + nranks - 1) / nranks;
  auto al = [](size_t x) { return (x + 4095) / 4096 * 4096; };
  const size_t osBuf = al((size_t)nranks * cap * ROWB), tsAg = al((size_t)nranks * capo * ROWB), tsBuf = 2 * tsAg;
  if (q) { q->cap = cap; q->capo = capo; q->osOff = 0; q->osBuf = osBuf; q->tsOff = 3 * osBuf; q->tsBuf = tsBuf; q->tsAg = tsAg; }
  return (3 * osBuf + 3 * tsBuf + (1 << 21) - 1) / (1 << 21) * (1 << 21);
}
size_t win_bytes(int max_rows, int hidden, int nranks) { (void)hidden; return nranks == 4 ? layout(max_rows, nranks, nullptr) : 0; }

__global__ void fill_sentinel(char* base, int nslot, int cap) {
  const size_t nv = (size_t)nslot * cap * VB / 4, ns = (size_t)nslot * cap * SB / 4, nm = (size_t)nslot * cap * MB / 4;
  uint32_t* w = (uint32_t*)base;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nv + ns + nm; i += (size_t)gridDim.x * blockDim.x)
    w[i] = i < nv ? SENT_VAL32 : i < nv + ns ? 0xFFFFFFFFu : SENT_RS;
}

// BV (two-shot B/C organisation, compile time so each kernel carries one path): 0 per-warp B then C, 1 owner 2-warp teams, 2 B/C by quad
template <int MODE, int ALGO, int BV> const void* kq(bool q) { return q ? (const void*)l4_kernel<4, MODE, ALGO, true, BV> : (const void*)l4_kernel<4, MODE, ALGO, false, BV>; }
template <int ALGO, int BV> const void* km(int mode, bool q) { return mode == 0 ? kq<0, ALGO, BV>(q) : mode == 1 ? kq<1, ALGO, BV>(q) : kq<2, ALGO, BV>(q); }
const void* pickk(int mode, int algo, bool q, int bv = 0) {
  if (algo == FP4AR_ONESHOT) return km<FP4AR_ONESHOT, 0>(mode, q);
  return bv == 3 ? km<FP4AR_TWOSHOT, 3>(mode, q) : bv == 2 ? km<FP4AR_TWOSHOT, 2>(mode, q) : bv == 1 ? km<FP4AR_TWOSHOT, 1>(mode, q) : km<FP4AR_TWOSHOT, 0>(mode, q);
}

int init_impl(Fp4ArCtx* c, int max_rows, int hidden) {
  if (hidden != H || c->nranks != 4) { fprintf(stderr, "l4: need hidden=6144, nranks 4\n"); return -1; }
  Priv* q = new Priv{}; q->max_rows = max_rows;
  layout(max_rows, c->nranks, q);
  if (cudaMalloc(&q->ctrl, 16) != cudaSuccess) return -1;
  cudaMemset(q->ctrl, 0, 16);
  char* self = c->peer[c->rank];
  for (int b = 0; b < 3; b++) {
    fill_sentinel<<<1024, 256>>>(self + q->osOff + b * q->osBuf, c->nranks, q->cap);
    fill_sentinel<<<1024, 256>>>(self + q->tsOff + b * q->tsBuf, c->nranks, q->capo);
    fill_sentinel<<<1024, 256>>>(self + q->tsOff + b * q->tsBuf + q->tsAg, c->nranks, q->capo);
  }
  for (int m = 0; m < 3; m++)
    for (int a = 0; a < 2; a++)
      for (int qq = 0; qq < 8; qq++)
        if (cudaFuncSetAttribute(pickk(m, a, qq & 1, qq >> 1), cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM) != cudaSuccess) {
          fprintf(stderr, "l4: smem attr %zu failed\n", SMEM); return -1;
        }
  if (cudaDeviceSynchronize() != cudaSuccess) return -1;
  float* t; cudaMalloc(&t, 4);  // all ranks armed before anyone pushes
  if (ncclAllReduce(t, t, 1, ncclFloat, ncclSum, c->comm, 0) != ncclSuccess) return -1;
  cudaDeviceSynchronize(); cudaFree(t);
  c->priv = q;
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int launch_impl(Fp4ArCtx* c, const Fp4ArArgs* a, cudaStream_t s) {
  Priv* q = (Priv*)c->priv;
  if (a->rows > q->max_rows || a->rows <= 0 || a->hidden != H || (a->algo == FP4AR_TWOSHOT && a->rows % 4)) return -1;
  LP p{};
  p.x = a->x; p.out = a->out; p.xq = a->xq; p.R = a->rows; p.me = c->rank; p.flags = a->tune[2];
  for (int r = 0; r < 4; r++) p.gs[r] = a->gs[r];
  p.G = a->tune[0] > 0 ? std::min(a->tune[0], c->sm_count) : c->sm_count;
  p.cap = q->cap; p.capo = q->capo;
  if (!(p.flags & 8) && a->rows / p.G > TEAM_MAX) p.flags |= F_NOTEAM;  // owner teams only at small rows/CTA (bit3 forces on)
  if (!(p.flags & 16) && a->rows / p.G > BCQ_MIN) p.flags |= F_BCQ;  // B/C by quad at large rows/CTA (bit4 disables)  
  // two-shot warp specialization: tune[1] = A warps, tune[3] = B warps (C = rest); applies when rows/CTA >= tune thresh (WS_MIN)
  if (a->algo == FP4AR_TWOSHOT && a->tune[1] > 0 && a->rows / p.G >= WS_MIN) {
    p.nwa = a->tune[1]; p.nwb = a->tune[3] > 0 ? a->tune[3] : 2;
    if (p.nwa + p.nwb >= NW) return -1;
  }
  for (int r = 0; r < c->nranks; r++) p.peer[r] = c->peer[r];
  p.osOff = q->osOff; p.osBuf = q->osBuf; p.tsOff = q->tsOff; p.tsBuf = q->tsBuf; p.tsAg = q->tsAg; p.ctrl = q->ctrl;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(p.G); cfg.blockDim = dim3(32 * (NW + 1)); cfg.stream = s; cfg.dynamicSmemBytes = SMEM;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = (p.flags & F_NOPDL) ? 0 : 1;
  void* args[] = {&p};
  int bv = (p.flags & F_BCQ) ? 2 : (p.flags & F_NOTEAM) ? 0 : 1;
  if (a->algo == FP4AR_TWOSHOT && bv == 2 && a->tune[3] > 1) { bv = 3; p.nst = a->tune[3]; }  // tune d = segments (staged B/C by quad)
  return cudaLaunchKernelExC(&cfg, pickk(a->mode, a->algo, a->xq != nullptr, bv), args) == cudaSuccess ? 0 : -1;
}

void fini_impl(Fp4ArCtx* c) { Priv* q = (Priv*)c->priv; if (q) { cudaFree(q->ctrl); delete q; c->priv = nullptr; } }
}  // namespace

#define LSTR2(a) #a
#define LSTR(a) LSTR2(a)
static const Fp4ArVariant kVar = {LSTR(FP4AR_NAME), true, win_bytes, init_impl, launch_impl, fini_impl, nullptr};
const Fp4ArVariant* FP4AR_REGISTER() { return &kVar; }
