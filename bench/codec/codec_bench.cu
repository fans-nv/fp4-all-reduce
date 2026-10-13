// Single-GPU codec microbenchmark + bit-exactness tests for codec_fast.cuh vs the spec-v2 reference (fp4_codec.cuh, ref.cuh).
// Usage: codec_bench [test|bench|all] [rows list, default 1024,2048,4096,8192,16384]
#include "codec_fast.cuh"
#include "ref.cuh"  // harness/ref.cuh, via -I harness
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <random>
#ifndef ENC_ONLY
#define ENC_ONLY 0
#endif

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)
using namespace fp4c;
using bf16 = __nv_bfloat16;
constexpr int MAXS = 8;
struct Slots { char* b[MAXS]; int R; };  // ref::Slot layout per source
__device__ __forceinline__ ref::Slot sl(const Slots& s, int i) { return ref::Slot{s.b[i], s.R}; }
__device__ __forceinline__ fp4f::EncRow erow(ref::Slot s, int row) { return fp4f::EncRow{s.q(row), s.sf(row), s.rs() + row}; }

// ============================== exhaustive / random unit tests ==============================
// T1: all non-negative fp32 bit patterns: div6_rn == __fdiv_rn(x,6); e4m3x2 byte == __nv_cvt_float_to_fp8 of the ref chain.
// bad[0]: scale-byte mismatches + fp32 mismatches with a normal quotient (must be 0); bad[1]: fp32-only mismatches with a
// subnormal quotient (x < 6 * 2^-126; their E4M3 byte is 0 either way).
__global__ void t_div6(unsigned long long* bad) {
  unsigned long long b = 0, bs = 0;
  for (uint64_t u = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; u < 0x80000000ull; u += (uint64_t)gridDim.x * blockDim.x) {
    const float x = __uint_as_float((uint32_t)u);
    const float a = __fdiv_rn(x, 6.f), c = fp4f::div6_rn(x);
    const bool eq = (__float_as_uint(a) == __float_as_uint(c)) || (isnan(a) && isnan(c));
    const uint32_t sr = __nv_cvt_float_to_fp8(a, __NV_SATFINITE, __NV_E4M3);
    const uint32_t s2 = fp4f::e4m3x2(c, x);
    const uint32_t sx = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
    b += ((s2 & 0xff) != sr) + ((s2 >> 8) != sx) + (!eq && !(a < 0x1p-126f));
    bs += !eq && (a < 0x1p-126f);
  }
  if (b) atomicAdd(bad, b);
  if (bs) atomicAdd(bad + 1, bs);
}
// T2: div_e4m3 vs __fdiv_rn for every fp32 inv in [2^-100, 2^100] x every E4M3 byte 0x01..0x7E (exhaustive).
__global__ void t_div_e4m3(unsigned long long* bad) {
  const uint32_t lo = 0x0d800000u /* 2^-100 */, hi = 0x71800000u /* 2^100 */;
  const uint32_t rtab = fp4f::rcp_tab(threadIdx.x & 31);
  unsigned long long b = 0;
  const uint64_t n = hi - lo + 1, st = (uint64_t)gridDim.x * blockDim.x;
  for (uint64_t i0 = (uint64_t)blockIdx.x * blockDim.x; i0 < n; i0 += st) {  // warp-uniform trip count
    const uint64_t i = i0 + threadIdx.x; const bool v = i < n;
    const float inv = __uint_as_float(lo + (uint32_t)(v ? i : 0));
    for (uint32_t s = 1; s < 0x7f; s++) {
      const float sff = e4m3f(s);
      const float a = __fdiv_rn(inv, sff), c = fp4f::div_e4m3(inv, sff, rtab);
      b += v && (__float_as_uint(a) != __float_as_uint(c));
    }
  }
  if (b) atomicAdd(bad, b);
}
__device__ __forceinline__ uint32_t rng(uint64_t& s) { s = s * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(s >> 32) ^ (uint32_t)s; }
__device__ __forceinline__ uint32_t rbf16_finite(uint64_t& s) {  // random finite BF16 bits (all exponents incl. subnormal)
  uint32_t h = rng(s) & 0xffff; if (((h >> 7) & 0xff) == 0xff) h &= 0xbfff; return h;
}
// T3: random pairs: enc_pair_fast<false> vs fp4c::enc_pair, inv from amax in {pair max, larger, tiny (inv=inf), 0}.
__global__ void t_enc_pair(unsigned long long n, uint64_t seed, unsigned long long* bad) {
  unsigned long long b = 0;
  for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
    uint64_t s = seed ^ (i * 0x9e3779b97f4a7c15ull);
    uint4 v[4]; uint32_t* w = (uint32_t*)v;
    const uint32_t mode = rng(s) & 7, ex = rng(s);
    for (int k = 0; k < 16; k++) {
      uint32_t lo = rbf16_finite(s), hi = rbf16_finite(s);
      if (mode == 1) { lo &= 0x807f | ((ex & 0x1f) << 7); hi &= 0x807f | ((ex & 0x1f) << 7); }   // tiny / subnormal
      if (mode == 2) { lo = (lo & 0x80ff) | 0x3f00; hi = (hi & 0x80ff) | 0x3f00; }             // narrow range ~[0.5,2)
      if (mode == 3 && (rng(s) & 3)) { lo &= 0x8000; hi &= 0x8000; }                           // mostly +-0
      w[k] = lo | (hi << 16);
    }
    float am = 0.f; for (int k = 0; k < 4; k++) am = amax16B(v[k], am);
    const uint32_t r = rng(s) & 7;
    if (r == 1) am = __uint_as_float((__float_as_uint(am) + (rng(s) & 0x3ffffff)) & 0xffff0000u);
    if (r == 2) am = __uint_as_float(rng(s) & 0x00ff0000u);
    if (r == 3) am = 0.f;
    if (!(am < 3.3e38f)) am = 3.3e38f;
    float inv, rs; row_scales(am, inv, rs);
    uint4 q1, q2; uint16_t sf1; enc_pair(v, inv, q1, sf1);
    const uint32_t sf2 = fp4f::enc_pair_fast<false>(v, inv, q2);
    const uint4 q3 = canon_neg0(q1); uint4 q4; fp4f::enc_pair_fast<true>(v, inv, q4);
    b += (q1.x != q2.x) | (q1.y != q2.y) | (q1.z != q2.z) | (q1.w != q2.w) | (sf1 != sf2) |
         (q3.x != q4.x) | (q3.y != q4.y) | (q3.z != q4.z) | (q3.w != q4.w) | (sf2 == 0xff) | ((sf2 >> 8) == 0xff) | is_sent_val(q4);
  }
  if (b) atomicAdd(bad, b);
}
// T4: FP32-input pair encode (two-shot) vs fp4c::enc_pair_f on random floats incl. specials.
__global__ void t_enc_pair_f(unsigned long long n, uint64_t seed, unsigned long long* bad) {
  unsigned long long b = 0;
  for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
    uint64_t s = seed ^ (i * 0xd1b54a32d192ed03ull);
    float f[32]; const uint32_t mode = rng(s) & 3;
    for (int k = 0; k < 32; k++) {
      uint32_t u = rng(s); if (((u >> 23) & 0xff) == 0xff) u &= 0xbfffffff;
      if (mode == 1) u = (u & 0x807fffff) | (0x3f000000 + ((rng(s) & 7) << 23));
      if (mode == 2) u &= 0x80ffffff;
      f[k] = __uint_as_float(u);
    }
    float am = 0.f; for (int k = 0; k < 32; k++) am = fmaxf(am, fabsf(f[k]));
    float inv, rs; row_scales(am, inv, rs);
    uint4 q1, q2; uint16_t sf1; enc_pair_f(f, inv, q1, sf1);
    const uint32_t sf2 = fp4f::enc_pair_f_fast<false>(f, inv, q2);
    b += (q1.x != q2.x) | (q1.y != q2.y) | (q1.z != q2.z) | (q1.w != q2.w) | (sf1 != sf2);
  }
  if (b) atomicAdd(bad, b);
}
// T5: decode-accumulate vs fp4c::dec_acc: random nibbles, every scale byte 0x00..0x7F, random rs (all exponents) and acc.
__global__ void t_dec(unsigned long long n, uint64_t seed, unsigned long long* bad) {
  unsigned long long b = 0;
  for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
    uint64_t s = seed ^ (i * 0x94d049bb133111ebull);
    const uint4 q = make_uint4(rng(s), rng(s), rng(s), rng(s));
    const uint32_t sf = (uint32_t)(i & 0x7f) | ((rng(s) & 0x7f) << 8);
    uint32_t ru = rng(s) & 0x7fffffff; if ((ru >> 23) == 0xff) ru &= 0x7f7fffff;
    if (i & 0x100) ru = (ru & 0x007fffff) | 0x3a000000;  // realistic rs ~ 1e-3
    const float rs = __uint_as_float(ru);
    float a1[32], a2[32];
    for (int k = 0; k < 32; k++) {
      uint32_t u = rng(s); if (((u >> 23) & 0xff) == 0xff) u &= 0xbfffffff;
      if (i & 0x200) u = 0;
      a1[k] = a2[k] = __uint_as_float(u);
    }
    dec_acc(a1, q, (uint16_t)sf, rs); fp4f::dec_acc_fast(a2, q, sf, rs);
    for (int k = 0; k < 32; k++) b += !((__float_as_uint(a1[k]) == __float_as_uint(a2[k])) || (isnan(a1[k]) && isnan(a2[k])));
  }
  if (b) atomicAdd(bad, b);
}

// ============================== row kernels ==============================
// -- encode, registers, warp per row
template <bool CANON, int LDM>
__global__ void __launch_bounds__(256, 2) k_enc_reg(const bf16* x, Slots o, int R) {
  const int lane = threadIdx.x & 31; const ref::Slot s = sl(o, 0);
  for (int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); row < R; row += gridDim.x * (blockDim.x >> 5))
    fp4f::enc_row_reg<CANON, LDM>(x + (size_t)row * H, lane, erow(s, row));
}
// TMA ring shared by the kernels below: NST stages, one producer lane (last warp), stage-owned consumer warp groups
// (group g = warp / WPS consumes every use of stage g, so mbarrier parities are always exact).
struct Ring {
  uint64_t *full, *empty;
  __device__ void init(char* sm, int nst, int wps) {
    full = (uint64_t*)sm; empty = full + nst;
    if (threadIdx.x == 0) { for (int i = 0; i < nst; i++) { fp4f::mbar_init(full + i, 1); fp4f::mbar_init(empty + i, wps); } fp4f::mbar_fence_init(); }
    __syncthreads();
  }
};
// -- encode, TMA ring: NEW encoder warps, RPS rows per stage
template <int RPS, int NST, int NEW, bool CANON, bool LOWREG = true>
__global__ void __launch_bounds__(32 * (NEW + 1), 1) k_enc_tma(const bf16* x, Slots o, int R) {
  extern __shared__ __align__(128) char sm[];
  constexpr int WPS = NEW / NST; static_assert(NEW % NST == 0, "");
  Ring rg; rg.init(sm, NST, WPS);
  bf16* buf = (bf16*)(sm + 512);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nt = (R + RPS - 1) / RPS;
  if (warp == NEW) {
    if (lane == 0) {
      int it = 0;
      for (int t = blockIdx.x; t < nt; t += gridDim.x, it++) {
        const int st = it % NST;
        if (it >= NST) fp4f::mbar_wait(rg.empty + st, ((it / NST) & 1) ^ 1);
        const uint32_t bytes = min(RPS, R - t * RPS) * H * 2;
        fp4f::mbar_expect_tx(rg.full + st, bytes);
        fp4f::tma_g2s(buf + (size_t)st * RPS * H, x + (size_t)t * RPS * H, bytes, rg.full + st);
      }
    }
    return;
  }
  const ref::Slot s = sl(o, 0); const int g = warp / WPS, lw = warp % WPS;
  for (int it = g;; it += NST) {
    const int t = blockIdx.x + it * gridDim.x; if (t >= nt) break;
    fp4f::mbar_wait(rg.full + g, (it / NST) & 1);
    const int rows = min(RPS, R - t * RPS);
    for (int r = lw; r < rows; r += WPS) fp4f::enc_row_smem<CANON, LOWREG>(buf + ((size_t)g * RPS + r) * H, lane, erow(s, t * RPS + r));
    __syncwarp();
    if (lane == 0) fp4f::mbar_arrive(rg.empty + g);
  }
}
// -- decode-sum from global, warp per row (STG: smem-staged coalesced output)
template <int NS, int U, bool STG>
__global__ void __launch_bounds__(256) k_dec_glb(Slots in, int R, bf16* out) {
  __shared__ __align__(16) uint4 stb[8][128];
  const int lane = threadIdx.x & 31;
  for (int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); row < R; row += gridDim.x * (blockDim.x >> 5)) {
    fp4f::SrcRow src[NS];
#pragma unroll
    for (int s = 0; s < NS; s++) { const ref::Slot q = sl(in, s); src[s] = fp4f::SrcRow{q.q(row), q.sf(row), __ldcg(q.rs() + row)}; }
    fp4f::dec_row<NS, U, 0>(src, lane, out + (size_t)row * H, STG ? stb[threadIdx.x >> 5] : nullptr);
  }
}
// -- decode-sum from smem: TMA ring of NS sources x RPS rows, NDW decoder warps
template <int NS, int RPS, int NST, int NDW, int U, bool STG>
__global__ void __launch_bounds__(32 * (NDW + 1), 1) k_dec_tma(Slots in, int R, bf16* out) {
  extern __shared__ __align__(128) char sm[];
  constexpr int WPS = NDW / NST; static_assert(NDW % NST == 0, "");
  constexpr int STB = RPS * NS * (VB + SB);
  Ring rg; rg.init(sm, NST, WPS);
  char* buf = sm + 512;
  uint4* stg = (uint4*)(buf + NST * STB);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nt = (R + RPS - 1) / RPS;
  if (warp == NDW) {
    if (lane == 0) {
      int it = 0;
      for (int t = blockIdx.x; t < nt; t += gridDim.x, it++) {
        const int st = it % NST;
        if (it >= NST) fp4f::mbar_wait(rg.empty + st, ((it / NST) & 1) ^ 1);
        const int rows = min(RPS, R - t * RPS);
        fp4f::mbar_expect_tx(rg.full + st, rows * NS * (VB + SB));
        char* b = buf + (size_t)st * STB;
#pragma unroll
        for (int s = 0; s < NS; s++) {
          const ref::Slot q = sl(in, s);
          fp4f::tma_g2s(b + s * RPS * VB, q.q(t * RPS), rows * VB, rg.full + st);
          fp4f::tma_g2s(b + NS * RPS * VB + s * RPS * SB, q.sf(t * RPS), rows * SB, rg.full + st);
        }
      }
    }
    return;
  }
  const int g = warp / WPS, lw = warp % WPS;
  for (int it = g;; it += NST) {
    const int t = blockIdx.x + it * gridDim.x; if (t >= nt) break;
    const int rows = min(RPS, R - t * RPS);
    fp4f::mbar_wait(rg.full + g, (it / NST) & 1);
    const char* b = buf + (size_t)g * STB;
    for (int r = lw; r < rows; r += WPS) {
      const int row = t * RPS + r;
      fp4f::SrcRow src[NS];
#pragma unroll
      for (int s = 0; s < NS; s++)
        src[s] = fp4f::SrcRow{(const uint4*)(b + s * RPS * VB + r * VB), (const uint16_t*)(b + NS * RPS * VB + s * RPS * SB + r * SB),
                              __ldcg(sl(in, s).rs() + row)};
      fp4f::dec_row<NS, U, 1>(src, lane, out + (size_t)row * H, STG ? stg + 128 * warp : nullptr);
    }
    __syncwarp();
    if (lane == 0) fp4f::mbar_arrive(rg.empty + g);
  }
}
// -- one-shot compute path, registers/global: encode own row -> slot 0 (global), decode-sum slots 0..NS-1 -> out
template <int NS, int U>
__global__ void __launch_bounds__(256) k_os_reg(const bf16* x, Slots sl_, int R, bf16* out) {
  __shared__ __align__(16) uint4 stb[8][128];
  const int lane = threadIdx.x & 31;
  for (int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); row < R; row += gridDim.x * (blockDim.x >> 5)) {
    fp4f::enc_row_reg<true>(x + (size_t)row * H, lane, erow(sl(sl_, 0), row));
    __syncwarp();
    fp4f::SrcRow src[NS];
#pragma unroll
    for (int s = 0; s < NS; s++) { const ref::Slot q = sl(sl_, s); src[s] = fp4f::SrcRow{q.q(row), q.sf(row), __ldcg(q.rs() + row)}; }
    fp4f::dec_row<NS, U, 0>(src, lane, out + (size_t)row * H, stb[threadIdx.x >> 5]);
  }
}
// -- one-shot compute path, TMA: stage = RPS rows of x + peer sources 1..NS-1; own encoding goes to smem (source 0)
//    and to global slot 0 (in the fused kernel: to the peers); decode-sum from smem; staged coalesced output.
// PEERG: peer sources are read from global (as in a push design: peers wrote them into the local window), only x is TMA'd.
template <int NS, int RPS, int NST, int NW, int U, bool PEERG = false>
__global__ void __launch_bounds__(32 * (NW + 1), 1) k_os_tma(const bf16* x, Slots sl_, int R, bf16* out) {
  extern __shared__ __align__(128) char sm[];
  constexpr int WPS = NW / NST; static_assert(NW % NST == 0, "");
  constexpr int NSM = PEERG ? 1 : NS;
  constexpr int XB = RPS * H * 2, QB = RPS * NSM * VB, STB = XB + QB + RPS * NSM * SB;
  Ring rg; rg.init(sm, NST, WPS);
  char* buf = sm + 512;
  uint4* stg = (uint4*)(buf + NST * STB);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nt = (R + RPS - 1) / RPS;
  if (warp == NW) {
    if (lane == 0) {
      int it = 0;
      for (int t = blockIdx.x; t < nt; t += gridDim.x, it++) {
        const int st = it % NST;
        if (it >= NST) fp4f::mbar_wait(rg.empty + st, ((it / NST) & 1) ^ 1);
        const int rows = min(RPS, R - t * RPS);
        fp4f::mbar_expect_tx(rg.full + st, rows * (H * 2 + (NSM - 1) * (VB + SB)));
        char* b = buf + (size_t)st * STB;
        fp4f::tma_g2s(b, x + (size_t)t * RPS * H, rows * H * 2, rg.full + st);
#pragma unroll
        for (int s = 1; s < NSM; s++) {
          const ref::Slot q = sl(sl_, s);
          fp4f::tma_g2s(b + XB + s * RPS * VB, q.q(t * RPS), rows * VB, rg.full + st);
          fp4f::tma_g2s(b + XB + QB + s * RPS * SB, q.sf(t * RPS), rows * SB, rg.full + st);
        }
      }
    }
    return;
  }
  const int g = warp / WPS, lw = warp % WPS;
  for (int it = g;; it += NST) {
    const int t = blockIdx.x + it * gridDim.x; if (t >= nt) break;
    const int rows = min(RPS, R - t * RPS);
    fp4f::mbar_wait(rg.full + g, (it / NST) & 1);
    char* b = buf + (size_t)g * STB;
    for (int r = lw; r < rows; r += WPS) {
      const int row = t * RPS + r;
      uint4* oq = (uint4*)(b + XB + r * VB); uint16_t* osf = (uint16_t*)(b + XB + QB + r * SB);
      const float rs0 = fp4f::enc_row_smem<true>((const bf16*)(b + (size_t)r * H * 2), lane, fp4f::EncRow{oq, osf, nullptr});
      __syncwarp();
      {
        const ref::Slot q = sl(sl_, 0);
#pragma unroll
        for (int j = 0; j < 6; j++) { q.q(row)[lane + 32 * j] = oq[lane + 32 * j]; }
        if (lane < 24) ((uint4*)q.sf(row))[lane] = ((const uint4*)osf)[lane];
        if (lane == 0) q.rs()[row] = rs0;
      }
      fp4f::SrcRow src[NS];
#pragma unroll
      for (int s = 0; s < NS; s++)
        src[s] = (PEERG && s > 0) ? fp4f::SrcRow{sl(sl_, s).q(row), sl(sl_, s).sf(row), __ldcg(sl(sl_, s).rs() + row)} :
                 fp4f::SrcRow{(const uint4*)(b + XB + s * RPS * VB + r * VB), (const uint16_t*)(b + XB + QB + s * RPS * SB + r * SB),
                              s == 0 ? rs0 : __ldcg(sl(sl_, s).rs() + row)};
      fp4f::dec_row<NS, U, 1>(src, lane, out + (size_t)row * H, stg + 128 * warp);
    }
    __syncwarp();
    if (lane == 0) fp4f::mbar_arrive(rg.empty + g);
  }
}
// -- one-shot compute path, warp-specialized: per stage group g: WE encoder warps (own row -> smem source 0 + global
//    slot 0), then WD decoder warps (decode-sum from smem). Barriers per stage: full (TMA), encd (WE), empty (WD).
template <int NS, int RPS, int NST, int WE, int WD, int U>
__global__ void __launch_bounds__(32 * (NST * (WE + WD) + 1), 1) k_os_ws(const bf16* x, Slots sl_, int R, bf16* out) {
  extern __shared__ __align__(128) char sm[];
  constexpr int XB = RPS * H * 2, QB = RPS * NS * VB, STB = XB + QB + RPS * NS * SB, NWK = NST * (WE + WD);
  uint64_t* full = (uint64_t*)sm; uint64_t* encd = full + NST; uint64_t* empty = encd + NST;
  __shared__ float s_rs0[NST][RPS];
  if (threadIdx.x == 0) {
    for (int i = 0; i < NST; i++) { fp4f::mbar_init(full + i, 1); fp4f::mbar_init(encd + i, WE); fp4f::mbar_init(empty + i, WD); }
    fp4f::mbar_fence_init();
  }
  __syncthreads();
  char* buf = sm + 512;
  uint4* stg = (uint4*)(buf + NST * STB);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nt = (R + RPS - 1) / RPS;
  if (warp == NWK) {
    if (lane == 0) {
      int it = 0;
      for (int t = blockIdx.x; t < nt; t += gridDim.x, it++) {
        const int st = it % NST;
        if (it >= NST) fp4f::mbar_wait(empty + st, ((it / NST) & 1) ^ 1);
        const int rows = min(RPS, R - t * RPS);
        fp4f::mbar_expect_tx(full + st, rows * (H * 2 + (NS - 1) * (VB + SB)));
        char* b = buf + (size_t)st * STB;
        fp4f::tma_g2s(b, x + (size_t)t * RPS * H, rows * H * 2, full + st);
#pragma unroll
        for (int s = 1; s < NS; s++) {
          const ref::Slot q = sl(sl_, s);
          fp4f::tma_g2s(b + XB + s * RPS * VB, q.q(t * RPS), rows * VB, full + st);
          fp4f::tma_g2s(b + XB + QB + s * RPS * SB, q.sf(t * RPS), rows * SB, full + st);
        }
      }
    }
    return;
  }
  const bool isenc = warp < NST * WE;
  const int g = isenc ? warp / WE : (warp - NST * WE) / WD, lw = isenc ? warp % WE : (warp - NST * WE) % WD;
  for (int it = g;; it += NST) {
    const int t = blockIdx.x + it * gridDim.x; if (t >= nt) break;
    const int rows = min(RPS, R - t * RPS);
    const uint32_t ph = (it / NST) & 1;
    char* b = buf + (size_t)g * STB;
    if (isenc) {
      fp4f::mbar_wait(full + g, ph);
      for (int r = lw; r < rows; r += WE) {
        const int row = t * RPS + r;
        uint4* oq = (uint4*)(b + XB + r * VB); uint16_t* osf = (uint16_t*)(b + XB + QB + r * SB);
        const float rs0 = fp4f::enc_row_smem<true>((const bf16*)(b + (size_t)r * H * 2), lane, fp4f::EncRow{oq, osf, &s_rs0[g][r]});
        __syncwarp();
        const ref::Slot q = sl(sl_, 0);  // own encoding -> global (fused kernel: -> peers)
#pragma unroll
        for (int j = 0; j < 6; j++) q.q(row)[lane + 32 * j] = oq[lane + 32 * j];
        if (lane < 24) ((uint4*)q.sf(row))[lane] = ((const uint4*)osf)[lane];
        if (lane == 0) q.rs()[row] = rs0;
      }
      __syncwarp();
      if (lane == 0) fp4f::mbar_arrive(encd + g);  // release orders the smem writes (and s_rs0) for the decoders
    } else {
      fp4f::mbar_wait(encd + g, ph);
      for (int r = lw; r < rows; r += WD) {
        const int row = t * RPS + r;
        fp4f::SrcRow src[NS];
#pragma unroll
        for (int s = 0; s < NS; s++)
          src[s] = fp4f::SrcRow{(const uint4*)(b + XB + s * RPS * VB + r * VB), (const uint16_t*)(b + XB + QB + s * RPS * SB + r * SB),
                                s == 0 ? s_rs0[g][r] : __ldcg(sl(sl_, s).rs() + row)};
        fp4f::dec_row<NS, U, 1>(src, lane, out + (size_t)row * H, stg + 128 * (warp - NST * WE));
      }
      __syncwarp();
      if (lane == 0) fp4f::mbar_arrive(empty + g);
    }
  }
}
// -- two-shot phase B (owner rows [lo, hi)): TMA NS sources -> smem, red_reenc_row -> slot2 (global) + out
template <int NS, int RPS, int NST, int NW>
__global__ void __launch_bounds__(32 * (NW + 1), 1) k_ts_owner(Slots in, Slots s2, int lo, int hi, bf16* out) {
  extern __shared__ __align__(128) char sm[];
  constexpr int WPS = NW / NST; static_assert(NW % NST == 0, "");
  constexpr int STB = RPS * NS * (VB + SB);
  Ring rg; rg.init(sm, NST, WPS);
  char* buf = sm + 512;
  uint4* stg = (uint4*)(buf + NST * STB);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, R = hi - lo, nt = (R + RPS - 1) / RPS;
  if (warp == NW) {
    if (lane == 0) {
      int it = 0;
      for (int t = blockIdx.x; t < nt; t += gridDim.x, it++) {
        const int st = it % NST;
        if (it >= NST) fp4f::mbar_wait(rg.empty + st, ((it / NST) & 1) ^ 1);
        const int rows = min(RPS, R - t * RPS), r0 = lo + t * RPS;
        fp4f::mbar_expect_tx(rg.full + st, rows * NS * (VB + SB));
        char* b = buf + (size_t)st * STB;
#pragma unroll
        for (int s = 0; s < NS; s++) {
          const ref::Slot q = sl(in, s);
          fp4f::tma_g2s(b + s * RPS * VB, q.q(r0), rows * VB, rg.full + st);
          fp4f::tma_g2s(b + NS * RPS * VB + s * RPS * SB, q.sf(r0), rows * SB, rg.full + st);
        }
      }
    }
    return;
  }
  const int g = warp / WPS, lw = warp % WPS; const ref::Slot o2 = sl(s2, 0);
  for (int it = g;; it += NST) {
    const int t = blockIdx.x + it * gridDim.x; if (t >= nt) break;
    const int rows = min(RPS, R - t * RPS);
    fp4f::mbar_wait(rg.full + g, (it / NST) & 1);
    const char* b = buf + (size_t)g * STB;
    for (int r = lw; r < rows; r += WPS) {
      const int row = lo + t * RPS + r;
      fp4f::SrcRow src[NS];
#pragma unroll
      for (int s = 0; s < NS; s++)
        src[s] = fp4f::SrcRow{(const uint4*)(b + s * RPS * VB + r * VB), (const uint16_t*)(b + NS * RPS * VB + s * RPS * SB + r * SB),
                              __ldcg(sl(in, s).rs() + row)};
      fp4f::red_reenc_row<NS, 1, true>(src, lane, erow(o2, row), out + (size_t)row * H, stg + 128 * warp);
    }
    __syncwarp();
    if (lane == 0) fp4f::mbar_arrive(rg.empty + g);
  }
}
// -- two-shot phase B, split: 6 warps per row (single pass, sum in registers); NST stages x RPS rows, group = 6*RPS warps
template <int NS, int RPS, int NST>
__global__ void __launch_bounds__(32 * (6 * RPS * NST + 1), 1) k_ts_owner6(Slots in, Slots s2, int lo, int hi, bf16* out) {
  extern __shared__ __align__(128) char sm[];
  constexpr int WPS = 6 * RPS, NW = WPS * NST, STB = RPS * NS * (VB + SB);
  static_assert(RPS * NST <= 15, "named barriers");
  Ring rg; rg.init(sm, NST, WPS);
  __shared__ float red[NST * RPS][6];
  char* buf = sm + 512;
  uint4* stg = (uint4*)(buf + NST * STB);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, R = hi - lo, nt = (R + RPS - 1) / RPS;
  if (warp == NW) {
    if (lane == 0) {
      int it = 0;
      for (int t = blockIdx.x; t < nt; t += gridDim.x, it++) {
        const int st = it % NST;
        if (it >= NST) fp4f::mbar_wait(rg.empty + st, ((it / NST) & 1) ^ 1);
        const int rows = min(RPS, R - t * RPS), r0 = lo + t * RPS;
        fp4f::mbar_expect_tx(rg.full + st, rows * NS * (VB + SB));
        char* b = buf + (size_t)st * STB;
#pragma unroll
        for (int s = 0; s < NS; s++) {
          const ref::Slot q = sl(in, s);
          fp4f::tma_g2s(b + s * RPS * VB, q.q(r0), rows * VB, rg.full + st);
          fp4f::tma_g2s(b + NS * RPS * VB + s * RPS * SB, q.sf(r0), rows * SB, rg.full + st);
        }
      }
    }
    return;
  }
  const int g = warp / WPS, lw = warp % WPS, r = lw / 6, j = lw % 6; const ref::Slot o2 = sl(s2, 0);
  for (int it = g;; it += NST) {
    const int t = blockIdx.x + it * gridDim.x; if (t >= nt) break;
    const int rows = min(RPS, R - t * RPS);
    fp4f::mbar_wait(rg.full + g, (it / NST) & 1);
    const char* b = buf + (size_t)g * STB;
    if (r < rows) {
      const int row = lo + t * RPS + r;
      fp4f::SrcRow src[NS];
#pragma unroll
      for (int s = 0; s < NS; s++)
        src[s] = fp4f::SrcRow{(const uint4*)(b + s * RPS * VB + r * VB), (const uint16_t*)(b + NS * RPS * VB + s * RPS * SB + r * SB),
                              __ldcg(sl(in, s).rs() + row)};
      fp4f::red_reenc_row_split<NS, 1, true>(src, lane, j, red[g * RPS + r], 1 + g * RPS + r, erow(o2, row), out + (size_t)row * H, stg + 128 * warp);
    }
    __syncwarp();
    if (lane == 0) fp4f::mbar_arrive(rg.empty + g);
  }
}
// -- two-shot phase C (non-owner rows): decode the single owner source (slot2) -> out; rows [lo, hi) skipped
template <int RPS, int NST, int NDW>
__global__ void __launch_bounds__(32 * (NDW + 1), 1) k_ts_gather(Slots s2, int R, int lo, int hi, bf16* out) {
  extern __shared__ __align__(128) char sm[];
  constexpr int WPS = NDW / NST; constexpr int STB = RPS * (VB + SB);
  Ring rg; rg.init(sm, NST, WPS);
  char* buf = sm + 512;
  uint4* stg = (uint4*)(buf + NST * STB);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int Rn = R - (hi - lo), nt = (Rn + RPS - 1) / RPS;  // compacted index space over the non-owned rows
  auto map = [&](int i) { return i < lo ? i : i + (hi - lo); };
  if (warp == NDW) {
    if (lane == 0) {
      int it = 0;
      for (int t = blockIdx.x; t < nt; t += gridDim.x, it++) {
        const int st = it % NST;
        if (it >= NST) fp4f::mbar_wait(rg.empty + st, ((it / NST) & 1) ^ 1);
        const int rows = min(RPS, Rn - t * RPS);
        fp4f::mbar_expect_tx(rg.full + st, rows * (VB + SB));
        char* b = buf + (size_t)st * STB; const ref::Slot q = sl(s2, 0);
        for (int r = 0; r < rows; r++) {  // rows may straddle the owned gap: per-row copies
          const int row = map(t * RPS + r);
          fp4f::tma_g2s(b + r * VB, q.q(row), VB, rg.full + st);
          fp4f::tma_g2s(b + RPS * VB + r * SB, q.sf(row), SB, rg.full + st);
        }
      }
    }
    return;
  }
  const int g = warp / WPS, lw = warp % WPS;
  for (int it = g;; it += NST) {
    const int t = blockIdx.x + it * gridDim.x; if (t >= nt) break;
    const int rows = min(RPS, Rn - t * RPS);
    fp4f::mbar_wait(rg.full + g, (it / NST) & 1);
    const char* b = buf + (size_t)g * STB;
    for (int r = lw; r < rows; r += WPS) {
      const int row = map(t * RPS + r);
      fp4f::SrcRow src{(const uint4*)(b + r * VB), (const uint16_t*)(b + RPS * VB + r * SB), __ldcg(sl(s2, 0).rs() + row)};
      fp4f::dec_row<1, 3, 1>(&src, lane, out + (size_t)row * H, stg + 128 * warp);
    }
    __syncwarp();
    if (lane == 0) fp4f::mbar_arrive(rg.empty + g);
  }
}
// -- two-shot compute path (warp per row, global): encode own row -> slot 0; if row in [own_lo, own_hi): decode-sum NS
//    sources twice (amax pass, encode pass), re-encode -> slot2, write decode of the re-encoding; else decode slot2 -> out.
template <int NS>
__global__ void __launch_bounds__(256) k_ts_reg(const bf16* x, Slots sl_, Slots s2, int R, int own_lo, int own_hi, bf16* out) {
  __shared__ __align__(16) uint4 stb[8][128];
  const int lane = threadIdx.x & 31; uint4* st = stb[threadIdx.x >> 5];
  for (int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); row < R; row += gridDim.x * (blockDim.x >> 5)) {
    fp4f::enc_row_reg<true>(x + (size_t)row * H, lane, erow(sl(sl_, 0), row));
    __syncwarp();
    const ref::Slot o2 = sl(s2, 0);
    if (row >= own_lo && row < own_hi) {
      fp4f::SrcRow src[NS];
#pragma unroll
      for (int s = 0; s < NS; s++) { const ref::Slot q = sl(sl_, s); src[s] = fp4f::SrcRow{q.q(row), q.sf(row), __ldcg(q.rs() + row)}; }
      float am = 0.f;
#pragma unroll 1
      for (int j = 0; j < 6; j++) {
        const int pp = lane + 32 * j; float acc[32]; zero32(acc);
#pragma unroll
        for (int s = 0; s < NS; s++) fp4f::dec_acc_fast(acc, __ldcg(src[s].q + pp), __ldcg(src[s].sf + pp), src[s].rs);
        am = amax32(acc, am);
      }
      float inv, rs2; row_scales(warp_amax(am), inv, rs2);
#pragma unroll 1
      for (int j = 0; j < 6; j++) {
        const int pp = lane + 32 * j; float acc[32]; zero32(acc);
#pragma unroll
        for (int s = 0; s < NS; s++) fp4f::dec_acc_fast(acc, __ldcg(src[s].q + pp), __ldcg(src[s].sf + pp), src[s].rs);
        uint4 q; const uint32_t sf = fp4f::enc_pair_f_fast<true>(acc, inv, q);
        o2.q(row)[pp] = q; o2.sf(row)[pp] = (uint16_t)sf;
        zero32(acc); fp4f::dec_acc_fast(acc, q, sf, rs2);
        uint4 o4[4]; fp4f::pack_bf16x32(acc, o4);
        fp4f::store_block_staged(st, lane, o4, (uint4*)(out + (size_t)row * H) + 128 * j);
      }
      if (lane == 0) o2.rs()[row] = rs2;
    } else {
      fp4f::SrcRow src{o2.q(row), o2.sf(row), __ldcg(o2.rs() + row)};
      fp4f::dec_row<1, 3, 0>(&src, lane, out + (size_t)row * H, st);
    }
  }
}
// copy roofline
__global__ void __launch_bounds__(512) k_copy(const uint4* __restrict__ a, uint4* __restrict__ b, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, st = (size_t)gridDim.x * blockDim.x;
  for (; i + 3 * st < n; i += 4 * st) {
    uint4 v0 = fp4f::ld_nc_na(a + i), v1 = fp4f::ld_nc_na(a + i + st), v2 = fp4f::ld_nc_na(a + i + 2 * st), v3 = fp4f::ld_nc_na(a + i + 3 * st);
    b[i] = v0; b[i + st] = v1; b[i + 2 * st] = v2; b[i + 3 * st] = v3;
  }
  for (; i < n; i += st) b[i] = a[i];
}
// one-shot ref from slots with canonicalized sources already in place
__global__ void k_canon(uint4* q, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) q[i] = canon_neg0(q[i]);
}


// ============================== CPU oracle (independent of every GPU codec) ==============================
namespace cpu {
static float bf(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t to_bf16(float f) {  // RN-even
  uint32_t u; memcpy(&u, &f, 4);
  if ((u & 0x7fffffff) > 0x7f800000) return (uint16_t)((u >> 16) | 0x40);
  u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16);
}
static float e4v[128];
static void init() { for (int b = 0; b < 128; b++) { int e = b >> 3, m = b & 7; e4v[b] = e ? ldexpf(1.f + m / 8.f, e - 7) : ldexpf(m / 8.f, -6); } e4v[127] = NAN; }
static int e4m3_enc(float x) {  // x >= 0 or NaN; RN-even, satfinite
  if (std::isnan(x)) return 0x7f;
  if (x >= 448.f) return 0x7e;
  int best = 0;
  for (int b = 1; b < 127; b++) {
    const double d0 = fabs((double)x - e4v[best]), d1 = fabs((double)x - e4v[b]);
    if (d1 < d0 || (d1 == d0 && (b % 2 == 0))) best = b;
    if (e4v[b] > x) break;
  }
  return best;
}
static const float e2v[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -.5f, -1, -1.5f, -2, -3, -4, -6};
static int e2m1_enc(float x) {  // RN-even, satfinite; NaN -> 0x7 (measured cvt.rn.satfinite.e2m1x2.f32 on sm_100a)
  if (std::isnan(x)) return 7;
  const float a = fabsf(x); const int sg = std::signbit(x) ? 8 : 0;
  if (a >= 6) return sg | 7;
  int best = 0;
  for (int i = 1; i < 8; i++) { const float d0 = fabsf(a - e2v[best]), d1 = fabsf(a - e2v[i]); if (d1 < d0 || (d1 == d0 && i % 2 == 0)) best = i; }
  return sg | best;
}
// Encode one row of H floats into (3072 value bytes, 384 scale bytes, rs).
static void enc_row(const float* f, uint8_t* q, uint8_t* sf, float* rs, bool canon) {
  float am = 0.f; for (int i = 0; i < H; i++) am = std::max(am, fabsf(f[i]));
  const float inv = am > 0.f ? 2688.f / am : 0.f; *rs = am / 2688.f;
  for (int b = 0; b < H / 16; b++) {
    float bm = 0.f; for (int i = 0; i < 16; i++) bm = std::max(bm, fabsf(f[16 * b + i]));
    const int s = e4m3_enc((bm * inv) / 6.f); sf[b] = (uint8_t)s;
    const float sff = e4v[s], m = sff > 0.f ? inv / sff : 0.f;
    for (int i = 0; i < 16; i += 2) {
      int n0 = e2m1_enc(f[16 * b + i] * m), n1 = e2m1_enc(f[16 * b + i + 1] * m);
      if (canon) { if (n0 == 8) n0 = 0; if (n1 == 8) n1 = 0; }
      q[(16 * b + i) / 2] = (uint8_t)(n0 | (n1 << 4));
    }
  }
}
static void dec_acc_row(float* acc, const uint8_t* q, const uint8_t* sf, float rs) {
  for (int b = 0; b < H / 16; b++) {
    const float S = e4v[sf[b]] * rs;
    for (int i = 0; i < 16; i++) { const int n = (q[(16 * b + i) / 2] >> (4 * (i & 1))) & 15; acc[16 * b + i] = fmaf(e2v[n], S, acc[16 * b + i]); }
  }
}
// slot layout as ref::Slot: [R][VB] q, [R][SB] sf, [R] rs
static void enc_slot(const std::vector<uint16_t>& x, int R, std::vector<uint8_t>& slot, bool canon) {
  slot.assign((size_t)R * ROW_BYTES, 0);
#pragma omp parallel for
  for (int r = 0; r < R; r++) {
    std::vector<float> f(H); for (int i = 0; i < H; i++) f[i] = bf(x[(size_t)r * H + i]);
    enc_row(f.data(), &slot[(size_t)r * VB], &slot[(size_t)R * VB + (size_t)r * SB], (float*)&slot[(size_t)R * (VB + SB)] + r, canon);
  }
}
static void oneshot(const std::vector<uint8_t>* sl, int P, int R, std::vector<uint16_t>& out) {
  out.assign((size_t)R * H, 0);
#pragma omp parallel for
  for (int r = 0; r < R; r++) {
    std::vector<float> acc(H, 0.f);
    for (int s = 0; s < P; s++) dec_acc_row(acc.data(), &sl[s][(size_t)r * VB], &sl[s][(size_t)R * VB + (size_t)r * SB], ((const float*)&sl[s][(size_t)R * (VB + SB)])[r]);
    for (int i = 0; i < H; i++) out[(size_t)r * H + i] = to_bf16(acc[i]);
  }
}
static void twoshot(const std::vector<uint8_t>* sl, int P, int R, std::vector<uint16_t>& out) {
  out.assign((size_t)R * H, 0);
#pragma omp parallel for
  for (int r = 0; r < R; r++) {
    std::vector<float> acc(H, 0.f), a2(H, 0.f);
    for (int s = 0; s < P; s++) dec_acc_row(acc.data(), &sl[s][(size_t)r * VB], &sl[s][(size_t)R * VB + (size_t)r * SB], ((const float*)&sl[s][(size_t)R * (VB + SB)])[r]);
    std::vector<uint8_t> q(VB), sf(SB); float rs;
    enc_row(acc.data(), q.data(), sf.data(), &rs, false);
    dec_acc_row(a2.data(), q.data(), sf.data(), rs);
    for (int i = 0; i < H; i++) out[(size_t)r * H + i] = to_bf16(a2[i]);
  }
}
}  // namespace cpu

// ============================== host ==============================
static int g_sms;
static char* g_flush; static size_t g_flush_bytes = 512ull << 20;
static bool g_cold = true;
static float time_med(std::function<void()> f, int iters = 25, bool cold = true) {
  cold = cold && g_cold;
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  std::vector<float> v;
  for (int i = 0; i < 3; i++) f();
  for (int i = 0; i < iters; i++) {
    if (cold) CK(cudaMemsetAsync(g_flush, i & 0xff, g_flush_bytes));
    CK(cudaEventRecord(a)); f(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); v.push_back(ms * 1000.f);
  }
  CK(cudaGetLastError());
  std::sort(v.begin(), v.end()); return v[v.size() / 2];
}
template <typename K> static void set_smem(K k, int bytes) { CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)); }
static int nregs(const void* k) { cudaFuncAttributes a; CK(cudaFuncGetAttributes(&a, k)); return a.numRegs; }

struct Bufs {
  int R; bf16* x[MAXS]; char* slot[MAXS]; char* slot2; bf16 *out, *ref; char* tmp;
};
static size_t SLB(int R) { return ref::slot_bytes(R); }

// Host-side special rows: written over the Gaussian input of source `src` at rows r % 64 == k (k < 12).
static void make_specials(std::vector<uint16_t>& h, int R, int src) {
  std::mt19937 g(1234 + src);
  auto rbits = [&]() { uint16_t u = g() & 0xffff; if (((u >> 7) & 0xff) == 0xff) u &= 0xbfff; return u; };
  for (int row = 0; row < R; row++) {
    const int k = (row + 7 * src) % 64; uint16_t* p = h.data() + (size_t)row * H;
    switch (k) {
      case 0: for (int i = 0; i < H; i++) p[i] = 0; break;                                   // all zero (amax 0)
      case 1: for (int i = 0; i < H; i++) p[i] = 0x8000; break;                              // all -0
      case 2: for (int i = 0; i < H; i++) p[i] = rbits(); break;                             // random finite bits
      case 3: for (int i = 0; i < H; i++) p[i] = (g() & 0x807f) | ((g() & 1) << 7); break;   // subnormal/tiny: inv = inf
      case 4: for (int i = 0; i < H; i++) p[i] = (g() & 0x80ff) | 0x7e00 | ((g() & 0x7f) == 0 ? 0x0080 : 0); break;  // huge
      case 5: for (int i = 0; i < H; i++) { int b = (i / 16) % 4; p[i] = b == 0 ? 0 : b == 1 ? (uint16_t)((g() & 0x807f) | 0x0100) : b == 2 ? (uint16_t)((g() & 0x80ff) | 0x4500) : (uint16_t)(g() & 0x8000); } break;
      case 6: for (int i = 0; i < H; i++) p[i] = (g() & 0x80ff) | 0x3f00; p[g() % H] = 0x7149; break;  // one 1e30 outlier: scales underflow
      case 7: for (int i = 0; i < H; i++) { static const uint16_t mid[] = {0x3e80, 0x3f40, 0x3fa0, 0x3fe0, 0x4020, 0x4060, 0x40a0, 0x40d0}; p[i] = mid[g() & 7] | (g() & 0x8000); } break;  // midpoints
      case 8: for (int i = 0; i < H; i++) p[i] = (i % 16 == 0) ? 0x43e0 : (uint16_t)((g() & 0x807f) | 0x3800); break;  // saturation-ish blocks
      case 9: for (int i = 0; i < H; i++) p[i] = (g() & 1) ? 0x0001 : 0x8001; break;          // min subnormal +-
      case 10: for (int i = 0; i < H; i++) p[i] = 0x7f7f ^ (g() & 0x8000); break;             // +-bf16 max
      case 11: for (int i = 0; i < H; i++) p[i] = (g() & 0x80ff) | (uint16_t)((0x30 + (i / 16) % 0x20) << 8); break;  // block-wise exponent ramp
      default: break;
    }
  }
}

static unsigned long long cmp_bytes(const void* a, const void* b, size_t n) {
  std::vector<uint8_t> ha(n), hb(n); CK(cudaMemcpy(ha.data(), a, n, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hb.data(), b, n, cudaMemcpyDeviceToHost));
  unsigned long long m = 0; for (size_t i = 0; i < n; i++) m += ha[i] != hb[i]; return m;
}

int main(int argc, char** argv) {
  std::string what = argc > 1 ? argv[1] : "all";
  std::vector<int> rows_list = {1024, 2048, 4096, 8192, 16384};
  if (argc > 2) { rows_list.clear(); std::string s = argv[2]; size_t p = 0; while (p < s.size()) { size_t c = s.find(',', p); if (c == std::string::npos) c = s.size(); rows_list.push_back(std::stoi(s.substr(p, c - p))); p = c + 1; } }
  if (getenv("HOT")) g_cold = false;
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0)); g_sms = prop.multiProcessorCount;
  printf("# device %s SMs %d smem/blk optin %zu L2 %d MB\n", prop.name, g_sms, prop.sharedMemPerBlockOptin, prop.l2CacheSize >> 20);
  CK(cudaMalloc(&g_flush, g_flush_bytes));
  unsigned long long* dbad; CK(cudaMalloc(&dbad, 16));
  bool allok = true;
  auto unit = [&](const char* name, std::function<void()> f) {
    CK(cudaMemset(dbad, 0, 16)); cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
    f(); cudaEventRecord(b); CK(cudaEventSynchronize(b)); CK(cudaGetLastError());
    float ms; cudaEventElapsedTime(&ms, a, b);
    unsigned long long h; CK(cudaMemcpy(&h, dbad, 8, cudaMemcpyDeviceToHost));
    printf("UNIT %-44s mismatches %llu  (%.0f ms)  %s\n", name, h, ms, h ? "FAIL" : "ok"); allok &= h == 0;
  };
  if (what == "test" || what == "all") {
    unit("div6+e4m3x2 exhaustive (2^31 fp32)", [&] { t_div6<<<g_sms * 8, 256>>>(dbad); });
    { unsigned long long h[2]; CK(cudaMemcpy(h, dbad, 16, cudaMemcpyDeviceToHost));
      printf("INFO div6_rn fp32 differs only for subnormal quotients (x < 6*2^-126): %llu cases, E4M3 byte identical\n", h[1]); }
    unit("div_e4m3 exhaustive (inv in [2^-100,2^100] x 126)", [&] { t_div_e4m3<<<g_sms * 8, 256>>>(dbad); });
    unit("enc_pair_fast bf16 random 2^28 pairs", [&] { t_enc_pair<<<g_sms * 8, 256>>>(1ull << 28, 0x1234, dbad); });
    unit("enc_pair_f_fast fp32 random 2^26 pairs", [&] { t_enc_pair_f<<<g_sms * 8, 256>>>(1ull << 26, 0x777, dbad); });
    unit("dec_acc_fast random 2^28 pairs", [&] { t_dec<<<g_sms * 8, 256>>>(1ull << 28, 0x4242, dbad); });
  }
  const int maxR = *std::max_element(rows_list.begin(), rows_list.end());
  Bufs B; B.R = maxR;
  for (int s = 0; s < 4; s++) { CK(cudaMalloc(&B.x[s], (size_t)maxR * H * 2)); CK(cudaMalloc(&B.slot[s], SLB(maxR))); }
  CK(cudaMalloc(&B.slot2, SLB(maxR))); CK(cudaMalloc(&B.out, (size_t)maxR * H * 2)); CK(cudaMalloc(&B.ref, (size_t)maxR * H * 2));
  CK(cudaMalloc(&B.tmp, SLB(maxR)));
  char* refslot[4]; for (int s = 0; s < 4; s++) CK(cudaMalloc(&refslot[s], SLB(maxR)));

  const int G8 = g_sms * 8;
  // ---- variant tables: name, smem bytes, launcher(R)
  struct Var { std::string name; std::function<void(int, Slots, Slots, bf16*)> run; };
#define SMEM_ENC(RPS, NST) (512 + (NST) * (RPS) * H * 2)
#define SMEM_DEC(NS, RPS, NST, NDW) (512 + (NST) * (RPS) * (NS) * (VB + SB) + (NDW) * 2048)
#define SMEM_OS(NS, RPS, NST, NW) (512 + (NST) * ((RPS) * H * 2 + (RPS) * (NS) * (VB + SB)) + (NW) * 2048)
#define OS_PG(NS, RPS, NST, NW, U) { auto k = k_os_tma<NS, RPS, NST, NW, U, true>; const int smb = SMEM_OS(1, RPS, NST, NW); set_smem(k, smb); \
    printf("# regs os_tma peer-global P%d RPS%d NST%d %dw: %d smem %d\n", NS, RPS, NST, NW, nregs((void*)k), smb); \
    oss[NS].push_back({"oneshot_tma peers-global RPS" #RPS " NST" #NST " " #NW "w", [=](int R, Slots, Slots in, bf16* o) { k<<<g_sms, 32 * (NW + 1), smb>>>(B.x[0], in, R, o); }}); }
#define ENC_TMA(RPS, NST, NEW) ENC_TMA_(RPS, NST, NEW, true, "")
#define ENC_TMA_(RPS, NST, NEW, LR, SFX) { auto k = k_enc_tma<RPS, NST, NEW, true, LR>; set_smem(k, SMEM_ENC(RPS, NST)); \
    printf("# regs enc_tma RPS%d NST%d %dw%s: %d\n", RPS, NST, NEW, SFX, nregs((void*)k)); \
    encs.push_back({std::string("enc_tma RPS" #RPS " NST" #NST " " #NEW "w") + SFX, [=](int R, Slots o, Slots, bf16*) { k<<<g_sms, 32 * (NEW + 1), SMEM_ENC(RPS, NST)>>>(B.x[0], o, R); }}); }
#define DEC_TMA(NS, RPS, NST, NDW, U, STG) { auto k = k_dec_tma<NS, RPS, NST, NDW, U, STG>; set_smem(k, SMEM_DEC(NS, RPS, NST, NDW)); \
    printf("# regs dec_tma P%d RPS%d NST%d %dw U%d stg%d: %d\n", NS, RPS, NST, NDW, U, STG, nregs((void*)k)); \
    decs[NS].push_back({"dec_tma RPS" #RPS " NST" #NST " " #NDW "w U" #U " stg" #STG, [=](int R, Slots, Slots in, bf16* o) { k<<<g_sms, 32 * (NDW + 1), SMEM_DEC(NS, RPS, NST, NDW)>>>(in, R, o); }}); }
#define DEC_GLB(NS, U, STG) { auto k = k_dec_glb<NS, U, STG>; printf("# regs dec_glb P%d U%d stg%d: %d\n", NS, U, STG, nregs((void*)k)); \
    decs[NS].push_back({"dec_glb U" #U " stg" #STG, [=](int R, Slots, Slots in, bf16* o) { k<<<g_sms * 3, 256>>>(in, R, o); }}); }
#define OS_TMA(NS, RPS, NST, NW, U) { auto k = k_os_tma<NS, RPS, NST, NW, U>; set_smem(k, SMEM_OS(NS, RPS, NST, NW)); \
    printf("# regs os_tma P%d RPS%d NST%d %dw: %d\n", NS, RPS, NST, NW, nregs((void*)k)); \
    oss[NS].push_back({"oneshot_tma RPS" #RPS " NST" #NST " " #NW "w", [=](int R, Slots, Slots in, bf16* o) { k<<<g_sms, 32 * (NW + 1), SMEM_OS(NS, RPS, NST, NW)>>>(B.x[0], in, R, o); }}); }
  std::vector<Var> encs, decs[5], oss[5];
  encs.push_back({"enc ref (harness ref_encode)", [=](int R, Slots o, Slots, bf16*) { ref::ref_encode<<<G8, 256>>>(B.x[0], o.b[0], R); }});
  encs.push_back({"enc_reg L1::no_allocate", [=](int R, Slots o, Slots, bf16*) { k_enc_reg<true, 0><<<g_sms * 2, 256>>>(B.x[0], o, R); }});
  encs.push_back({"enc_reg L1-allocating", [=](int R, Slots o, Slots, bf16*) { k_enc_reg<true, 1><<<g_sms * 2, 256>>>(B.x[0], o, R); }});
  ENC_TMA(4, 4, 8) ENC_TMA(4, 4, 16) ENC_TMA(3, 6, 18) ENC_TMA(6, 3, 18) ENC_TMA(2, 9, 18) ENC_TMA(1, 18, 18)
#if !ENC_ONLY
  DEC_GLB(2, 3, false) DEC_GLB(2, 3, true) DEC_GLB(2, 2, true)
  DEC_TMA(2, 8, 2, 16, 3, true) DEC_TMA(2, 4, 4, 16, 3, true) DEC_TMA(2, 8, 2, 8, 3, true)
  DEC_GLB(4, 2, false) DEC_GLB(4, 2, true) DEC_GLB(4, 1, true)
  DEC_TMA(4, 4, 3, 12, 2, true) DEC_TMA(4, 2, 4, 8, 2, true) DEC_TMA(4, 4, 2, 8, 2, true) DEC_TMA(4, 4, 3, 12, 3, true)
  OS_TMA(2, 2, 5, 10, 3) OS_TMA(2, 1, 10, 10, 3) OS_PG(2, 2, 6, 12, 3) OS_PG(2, 1, 12, 12, 3)
  OS_TMA(4, 2, 4, 8, 2) OS_TMA(4, 1, 8, 8, 2) OS_PG(4, 2, 6, 12, 2) OS_PG(4, 1, 12, 12, 2)
#define SMEM_WS(NS, RPS, NST, WD) (512 + (NST) * ((RPS) * H * 2 + (RPS) * (NS) * (VB + SB)) + (NST) * (WD) * 2048)
#define OS_WS(NS, RPS, NST, WE, WD, U) { auto k = k_os_ws<NS, RPS, NST, WE, WD, U>; set_smem(k, SMEM_WS(NS, RPS, NST, WD)); \
    printf("# regs os_ws P%d RPS%d NST%d WE%d WD%d: %d (smem %d)\n", NS, RPS, NST, WE, WD, nregs((void*)k), SMEM_WS(NS, RPS, NST, WD)); \
    oss[NS].push_back({"oneshot_ws RPS" #RPS " NST" #NST " enc" #WE " dec" #WD, [=](int R, Slots, Slots in, bf16* o) { k<<<g_sms, 32 * (NST * (WE + WD) + 1), SMEM_WS(NS, RPS, NST, WD)>>>(B.x[0], in, R, o); }}); }
  OS_WS(2, 2, 5, 2, 2, 3) OS_WS(2, 2, 4, 2, 2, 3) OS_WS(2, 1, 8, 1, 1, 3)
  OS_WS(4, 2, 4, 2, 2, 2) OS_WS(4, 2, 3, 2, 4, 2) OS_WS(4, 1, 7, 1, 2, 2)
  oss[2].push_back({"oneshot_reg", [=](int R, Slots, Slots in, bf16* o) { k_os_reg<2, 3><<<g_sms * 2, 256>>>(B.x[0], in, R, o); }});
  oss[4].push_back({"oneshot_reg", [=](int R, Slots, Slots in, bf16* o) { k_os_reg<4, 2><<<g_sms * 2, 256>>>(B.x[0], in, R, o); }});
#endif

  cpu::init();
  std::vector<uint8_t> cslot[4], cslot0c;
  for (int R : rows_list) {
    for (int s = 0; s < 4; s++) {
      ref::gen_input<<<G8, 256>>>(B.x[s], s, R, 0xC0DEC + s); ref::gen_outliers<<<1, 32>>>(B.x[s], s, R, 0xC0DEC + s);
      std::vector<uint16_t> h((size_t)R * H); CK(cudaMemcpy(h.data(), B.x[s], h.size() * 2, cudaMemcpyDeviceToHost));
      make_specials(h, R, s); CK(cudaMemcpy(B.x[s], h.data(), h.size() * 2, cudaMemcpyHostToDevice));
      ref::ref_encode<<<G8, 256>>>(B.x[s], refslot[s], R);
      cpu::enc_slot(h, R, cslot[s], false);
      if (s == 0) cpu::enc_slot(h, R, cslot0c, true);
      CK(cudaMemcpy(B.tmp, cslot[s].data(), SLB(R), cudaMemcpyHostToDevice));
      { unsigned long long m = cmp_bytes(B.tmp, refslot[s], SLB(R));
        printf("ORACLE R=%-6d src %d: GPU ref_encode vs CPU oracle %llu bytes differ\n", R, s, m); }
      CK(cudaMemcpy(refslot[s], cslot[s].data(), SLB(R), cudaMemcpyHostToDevice));  // CPU oracle is the reference
    }
    CK(cudaDeviceSynchronize());
    const size_t nQ = (size_t)R * VB / 16;
    Slots S0{{B.tmp}, R};
    CK(cudaMemcpy(B.slot[3], cslot0c.data(), SLB(R), cudaMemcpyHostToDevice));
    auto chk = [&](const std::string& name, const void* a, const void* b, size_t n) {
      CK(cudaDeviceSynchronize()); unsigned long long m = cmp_bytes(a, b, n);
      printf("CHECK R=%-6d %-44s %s (%llu bytes differ)\n", R, name.c_str(), m ? "FAIL" : "bit-exact", m); allok &= m == 0;
    };
    const double inB = (double)R * H * 2, encB = (double)R * ROW_BYTES;
    const float tcp = time_med([&] { k_copy<<<g_sms * 4, 512>>>((const uint4*)B.x[0], (uint4*)B.out, (size_t)R * H * 2 / 16); });
    const float tcp2 = time_med([&] { CK(cudaMemcpyAsync(B.out, B.x[0], (size_t)R * H * 2, cudaMemcpyDeviceToDevice)); });
    const double roof = std::max(2 * inB / tcp / 1e3, 2 * inB / tcp2 / 1e3);
    printf("ROOF R=%-6d copy kernel %.1f us %.0f GB/s | memcpy %.1f us %.0f GB/s | roof %.0f GB/s\n", R, tcp, 2 * inB / tcp / 1e3, tcp2, 2 * inB / tcp2 / 1e3, roof);
    auto rep = [&](const std::string& name, float us, double bytes) {
      printf("PERF R=%-6d %-46s %8.2f us %7.0f GB/s %5.1f%% roof (floor %.1f us)\n", R, name.c_str(), us, bytes / us / 1e3, 100. * bytes / us / 1e3 / roof, bytes / roof / 1e3);
    };
    // (a) encode
    for (auto& v : encs) {
      CK(cudaMemset(B.tmp, 0, SLB(R)));
      rep(v.name, time_med([&] { v.run(R, S0, S0, nullptr); }), inB + encB);
      if (v.name.find("ref") == std::string::npos) chk(v.name, B.tmp, B.slot[3], SLB(R));
    }
#if !ENC_ONLY
    for (int P : {2, 4}) {
      char* scratch; CK(cudaMalloc(&scratch, SLB(R) * P));
      for (int s = 0; s < P; s++) { CK(cudaMemcpy(B.slot[s], refslot[s], SLB(R), cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(scratch + s * SLB(R), refslot[s], SLB(R), cudaMemcpyDeviceToDevice)); }
      Slots SS{{B.slot[0], B.slot[1], B.slot[2], B.slot[3]}, R};
      // (b) decode-sum
      std::vector<uint8_t> srcs[4] = {cslot[0], cslot[1], cslot[2], cslot[3]};
      std::vector<uint16_t> ho;
      cpu::oneshot(srcs, P, R, ho); CK(cudaMemcpy(B.ref, ho.data(), ho.size() * 2, cudaMemcpyHostToDevice));
      ref::ref_oneshot<<<G8, 256>>>(scratch, P, R, B.out);
      { CK(cudaDeviceSynchronize()); unsigned long long m = cmp_bytes(B.out, B.ref, (size_t)R * H * 2);
        printf("ORACLE R=%-6d P=%d GPU ref_oneshot vs CPU oracle %llu bytes differ\n", R, P, m); }
      const double dB = (double)P * encB + inB;
      rep("dec ref P=" + std::to_string(P) + " (harness ref_oneshot)", time_med([&] { ref::ref_oneshot<<<G8, 256>>>(scratch, P, R, B.out); }), dB);
      for (auto& v : decs[P]) {
        CK(cudaMemset(B.out, 0, (size_t)R * H * 2));
        const std::string nm = v.name + " P=" + std::to_string(P);
        rep(nm, time_med([&] { v.run(R, SS, SS, B.out); }), dB); chk(nm, B.out, B.ref, (size_t)R * H * 2);
      }
      // (c) one-shot compute: own source 0 encoded from x[0] (canonical -0); reference on canonical source 0
      k_canon<<<G8, 256>>>((uint4*)scratch, nQ);
      srcs[0] = cslot0c; cpu::oneshot(srcs, P, R, ho); CK(cudaMemcpy(B.ref, ho.data(), ho.size() * 2, cudaMemcpyHostToDevice));
      { std::vector<uint8_t> keep = srcs[0]; srcs[0] = cslot[0]; std::vector<uint16_t> hn; cpu::oneshot(srcs, P, R, hn); srcs[0] = keep;
        unsigned long long m = 0; for (size_t i = 0; i < hn.size(); i++) m += hn[i] != ho[i];
        printf("NOTE R=%-6d P=%d -0 canonicalization changes %llu oracle one-shot output elements\n", R, P, m); }
      const double oB = inB + encB + (double)(P - 1) * encB + inB;
      for (auto& v : oss[P]) {
        CK(cudaMemset(B.out, 0, (size_t)R * H * 2));
        const std::string nm = v.name + " P=" + std::to_string(P);
        rep(nm, time_med([&] { v.run(R, SS, SS, B.out); }), oB); chk(nm, B.out, B.ref, (size_t)R * H * 2);
      }
      // (c) two-shot compute: own rows [0, R/P); slot2 for the other rows prefilled by a full-ownership run
      cpu::twoshot(srcs, P, R, ho); CK(cudaMemcpy(B.ref, ho.data(), ho.size() * 2, cudaMemcpyHostToDevice));
      ref::ref_twoshot<<<G8, 256>>>(scratch, P, R, B.out);
      { CK(cudaDeviceSynchronize()); unsigned long long m = cmp_bytes(B.out, B.ref, (size_t)R * H * 2);
        printf("ORACLE R=%-6d P=%d GPU ref_twoshot (canonical src0) vs CPU oracle %llu bytes differ\n", R, P, m); }
      Slots T2{{B.slot2}, R};
      auto kts = P == 2 ? k_ts_reg<2> : k_ts_reg<4>;
      kts<<<g_sms * 2, 256>>>(B.x[0], SS, T2, R, 0, R, B.out); chk("twoshot_reg (all owned) P=" + std::to_string(P), B.out, B.ref, (size_t)R * H * 2);
      CK(cudaMemset(B.out, 0, (size_t)R * H * 2));
      const int own = R / P;
      const double tB = inB + encB + (double)own * ROW_BYTES * (P + 1) + (double)(R - own) * ROW_BYTES + inB;
      const std::string nm = "twoshot_reg (own R/P) P=" + std::to_string(P);
      rep(nm, time_med([&] { kts<<<g_sms * 2, 256>>>(B.x[0], SS, T2, R, 0, own, B.out); }), tB); chk(nm, B.out, B.ref, (size_t)R * H * 2);
      {  // TMA composition: enc_tma (all rows) + k_ts_owner (own rows) + k_ts_gather (other rows)
        auto kE = k_enc_tma<3, 6, 18, true>;
        auto kB = P == 2 ? k_ts_owner<2, 4, 3, 12> : k_ts_owner<4, 2, 3, 12>;
        const int smB = 512 + 3 * (P == 2 ? 4 : 2) * P * (VB + SB) + 12 * 2048; set_smem(kB, smB);
        auto kC = k_ts_gather<8, 4, 16>; const int smC = 512 + 4 * 8 * (VB + SB) + 16 * 2048; set_smem(kC, smC);
        auto run3 = [&](int lo, int hi) {
          kE<<<g_sms, 32 * 19, SMEM_ENC(3, 6)>>>(B.x[0], SS, R);
          kB<<<g_sms, 32 * 13, smB>>>(SS, T2, lo, hi, B.out);
          if (hi - lo < R) kC<<<g_sms, 32 * 17, smC>>>(T2, R, lo, hi, B.out);
        };
        CK(cudaMemset(B.out, 0, (size_t)R * H * 2)); CK(cudaMemset(B.slot2, 0, SLB(R)));
        run3(0, R); chk("twoshot_tma (all owned) P=" + std::to_string(P), B.out, B.ref, (size_t)R * H * 2);
        CK(cudaMemset(B.out, 0, (size_t)R * H * 2));
        const std::string nm2 = "twoshot_tma 3 kernels (own R/P) P=" + std::to_string(P);
        rep(nm2, time_med([&] { run3(0, own); }), tB); chk(nm2, B.out, B.ref, (size_t)R * H * 2);
        rep("  phase A enc_tma all rows", time_med([&] { kE<<<g_sms, 32 * 19, SMEM_ENC(3, 6)>>>(B.x[0], SS, R); }), inB + encB);
        rep("  phase B owner R/P P=" + std::to_string(P), time_med([&] { kB<<<g_sms, 32 * 13, smB>>>(SS, T2, 0, own, B.out); }), (double)own * ROW_BYTES * (P + 1) + (double)own * H * 2);
        {
          auto kB6 = P == 2 ? k_ts_owner6<2, 1, 4> : k_ts_owner6<4, 1, 4>;
          const int nw6 = 24;
          const int smB6 = 512 + 4 * P * (VB + SB) + nw6 * 2048; set_smem(kB6, smB6);
          printf("# regs ts_owner6 P%d: %d smem %d\n", P, nregs((void*)kB6), smB6);
          CK(cudaMemset(B.out, 0, (size_t)R * H * 2)); CK(cudaMemset(B.slot2, 0, SLB(R)));
          kE<<<g_sms, 32 * 19, SMEM_ENC(3, 6)>>>(B.x[0], SS, R);
          kB6<<<g_sms, 32 * (nw6 + 1), smB6>>>(SS, T2, 0, R, B.out);
          chk("ts_owner6 (all owned) P=" + std::to_string(P), B.out, B.ref, (size_t)R * H * 2);
          rep("  phase B owner R/P split-6 P=" + std::to_string(P), time_med([&] { kB6<<<g_sms, 32 * (nw6 + 1), smB6>>>(SS, T2, 0, own, B.out); }), (double)own * ROW_BYTES * (P + 1) + (double)own * H * 2);
          CK(cudaMemset(B.out, 0, (size_t)R * H * 2));
          const std::string nm3 = "twoshot_tma 3 kernels split-6 owner P=" + std::to_string(P);
          rep(nm3, time_med([&] { kE<<<g_sms, 32 * 19, SMEM_ENC(3, 6)>>>(B.x[0], SS, R); kB6<<<g_sms, 32 * (nw6 + 1), smB6>>>(SS, T2, 0, own, B.out);
                                  kC<<<g_sms, 32 * 17, smC>>>(T2, R, 0, own, B.out); }), tB);
          chk(nm3, B.out, B.ref, (size_t)R * H * 2);
        }
        rep("  phase C gather R-R/P", time_med([&] { kC<<<g_sms, 32 * 17, smC>>>(T2, R, 0, own, B.out); }), (double)(R - own) * (ROW_BYTES + H * 2));
      }
      CK(cudaFree(scratch));
    }
#endif
  }
  printf("RESULT %s\n", allok ? "ALL BIT-EXACT" : "MISMATCHES PRESENT");
  return allok ? 0 : 1;
}
