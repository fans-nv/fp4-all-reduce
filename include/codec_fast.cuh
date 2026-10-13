// fp4-allreduce: fast NVFP4 row codec (namespace fp4f), bit-identical to fp4_codec.cuh.
// Optimized NVFP4 row codec, bit-identical to include/fp4_codec.cuh (spec v2). API doc: README.md (same dir).
// Wire layout is the linear NVFP4 layout of fp4_codec.cuh: value byte i/2 (low nibble = even element),
// scale byte i/16, one FP32 row scale per row.
#pragma once
#include "fp4_codec.cuh"
#include <cuda_fp8.h>

// FP4F_FASTDIV=1: divide-free m = inv/e4m3(s) in the warp-level row encoders (exact; see div_e4m3). Default off: the
// per-row uniform branch duplicates the unrolled encode body and measured slower on GB200.
#ifndef FP4F_FASTDIV
#define FP4F_FASTDIV 0
#endif
namespace fp4f {
using namespace fp4c;

// ======================= scalar / packed primitives =======================
__device__ __forceinline__ float2 mul2(float2 a, float2 b) {
  float2 d;
  asm("{.reg .b64 A, B, D;\nmov.b64 A, {%2, %3};\nmov.b64 B, {%4, %5};\nmul.rn.f32x2 D, A, B;\nmov.b64 {%0, %1}, D;}"
      : "=f"(d.x), "=f"(d.y) : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return d;
}
__device__ __forceinline__ float2 fma2(float2 a, float2 b, float2 c) {
  float2 d;
  asm("{.reg .b64 A, B, C, D;\nmov.b64 A, {%2, %3};\nmov.b64 B, {%4, %5};\nmov.b64 C, {%6, %7};\n"
      "fma.rn.f32x2 D, A, B, C;\nmov.b64 {%0, %1}, D;}"
      : "=f"(d.x), "=f"(d.y) : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y), "f"(c.x), "f"(c.y));
  return d;
}
// rn(x / 6) for x >= 0 (incl. subnormal, inf, NaN): Markstein correction from rn(1/6); verified exhaustively.
__device__ __forceinline__ float div6_rn(float x) {
  const float r6 = 0x1.555556p-3f;
  const float q0 = __fmul_rn(x, r6);
  const float q1 = __fmaf_rn(__fmaf_rn(-q0, 6.f, x), r6, q0);
  return x == __int_as_float(0x7f800000) ? x : q1;
}
// Two E4M3 bytes (lo = first) from two non-negative floats, satfinite rn.
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
// Two E4M3 bytes -> two floats (exact).
__device__ __forceinline__ float2 e4m3x2_f(uint32_t s2) {
  uint32_t h;
  asm("{.reg .b16 t;\ncvt.u16.u32 t, %1;\ncvt.rn.f16x2.e4m3x2 %0, t;}" : "=r"(h) : "r"(s2));
  return __half22float2(*(__half2*)&h);
}
// Four E2M1x2 bytes of one word -> 8 floats.
__device__ __forceinline__ void e2m1x8_f(uint32_t w, float2* f) {
  __half2 h[4]; e2m1x8_dec(w, h);
#pragma unroll
  for (int i = 0; i < 4; i++) f[i] = __half22float2(h[i]);
}
// 8 floats -> E2M1 word (element 0 in low nibble). Bytes leave the asm as 16-bit values: packing .b8 registers with
// mov.b32 {b0,b1,b2,b3} is miscompiled by ptxas 13.0 on sm_100a when an encode feeds a decode in registers.
__device__ __forceinline__ uint32_t e2m1x2_2(float2 g) {
  unsigned short r;
  asm("{\n.reg .b8 t;\ncvt.rn.satfinite.e2m1x2.f32 t, %2, %1;\nmov.b16 %0, {t, 0};\n}" : "=h"(r) : "f"(g.x), "f"(g.y));
  return r;
}
// FP4F_PACK2=1: two cvts packed into one 16-bit value (mov.b16 {t0, t1}); halves the byte-combine work.
#ifndef FP4F_PACK2
#define FP4F_PACK2 0
#endif
__device__ __forceinline__ uint32_t e2m1x4_2(float2 g0, float2 g1) {
  unsigned short r;
  asm("{\n.reg .b8 t0, t1;\ncvt.rn.satfinite.e2m1x2.f32 t0, %2, %1;\ncvt.rn.satfinite.e2m1x2.f32 t1, %4, %3;\n"
      "mov.b16 %0, {t0, t1};\n}" : "=h"(r) : "f"(g0.x), "f"(g0.y), "f"(g1.x), "f"(g1.y));
  return r;
}
__device__ __forceinline__ uint32_t e2m1x8_2(const float2* g) {
  if (FP4F_PACK2) return e2m1x4_2(g[0], g[1]) | (e2m1x4_2(g[2], g[3]) << 16);
  return e2m1x2_2(g[0]) | (e2m1x2_2(g[1]) << 8) | (e2m1x2_2(g[2]) << 16) | (e2m1x2_2(g[3]) << 24);
}
// BF16 pair word -> float2 (exact; 1 op each).
__device__ __forceinline__ float2 bf2f(uint32_t u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }

// max |x| over 8 BF16 (packed; exact), returned as BF16x2 word (max still split across halves).
__device__ __forceinline__ __nv_bfloat162 habsmax8(uint4 v, __nv_bfloat162 m) {
  const __nv_bfloat162* b = (const __nv_bfloat162*)&v;
#pragma unroll
  for (int i = 0; i < 4; i++) m = __hmax2(m, __habs2(b[i]));
  return m;
}
__device__ __forceinline__ float hmax_halves(__nv_bfloat162 m) {
  const uint32_t u = *(uint32_t*)&m;
  return fmaxf(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u));
}
__device__ __forceinline__ __nv_bfloat162 hzero2() { return __floats2bfloat162_rn(0.f, 0.f); }

// ======================= block / pair encode =======================
// Exact rn(inv / sff) without a divide, for warp-converged callers: rc = rn(1/sff) from an 8-entry reciprocal-mantissa table
// held one entry per lane (rtab = rcp_tab(lane)) + exponent arithmetic, then one Markstein correction. Valid (verified
// exhaustively against __fdiv_rn) for inv in [2^-100, 2^100] and every E4M3 sff > 0; the caller guards the range per row.
__device__ __forceinline__ uint32_t rcp_tab(int lane) { return __float_as_uint(__frcp_rn(1.f + 0.125f * (float)(lane & 7))); }
__device__ __forceinline__ bool fast_div_ok(float inv) { return inv >= 0x1p-100f && inv <= 0x1p100f; }
__device__ __forceinline__ float div_e4m3(float inv, float sff, uint32_t rtab) {
  const uint32_t sb = __float_as_uint(sff);
  const uint32_t t = __shfl_sync(0xffffffffu, rtab, (sb >> 20) & 7);
  const float rc = __uint_as_float(t + 0x3F800000u - (sb & 0x7F800000u));
  const float q0 = __fmul_rn(inv, rc);
  return __fmaf_rn(__fmaf_rn(-q0, sff, inv), rc, q0);
}
// Scales of two 16-blocks: sf2 = two E4M3 bytes, m = inv / e4m3(s) (0 if s decodes to 0 or NaN). Bit-identical to fp4c.
// fast (warp-uniform) selects div_e4m3 (all 32 lanes must call); otherwise IEEE __fdiv_rn.
__device__ __forceinline__ void pair_scales(float bm0, float bm1, float inv, uint32_t& sf2, float& m0, float& m1,
                                            bool fast = false, uint32_t rtab = 0) {
  sf2 = e4m3x2(div6_rn(__fmul_rn(bm0, inv)), div6_rn(__fmul_rn(bm1, inv)));
  const float2 s = e4m3x2_f(sf2);
  float d0, d1;
  if (fast) { d0 = div_e4m3(inv, s.x, rtab); d1 = div_e4m3(inv, s.y, rtab); }
  else { d0 = __fdiv_rn(inv, s.x); d1 = __fdiv_rn(inv, s.y); }
  m0 = s.x > 0.f ? d0 : 0.f;
  m1 = s.y > 0.f ? d1 : 0.f;
}
// 8 BF16 (one 16 B chunk) * m -> E2M1 word.
template <bool CANON>
__device__ __forceinline__ uint32_t enc_chunk_bf16(uint4 v, float m) {
  const float2 mm = make_float2(m, m);
  float2 g[4];
  g[0] = mul2(bf2f(v.x), mm); g[1] = mul2(bf2f(v.y), mm); g[2] = mul2(bf2f(v.z), mm); g[3] = mul2(bf2f(v.w), mm);
  const uint32_t w = e2m1x8_2(g);
  return CANON ? canon_neg0(w) : w;
}
// Encode one pair (32 BF16 = v[0..3]) given block maxima; q = 16 value bytes, returns sf2.
template <bool CANON>
__device__ __forceinline__ uint32_t enc_pair_bm(const uint4* v, float bm0, float bm1, float inv, uint4& q,
                                                bool fast = false, uint32_t rtab = 0) {
  uint32_t sf2; float m0, m1; pair_scales(bm0, bm1, inv, sf2, m0, m1, fast, rtab);
  q = make_uint4(enc_chunk_bf16<CANON>(v[0], m0), enc_chunk_bf16<CANON>(v[1], m0),
                 enc_chunk_bf16<CANON>(v[2], m1), enc_chunk_bf16<CANON>(v[3], m1));
  return sf2;
}
// Drop-in for fp4c::enc_pair (computes block maxima itself).
template <bool CANON>
__device__ __forceinline__ uint32_t enc_pair_fast(const uint4* v, float inv, uint4& q) {
  const float bm0 = hmax_halves(habsmax8(v[1], habsmax8(v[0], hzero2())));
  const float bm1 = hmax_halves(habsmax8(v[3], habsmax8(v[2], hzero2())));
  return enc_pair_bm<CANON>(v, bm0, bm1, inv, q);
}
// FP32 input variant (two-shot owner re-encode): drop-in for fp4c::enc_pair_f. f = 32 floats.
template <bool CANON>
__device__ __forceinline__ uint32_t enc_pair_f_fast(const float* f, float inv, uint4& q) {
  float bm0 = 0.f, bm1 = 0.f;
#pragma unroll
  for (int i = 0; i < 16; i++) { bm0 = fmaxf(bm0, fabsf(f[i])); bm1 = fmaxf(bm1, fabsf(f[16 + i])); }
  uint32_t sf2; float m0, m1; pair_scales(bm0, bm1, inv, sf2, m0, m1);
  uint32_t w[4];
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const float2 mm = k < 2 ? make_float2(m0, m0) : make_float2(m1, m1);
    float2 g[4];
#pragma unroll
    for (int i = 0; i < 4; i++) g[i] = mul2(make_float2(f[8 * k + 2 * i], f[8 * k + 2 * i + 1]), mm);
    w[k] = e2m1x8_2(g);
    if (CANON) w[k] = canon_neg0(w[k]);
  }
  q = make_uint4(w[0], w[1], w[2], w[3]);
  return sf2;
}

// ======================= decode-accumulate =======================
// acc[0..31] = fmaf(e2m1(q), e4m3(sf)*rs, acc) for one pair; bit-identical to fp4c::dec_acc.
__device__ __forceinline__ void dec_acc_fast(float* acc, uint4 q, uint32_t sf2, float rs) {
  const float2 e = e4m3x2_f(sf2);
  const float2 S0 = make_float2(__fmul_rn(e.x, rs), __fmul_rn(e.x, rs));
  const float2 S1 = make_float2(__fmul_rn(e.y, rs), __fmul_rn(e.y, rs));
  const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int k = 0; k < 4; k++) {
    float2 t[4]; e2m1x8_f(w[k], t);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 a = make_float2(acc[8 * k + 2 * i], acc[8 * k + 2 * i + 1]);
      a = fma2(t[i], k < 2 ? S0 : S1, a);
      acc[8 * k + 2 * i] = a.x; acc[8 * k + 2 * i + 1] = a.y;
    }
  }
}
// 32 FP32 -> 4 x 16 B BF16 words.
__device__ __forceinline__ void pack_bf16x32(const float* acc, uint4* u) {
  __nv_bfloat162* b = (__nv_bfloat162*)u;
#pragma unroll
  for (int i = 0; i < 16; i++) b[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
}

// ======================= memory helpers =======================
__device__ __forceinline__ uint4 ld_nc_na(const void* p) {  // streaming read-only, no L1 allocate
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 ld_cg(const void* p) { return __ldcg((const uint4*)p); }
__device__ __forceinline__ void st_na(void* p, uint4 v) {  // streaming store
  asm volatile("st.global.L1::no_allocate.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}"
               :: "r"(smem_u32(b)), "r"(parity) : "memory");
}
// 1-D TMA bulk copy global -> smem, completes on mbarrier (bytes % 16 == 0, 16 B aligned).
__device__ __forceinline__ void tma_g2s(void* s, const void* g, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(s)), "l"(g), "r"(bytes), "r"(smem_u32(b)) : "memory");
}
// 1-D TMA bulk store smem -> global (bulk_group). Call fence_async_smem() in every writing thread + sync first.
__device__ __forceinline__ void tma_s2g(void* g, const void* s, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(g), "r"(smem_u32(s)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait() { asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// ======================= warp-level row functions =======================
// Output descriptor for one encoded row: q = 3072 value bytes (192 x uint4), sf = 384 scale bytes, rs = row scale.
// Any of them may point to global, peer (NVLink) or shared memory; rs may be nullptr (row scale is also returned).
struct EncRow { uint4* q; uint16_t* sf; float* rs; };

// (E-REG) Single-pass warp encode of one BF16 row from global, whole row in registers (~24 x uint4 / lane).
// Lane handles pairs lane + 32 j. Needs ~130 regs/thread: use where occupancy is not the binding limit.
// LDM: 0 = ld.global.nc.L1::no_allocate, 1 = ld.global.nc (L1-allocating; merges the 64 B lane stride in L1).
template <bool CANON, int LDM = 1>
__device__ __forceinline__ float enc_row_reg(const __nv_bfloat16* xr, int lane, EncRow o) {
  uint4 v[6][4];
#pragma unroll
  for (int j = 0; j < 6; j++)
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const uint4* p = (const uint4*)xr + 4 * (lane + 32 * j) + k;
      v[j][k] = LDM == 0 ? ld_nc_na(p) : __ldg(p);
    }
  float bm[6][2]; float am = 0.f;
#pragma unroll
  for (int j = 0; j < 6; j++) {
    bm[j][0] = hmax_halves(habsmax8(v[j][1], habsmax8(v[j][0], hzero2())));
    bm[j][1] = hmax_halves(habsmax8(v[j][3], habsmax8(v[j][2], hzero2())));
    am = fmaxf(am, fmaxf(bm[j][0], bm[j][1]));
  }
  float inv, rs; row_scales(warp_amax(am), inv, rs);
  const bool fast = FP4F_FASTDIV && fast_div_ok(inv); const uint32_t rtab = FP4F_FASTDIV ? rcp_tab(lane) : 0u;
#pragma unroll
  for (int j = 0; j < 6; j++) {
    uint4 q; const uint32_t sf2 = enc_pair_bm<CANON>(v[j], bm[j][0], bm[j][1], inv, q, fast, rtab);
    o.q[lane + 32 * j] = q; o.sf[lane + 32 * j] = (uint16_t)sf2;
  }
  if (lane == 0 && o.rs) *o.rs = rs;
  return rs;
}

__device__ __forceinline__ uint4 lds128(const void* p) {
  uint4 r;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(smem_u32(p)));
  return r;
}
// (E-SMEM) Warp encode of one BF16 row resident in shared memory (12288 B, 16 B aligned). Pass 1 reads smem for the row
// amax only; pass 2 re-reads each pair (volatile, so the row is never held in registers), recomputes its block maxima
// and encodes: ~64 regs, one HBM read upstream (TMA). Chunk order is XOR-swizzled per lane ((lane>>1)&3) so every
// LDS.128 quarter-warp hits 8 distinct 16 B bank groups. Must be called by all 32 lanes (shuffles).
// LOWREG=true (default) forces the pass-2 re-read (volatile LDS) so the row never lives in registers: ~84 regs and
// measured 1.6-2x faster than letting the compiler cache the row (GB200, 16K rows, 18 encoder warps).
template <bool CANON, bool LOWREG = true>
__device__ __forceinline__ float enc_row_smem(const __nv_bfloat16* sx, int lane, EncRow o) {
  const uint4* s = (const uint4*)sx;
  const int f = (lane >> 1) & 3;
  uint32_t bmp[6];  // per pair: block maxima as BF16 (low half = block 0), kept from pass 1
  float am = 0.f;
#pragma unroll
  for (int j = 0; j < 6; j++) {
    const uint4* p = s + 4 * (lane + 32 * j);
    __nv_bfloat162 a = hzero2(), b = hzero2();
    a = habsmax8(LOWREG ? lds128(p + (0 ^ f)) : p[0 ^ f], a); a = habsmax8(LOWREG ? lds128(p + (1 ^ f)) : p[1 ^ f], a);
    b = habsmax8(LOWREG ? lds128(p + (2 ^ f)) : p[2 ^ f], b); b = habsmax8(LOWREG ? lds128(p + (3 ^ f)) : p[3 ^ f], b);
    const uint32_t ua = *(uint32_t*)&a, ub = *(uint32_t*)&b;  // chunks {f, f^1} -> block f>>1
    const uint32_t xa = max(ua & 0xffffu, ua >> 16), xb = max(ub & 0xffffu, ub >> 16);  // non-negative BF16: integer order
    bmp[j] = (f & 2) ? (xb | (xa << 16)) : (xa | (xb << 16));
    am = fmaxf(am, __uint_as_float(max(xa, xb) << 16));
  }
  float inv, rs; row_scales(warp_amax(am), inv, rs);
  const bool fast = FP4F_FASTDIV && fast_div_ok(inv); const uint32_t rtab = FP4F_FASTDIV ? rcp_tab(lane) : 0u;
#pragma unroll
  for (int j = 0; j < 6; j++) {
    const uint4* p = s + 4 * (lane + 32 * j);
    const float bm0 = __uint_as_float(bmp[j] << 16), bm1 = __uint_as_float(bmp[j] & 0xffff0000u);
    uint32_t sf2; float m0, m1; pair_scales(bm0, bm1, inv, sf2, m0, m1, fast, rtab);
    uint32_t w[4];
#pragma unroll
    for (int k = 0; k < 4; k++) w[k] = enc_chunk_bf16<CANON>(LOWREG ? lds128(p + (k ^ f)) : p[k ^ f], ((k ^ f) & 2) ? m1 : m0);
    // un-swizzle: out[c] = w[c ^ f]
    uint32_t t0 = (f & 1) ? w[1] : w[0], t1 = (f & 1) ? w[0] : w[1], t2 = (f & 1) ? w[3] : w[2], t3 = (f & 1) ? w[2] : w[3];
    const uint4 q = (f & 2) ? make_uint4(t2, t3, t0, t1) : make_uint4(t0, t1, t2, t3);
    o.q[lane + 32 * j] = q; o.sf[lane + 32 * j] = (uint16_t)sf2;
  }
  if (lane == 0 && o.rs) *o.rs = rs;
  return rs;
}

// Source descriptor for decode: pointers to one encoded row (any memory space).
struct SrcRow { const uint4* q; const uint16_t* sf; float rs; };

// Stores one warp's 32 x 64 B BF16 block (lane l holds chunks 4l..4l+3 in o4) to out (2 KB) with lane-contiguous
// 16 B stores, transposing through a per-warp 2 KB smem buffer (XOR-swizzled, conflict-free both ways).
__device__ __forceinline__ void store_block_staged(uint4* st, int lane, const uint4* o4, uint4* out) {
  const int f = (lane >> 1) & 3;
#pragma unroll
  for (int k = 0; k < 4; k++) st[4 * lane + (k ^ f)] = o4[k];
  __syncwarp();
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const int c = lane + 32 * k, l = c >> 2;
    out[c] = st[4 * l + ((c & 3) ^ ((l >> 1) & 3))];
  }
  __syncwarp();
}

// (D) Warp decode-sum of NS encoded rows (rank order 0..NS-1) -> BF16 row (12288 B) at out.
// U = pairs per lane loaded ahead (bytes in flight per warp = 576 * NS * U). LD: 0 = ld.global.cg, 1 = generic (smem).
// st: nullptr = direct 64 B-per-lane stores; else per-warp 2 KB smem staging buffer (16 B aligned) for coalesced stores.
template <int NS, int U, int LD>
__device__ __forceinline__ void dec_row(const SrcRow* src, int lane, __nv_bfloat16* out, uint4* st = nullptr) {
  static_assert(6 % U == 0, "U must divide 6");
#pragma unroll 1
  for (int j0 = 0; j0 < 6; j0 += U) {
    uint4 q[U][NS]; uint32_t sf[U][NS];
#pragma unroll
    for (int u = 0; u < U; u++)
#pragma unroll
      for (int s = 0; s < NS; s++) {
        const int pp = lane + 32 * (j0 + u);
        if (LD == 0) { q[u][s] = __ldcg(src[s].q + pp); sf[u][s] = __ldcg(src[s].sf + pp); }
        else { q[u][s] = src[s].q[pp]; sf[u][s] = src[s].sf[pp]; }
      }
#pragma unroll
    for (int u = 0; u < U; u++) {
      float acc[32]; zero32(acc);
#pragma unroll
      for (int s = 0; s < NS; s++) dec_acc_fast(acc, q[u][s], sf[u][s], src[s].rs);
      uint4 o4[4]; pack_bf16x32(acc, o4);
      if (st) store_block_staged(st, lane, o4, (uint4*)out + 128 * (j0 + u));
      else {
        uint4* op = (uint4*)out + 4 * (lane + 32 * (j0 + u));
#pragma unroll
        for (int k = 0; k < 4; k++) op[k] = o4[k];
      }
    }
  }
}

// (TS) Two-shot owner: warp decode-sum of NS sources (rank order), FP32 row amax, re-encode of the FP32 sum into o
// (CANON for Lamport), and BF16 out = decode of that re-encoding (bit-identical to every non-owner's decode).
// Sources are decoded twice (amax pass, encode pass); keep them in smem (LD=1) or L2-hot global (LD=0). Returns rs2.
template <int NS, int LD, bool CANON>
__device__ __forceinline__ float red_reenc_row(const SrcRow* src, int lane, EncRow o, __nv_bfloat16* out, uint4* st) {
  float am = 0.f;
#pragma unroll 1
  for (int j = 0; j < 6; j++) {
    const int pp = lane + 32 * j; float acc[32]; zero32(acc);
#pragma unroll
    for (int s = 0; s < NS; s++)
      dec_acc_fast(acc, LD == 0 ? __ldcg(src[s].q + pp) : src[s].q[pp], LD == 0 ? __ldcg(src[s].sf + pp) : src[s].sf[pp], src[s].rs);
    am = amax32(acc, am);
  }
  float inv, rs2; row_scales(warp_amax(am), inv, rs2);
#pragma unroll 1
  for (int j = 0; j < 6; j++) {
    const int pp = lane + 32 * j; float acc[32]; zero32(acc);
#pragma unroll
    for (int s = 0; s < NS; s++)
      dec_acc_fast(acc, LD == 0 ? __ldcg(src[s].q + pp) : src[s].q[pp], LD == 0 ? __ldcg(src[s].sf + pp) : src[s].sf[pp], src[s].rs);
    uint4 q; const uint32_t sf = enc_pair_f_fast<CANON>(acc, inv, q);
    o.q[pp] = q; o.sf[pp] = (uint16_t)sf;
    zero32(acc); dec_acc_fast(acc, q, sf, rs2);
    uint4 o4[4]; pack_bf16x32(acc, o4);
    store_block_staged(st, lane, o4, (uint4*)out + 128 * j);
  }
  if (lane == 0 && o.rs) *o.rs = rs2;
  return rs2;
}

// (TS-split) Two-shot owner, single pass: 6 warps share one row, warp j = pair block j (lane l -> pair l + 32 j).
// Decode-sum held in registers, row amax combined through red[6] (smem floats) and named barrier bar_id (192 threads),
// then re-encode, store, decode of the re-encoding (staged via st). Same results as red_reenc_row. Returns rs2.
template <int NS, int LD, bool CANON>
__device__ __forceinline__ float red_reenc_row_split(const SrcRow* src, int lane, int j, float* red, int bar_id, EncRow o,
                                                     __nv_bfloat16* out, uint4* st) {
  const int pp = lane + 32 * j; float acc[32]; zero32(acc);
#pragma unroll
  for (int s = 0; s < NS; s++)
    dec_acc_fast(acc, LD == 0 ? __ldcg(src[s].q + pp) : src[s].q[pp], LD == 0 ? __ldcg(src[s].sf + pp) : src[s].sf[pp], src[s].rs);
  const float wam = warp_amax(amax32(acc, 0.f));
  if (lane == 0) red[j] = wam;
  asm volatile("bar.sync %0, 192;" :: "r"(bar_id) : "memory");
  const float am = fmaxf(fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3])), fmaxf(red[4], red[5]));
  asm volatile("bar.sync %0, 192;" :: "r"(bar_id) : "memory");  // red[] reusable by the next row
  float inv, rs2; row_scales(am, inv, rs2);
  uint4 q; const uint32_t sf = enc_pair_f_fast<CANON>(acc, inv, q);
  o.q[pp] = q; o.sf[pp] = (uint16_t)sf;
  zero32(acc); dec_acc_fast(acc, q, sf, rs2);
  uint4 o4[4]; pack_bf16x32(acc, o4);
  store_block_staged(st, lane, o4, (uint4*)out + 128 * j);
  if (j == 0 && lane == 0 && o.rs) *o.rs = rs2;
  return rs2;
}

}  // namespace fp4f
