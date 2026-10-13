// fp4-allreduce: reference NVFP4 row codec v2 (namespace fp4c); the bitwise gate's ground truth.
// NVFP4 row codec v2 shared by the harness reference and every round-2 variant (spec: DESIGN-BRIEF.md).
// Encode is bit-identical to round 1 (ref/fp4_ar_r1.cu:84-107); decode-accumulate uses fmaf in rank order.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace fp4c {

// ---- wire layout per row (hidden = 6144) ----
constexpr int H = 6144;
constexpr int PAIRS = H / 32;            // 192 pairs of two 16-value blocks; one pair = 16 B values + 2 B scales
constexpr int VB = H / 2;                // 3072 value bytes
constexpr int SB = H / 16;               // 384 E4M3 scale bytes
constexpr int RSB = 4;                   // FP32 row scale
constexpr int ROW_BYTES = VB + SB + RSB; // 3460
constexpr float ROW_AMAX_TARGET = 2688.f;  // 448 (E4M3 max) * 6 (E2M1 max)

// ---- Lamport sentinels (never produced by the encoder after canonicalization) ----
constexpr uint32_t SENT_VAL32 = 0x88888888u;  // eight E2M1 -0 nibbles
constexpr uint8_t SENT_SF = 0xFFu;            // E4M3 NaN; satfinite scales are <= 0x7E
constexpr uint32_t SENT_RS = 0x80000000u;     // -0.0f; row scale is always >= +0

// Clears bit 3 of every nibble whose low 3 bits are zero (-0 -> +0); 3 int ops (LOP, IADD, LOP3).
__device__ __forceinline__ uint32_t canon_neg0(uint32_t w) {
  uint32_t s = (w & 0x77777777u) + 0x77777777u;  // bit3 of each nibble = (low3 != 0); no inter-nibble carry
  return w & (s | 0x77777777u);
}
__device__ __forceinline__ uint4 canon_neg0(uint4 q) {
  return make_uint4(canon_neg0(q.x), canon_neg0(q.y), canon_neg0(q.z), canon_neg0(q.w));
}
__device__ __forceinline__ bool is_sent_val(uint4 q) {
  return (q.x == SENT_VAL32) | (q.y == SENT_VAL32) | (q.z == SENT_VAL32) | (q.w == SENT_VAL32);
}

// ---- primitives ----
// Bytes cross the asm boundary as 16-bit values: packing .b8 regs with mov.b32 {b0,b1,b2,b3} let ptxas (13.0)
// forward an enc result into a decode with the wrong byte select (R.B1 of a single-byte pack) in ref_twoshot.
__device__ __forceinline__ uint32_t e2m1x2(float lo, float hi) {
  unsigned short r;
  asm("{\n.reg .b8 t;\ncvt.rn.satfinite.e2m1x2.f32 t, %2, %1;\nmov.b16 %0, {t, 0};\n}" : "=h"(r) : "f"(lo), "f"(hi));
  return r;
}
__device__ __forceinline__ uint32_t e2m1x8(const float* f) {
  return e2m1x2(f[0], f[1]) | (e2m1x2(f[2], f[3]) << 8) | (e2m1x2(f[4], f[5]) << 16) | (e2m1x2(f[6], f[7]) << 24);
}
__device__ __forceinline__ __half2 e2m1x2_dec(uint32_t byte) {
  unsigned short v = (unsigned short)(byte & 0xffu); uint32_t o;
  asm("{\n.reg .b8 t, z;\nmov.b16 {t, z}, %1;\ncvt.rn.f16x2.e2m1x2 %0, t;\n}" : "=r"(o) : "h"(v));
  return *(__half2*)&o;
}
__device__ __forceinline__ void e2m1x8_dec(uint32_t w, __half2* h) {
#pragma unroll
  for (int k = 0; k < 4; k++) h[k] = e2m1x2_dec(w >> (8 * k));
}
__device__ __forceinline__ float e4m3f(uint32_t b) {
  __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
  return __half2float(*(__half*)&hr);
}
__device__ __forceinline__ void unpack32(const uint4* v, float* f) {
  const __nv_bfloat162* b = (const __nv_bfloat162*)v;
#pragma unroll
  for (int i = 0; i < 16; i++) { float2 t = __bfloat1622float2(b[i]); f[2 * i] = t.x; f[2 * i + 1] = t.y; }
}

// ---- row scale ----
__device__ __forceinline__ float amax16B(uint4 t, float am) {
  const __nv_bfloat162* b = (const __nv_bfloat162*)&t;
#pragma unroll
  for (int i = 0; i < 4; i++) { float2 f = __bfloat1622float2(__habs2(b[i])); am = fmaxf(am, fmaxf(f.x, f.y)); }
  return am;
}
__device__ __forceinline__ float warp_amax(float am) {
#pragma unroll
  for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
  return am;
}
// inv = 2688/amax (0 if amax == 0), rs = amax/2688
__device__ __forceinline__ void row_scales(float am, float& inv, float& rs) {
  inv = am > 0.f ? __fdiv_rn(ROW_AMAX_TARGET, am) : 0.f;
  rs = __fdiv_rn(am, ROW_AMAX_TARGET);
}
// Row amax of one BF16 row by one warp (lane covers pairs lane + 32*j).
__device__ __forceinline__ float row_amax_warp(const __nv_bfloat16* xr, int lane) {
  float am = 0.f;
#pragma unroll
  for (int j = 0; j < 6; j++)
#pragma unroll
    for (int k = 0; k < 4; k++) am = amax16B(__ldg((const uint4*)xr + 4 * (lane + 32 * j) + k), am);
  return warp_amax(am);
}

// ---- encode: one pair (32 values) -> 16 B E2M1 values + 2 E4M3 scales (low byte = first block) ----
__device__ __forceinline__ void enc_pair_f(const float* f, float inv, uint4& q, uint16_t& sf) {
  uint32_t w[4]; uint32_t s[2];
#pragma unroll
  for (int b = 0; b < 2; b++) {
    float bm = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) bm = fmaxf(bm, fabsf(f[16 * b + i]));
    float sv = __fdiv_rn(__fmul_rn(bm, inv), 6.f);
    s[b] = (uint32_t)__nv_cvt_float_to_fp8(sv, __NV_SATFINITE, __NV_E4M3);
    float sff = e4m3f(s[b]);
    float m = sff > 0.f ? __fdiv_rn(inv, sff) : 0.f;
    float g[16];
#pragma unroll
    for (int i = 0; i < 16; i++) g[i] = __fmul_rn(f[16 * b + i], m);
    w[2 * b] = e2m1x8(g); w[2 * b + 1] = e2m1x8(g + 8);
  }
  q = make_uint4(w[0], w[1], w[2], w[3]); sf = (uint16_t)(s[0] | (s[1] << 8));
}
__device__ __forceinline__ void enc_pair(const uint4* v, float inv, uint4& q, uint16_t& sf) {
  float f[32]; unpack32(v, f); enc_pair_f(f, inv, q, sf);
}

// ---- decode-accumulate v2: acc = fmaf(e2m1(v), e4m3(s)*rs, acc); call once per source in rank order ----
__device__ __forceinline__ void dec_acc(float* acc, uint4 q, uint16_t sf, float rs) {
  const float s0 = __fmul_rn(e4m3f(sf & 0xff), rs), s1 = __fmul_rn(e4m3f(sf >> 8), rs);
  const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int k = 0; k < 4; k++) {
    __half2 h[4]; e2m1x8_dec(w[k], h); const float s = k < 2 ? s0 : s1;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 t = __half22float2(h[i]);
      acc[8 * k + 2 * i] = __fmaf_rn(t.x, s, acc[8 * k + 2 * i]);
      acc[8 * k + 2 * i + 1] = __fmaf_rn(t.y, s, acc[8 * k + 2 * i + 1]);
    }
  }
}
__device__ __forceinline__ void zero32(float* acc) {
#pragma unroll
  for (int i = 0; i < 32; i++) acc[i] = 0.f;
}
__device__ __forceinline__ float amax32(const float* acc, float am) {
#pragma unroll
  for (int i = 0; i < 32; i++) am = fmaxf(am, fabsf(acc[i]));
  return am;
}
// 32 FP32 -> 32 BF16 (64 B) at o.
__device__ __forceinline__ void store_out(__nv_bfloat16* o, const float* acc) {
  uint4 u[4]; __nv_bfloat162* b = (__nv_bfloat162*)u;
#pragma unroll
  for (int i = 0; i < 16; i++) b[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
#pragma unroll
  for (int k = 0; k < 4; k++) ((uint4*)o)[k] = u[k];
}

}  // namespace fp4c
