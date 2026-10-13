// fp4-allreduce: fused-norm epilogue math shared by lnorm.cu and the norm driver reference.
// Fused-norm epilogue math shared by the kernel (lnorm.cu) and the driver reference (drv_norm.cu). Spec: NOTES.md.
#pragma once
#include "fp4_codec.cuh"
#include <cuda_fp8.h>

namespace lne {
using namespace fp4c;
constexpr int QROW = VB + SB;  // 3456 B NVFP4 output row: 3072 E2M1 bytes then 384 E4M3 scales (linear)
constexpr int XROW = H + H / 32;  // 6336 B MXFP8 output row: 6144 E4M3 bytes then 192 UE8M0 scales (linear)
__host__ __device__ constexpr int orow_bytes(int epi) { return epi == 3 ? XROW : QROW; }

struct NormArgs {
  const __nv_bfloat16* res_in;   // [rows, H] residual input
  __nv_bfloat16* res_out;        // [rows, H] updated residual (may equal res_in)
  const __nv_bfloat16* w;        // [H] norm weight
  float eps, wbias;              // M3: eps 1e-6, Gemma weight_bias 1.0 (y = x * rinv * (wbias + w))
  int epi;                       // 0 = AllReduce only (out = BF16 sum), 1 = norm BF16 out, 2 = norm NVFP4 out, 3 = norm MXFP8 out
  uint8_t* outq;                 // epi 2: [rows] x 3456 B
  float qscale;                  // epi 2: static global scale, same meaning as vLLM scaled_fp4_quant's input_global_scale
};

__device__ __forceinline__ float rcp_approx(float x) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__device__ __forceinline__ float rsqrt_approx(float x) { float r; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
// rinv from the FP32 sum of squares (FlashInfer: rsqrt(sum_sq / d + eps)).
__device__ __forceinline__ float norm_rinv(float ss, float eps) { return rsqrt_approx(__fadd_rn(__fdiv_rn(ss, (float)H), eps)); }
__device__ __forceinline__ float bfw(uint32_t u, int hi) { return __uint_as_float(hi ? (u & 0xffff0000u) : (u << 16)); }
__device__ __forceinline__ float2 add2(float2 a, float2 b) {
  float2 d;
  asm("{.reg .b64 A, B, D;\nmov.b64 A, {%2, %3};\nmov.b64 B, {%4, %5};\nadd.rn.f32x2 D, A, B;\nmov.b64 {%0, %1}, D;}"
      : "=f"(d.x), "=f"(d.y) : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return d;
}
__device__ __forceinline__ float2 mul2(float2 a, float2 b) {
  float2 d;
  asm("{.reg .b64 A, B, D;\nmov.b64 A, {%2, %3};\nmov.b64 B, {%4, %5};\nmul.rn.f32x2 D, A, B;\nmov.b64 {%0, %1}, D;}"
      : "=f"(d.x), "=f"(d.y) : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return d;
}
__device__ __forceinline__ float2 fma2(float2 a, float2 b, float2 c) {
  float2 d;
  asm("{.reg .b64 A, B, C, D;\nmov.b64 A, {%2, %3};\nmov.b64 B, {%4, %5};\nmov.b64 C, {%6, %7};\nfma.rn.f32x2 D, A, B, C;\nmov.b64 {%0, %1}, D;}"
      : "=f"(d.x), "=f"(d.y) : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y), "f"(c.x), "f"(c.y));
  return d;
}
__device__ __forceinline__ float2 bf2(uint32_t u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }
// Packed f32x2 form of norm_pair (bit-identical: every lane op is the same IEEE rn op).
__device__ __forceinline__ void norm_pair2(const uint4* x, const uint4* w, float rinv, float wbias, float* y) {
  const float2 r2 = make_float2(rinv, rinv), b2 = make_float2(wbias, wbias);
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const uint32_t xu[4] = {x[k].x, x[k].y, x[k].z, x[k].w}, wu[4] = {w[k].x, w[k].y, w[k].z, w[k].w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float2 t = mul2(mul2(bf2(xu[i]), r2), add2(b2, bf2(wu[i])));
      y[8 * k + 2 * i] = t.x; y[8 * k + 2 * i + 1] = t.y;
    }
  }
}
// One pair (32 values): x = 32 BF16 (4 x uint4) -> y = bf16(x * rinv * (wbias + w)), w = 32 BF16 weights.
__device__ __forceinline__ void norm_pair(const uint4* x, const uint4* w, float rinv, float wbias, float* y) {
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const uint32_t xu[4] = {x[k].x, x[k].y, x[k].z, x[k].w}, wu[4] = {w[k].x, w[k].y, w[k].z, w[k].w};
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float a = bfw(xu[i >> 1], i & 1), b = bfw(wu[i >> 1], i & 1);
      y[8 * k + i] = __fmul_rn(__fmul_rn(a, rinv), __fadd_rn(wbias, b));
    }
  }
}
// vLLM scaled_fp4_quant arithmetic (cvt_warp_fp16_to_fp4, E4M3 scales) for one pair of 16-blocks of BF16-rounded y.
// q = 16 B E2M1 (element 0 in low nibble), returns two scale bytes (low = first block).
__device__ __forceinline__ uint32_t quant_pair_vllm(const float* y, float gsc, uint4& q) {
  float yb[32];
#pragma unroll
  for (int i = 0; i < 32; i++) yb[i] = __bfloat162float(__float2bfloat16_rn(y[i]));
  uint32_t w[4], sfb[2];
#pragma unroll
  for (int b = 0; b < 2; b++) {
    float vm = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) vm = fmaxf(vm, fabsf(yb[16 * b + i]));
    const float sfv = __fmul_rn(gsc, __fmul_rn(vm, rcp_approx(6.0f)));
    sfb[b] = (uint32_t)__nv_cvt_float_to_fp8(sfv, __NV_SATFINITE, __NV_E4M3);
    const float sff = e4m3f(sfb[b]);
    const float os = sff != 0.f ? rcp_approx(__fmul_rn(sff, rcp_approx(gsc))) : 0.f;
    float g[16];
#pragma unroll
    for (int i = 0; i < 16; i++) g[i] = __fmul_rn(yb[16 * b + i], os);
    w[2 * b] = e2m1x8(g); w[2 * b + 1] = e2m1x8(g + 8);
  }
  q = make_uint4(w[0], w[1], w[2], w[3]);
  return sfb[0] | (sfb[1] << 8);
}
// MXFP8 (E4M3 values, UE8M0 scale per 32) of one pair = one 32-block of bf16(y), vLLM _mxfp8_e4m3_quantize_torch semantics
// with an exact ceil(log2(amax/448)) from the bits of rn(amax/448): e = clamp(ceil + 127, 0, 254), v = e4m3_rn_satfinite(x * 2^(127-e)).
// q[0..1] = 32 value bytes (element 0 first); returns the scale byte.
__device__ __forceinline__ uint32_t quant_pair_mxfp8(const float* y, uint4* q) {
  float yb[32]; float am = 0.f;
#pragma unroll
  for (int i = 0; i < 32; i++) { yb[i] = __bfloat162float(__float2bfloat16_rn(y[i])); am = fmaxf(am, fabsf(yb[i])); }
  am = fmaxf(am, 1.17549435e-38f);
  const uint32_t qb = __float_as_uint(__fdiv_rn(am, 448.f));
  int ce = (int)((qb >> 23) & 0xff) - 127 + ((qb & 0x7fffffu) ? 1 : 0);  // normal q; subnormal q -> exponent field 0
  if (((qb >> 23) & 0xff) == 0) ce = -126;  // q < 2^-126: ceil(log2 q) <= -126 (clamped below anyway)
  int e = ce + 127; e = e < 0 ? 0 : e > 254 ? 254 : e;
  const float m = e == 254 ? __uint_as_float(0x00400000u) : __uint_as_float((uint32_t)(254 - e) << 23);  // 2^(127-e)
  uint32_t w[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const __nv_fp8x2_storage_t a = __nv_cvt_float2_to_fp8x2(make_float2(__fmul_rn(yb[4 * i], m), __fmul_rn(yb[4 * i + 1], m)), __NV_SATFINITE, __NV_E4M3);
    const __nv_fp8x2_storage_t b = __nv_cvt_float2_to_fp8x2(make_float2(__fmul_rn(yb[4 * i + 2], m), __fmul_rn(yb[4 * i + 3], m)), __NV_SATFINITE, __NV_E4M3);
    w[i] = (uint32_t)a | ((uint32_t)b << 16);
  }
  q[0] = make_uint4(w[0], w[1], w[2], w[3]); q[1] = make_uint4(w[4], w[5], w[6], w[7]);
  return (uint32_t)e;
}
}  // namespace lne
