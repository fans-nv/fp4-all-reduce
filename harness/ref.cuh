// Harness-only kernels: deterministic inputs, same-codec v2 references, error/compare, skew spin, empty kernel.
#pragma once
#include "fp4_codec.cuh"

namespace ref {
using namespace fp4c;

__device__ __forceinline__ uint32_t hsh(uint32_t a) {
  a ^= a >> 16; a *= 0x7feb352dU; a ^= a >> 15; a *= 0x846ca68bU; a ^= a >> 16; return a;
}
__constant__ float MAGS[8] = {1.f, .5f, 2.f, .25f, 1.5f, .75f, 3.f, .125f};
// Round-1 input: per-source magnitude-scaled Gaussian + 32 outliers of +-40*mag.
__global__ void gen_input(__nv_bfloat16* x, int src, int R, uint32_t seed) {
  const size_t N = (size_t)R * H;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < N; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h1 = hsh(seed ^ hsh((uint32_t)i * 2 + 1 + (uint32_t)(i >> 31))), h2 = hsh(h1 ^ 0x9e3779b9U ^ src);
    float u1 = (h1 >> 8) * (1.f / 16777216.f) + 1e-7f, u2 = (h2 >> 8) * (1.f / 16777216.f);
    x[i] = __float2bfloat16(sqrtf(-2.f * logf(u1)) * cospif(2.f * u2) * MAGS[src & 7]);
  }
}
__global__ void gen_outliers(__nv_bfloat16* x, int src, int R, uint32_t seed) {
  const size_t N = (size_t)R * H;
  if (threadIdx.x < 32) {
    uint32_t h = hsh(seed * 7919u + 104729u * (threadIdx.x + 1));
    size_t pos = ((size_t)h * 2654435761u) % N;
    x[pos] = __float2bfloat16(((hsh(h) & 1) ? 40.f : -40.f) * MAGS[src & 7]);
  }
}

// Reference slot layout: [R][VB] values, [R][SB] scales, [R] row scales.
struct Slot {
  char* b; int R;
  __device__ uint4* q(int row) const { return (uint4*)(b + (size_t)row * VB); }
  __device__ uint16_t* sf(int row) const { return (uint16_t*)(b + (size_t)R * VB + (size_t)row * SB); }
  __device__ float* rs() const { return (float*)(b + (size_t)R * (VB + SB)); }
};
__host__ __device__ inline size_t slot_bytes(int R) { return (size_t)R * ROW_BYTES; }

__global__ void ref_encode(const __nv_bfloat16* x, char* slot, int R) {
  const int lane = threadIdx.x & 31; Slot s{slot, R};
  for (int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); row < R; row += gridDim.x * (blockDim.x >> 5)) {
    const __nv_bfloat16* xr = x + (size_t)row * H;
    float inv, rs; row_scales(row_amax_warp(xr, lane), inv, rs);
    for (int j = 0; j < 6; j++) {
      const int pp = lane + 32 * j; uint4 v[4];
      for (int k = 0; k < 4; k++) v[k] = ((const uint4*)xr)[4 * pp + k];
      uint4 q; uint16_t sf; enc_pair(v, inv, q, sf);
      s.q(row)[pp] = q; s.sf(row)[pp] = sf;
    }
    if (lane == 0) s.rs()[row] = rs;
  }
}
// One-shot: out = sum over sources in rank order of decode(source).
__global__ void ref_oneshot(char* scratch, int n, int R, __nv_bfloat16* out) {
  const int lane = threadIdx.x & 31;
  for (int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); row < R; row += gridDim.x * (blockDim.x >> 5))
    for (int j = 0; j < 6; j++) {
      const int pp = lane + 32 * j; float acc[32]; zero32(acc);
      for (int s = 0; s < n; s++) {
        Slot sl{scratch + s * slot_bytes(R), R};
        dec_acc(acc, sl.q(row)[pp], sl.sf(row)[pp], sl.rs()[row]);
      }
      store_out(out + (size_t)row * H + 32 * pp, acc);
    }
}
// Two-shot: FP32 rank-order sum, re-encoded with its own row amax, output = decode of that re-encoding.
__device__ __forceinline__ void ref_sum_pair(char* scratch, int n, int R, int row, int pp, float* acc) {
  zero32(acc);
  for (int s = 0; s < n; s++) {
    Slot sl{scratch + s * slot_bytes(R), R};
    dec_acc(acc, sl.q(row)[pp], sl.sf(row)[pp], sl.rs()[row]);
  }
}
__global__ void ref_twoshot(char* scratch, int n, int R, __nv_bfloat16* out) {
  const int lane = threadIdx.x & 31;
  for (int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); row < R; row += gridDim.x * (blockDim.x >> 5)) {
    float am = 0.f, acc[32];
    for (int j = 0; j < 6; j++) { ref_sum_pair(scratch, n, R, row, lane + 32 * j, acc); am = amax32(acc, am); }
    float inv, rs2; row_scales(warp_amax(am), inv, rs2);
    for (int j = 0; j < 6; j++) {
      const int pp = lane + 32 * j;
      ref_sum_pair(scratch, n, R, row, pp, acc);
      uint4 q; uint16_t sf; enc_pair_f(acc, inv, q, sf);
      zero32(acc); dec_acc(acc, q, sf, rs2);
      store_out(out + (size_t)row * H + 32 * pp, acc);
    }
  }
}
// Exact FP32 sum in rank order (x[s] at stride N).
__global__ void ref_fp32(const __nv_bfloat16* x, int n, size_t N, float* out) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < N; i += (size_t)gridDim.x * blockDim.x) {
    float a = 0.f;
    for (int s = 0; s < n; s++) a = __fadd_rn(a, __bfloat162float(x[s * N + i]));
    out[i] = a;
  }
}
struct ErrAcc { double dsq, rsq, bsq; unsigned maxabs; unsigned long long mism; };
__global__ void err_kernel(const __nv_bfloat16* out, const float* r32, const __nv_bfloat16* rc, size_t N, ErrAcc* e) {
  double dsq = 0, rsq = 0, bsq = 0; float ma = 0; unsigned long long mm = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < N; i += (size_t)gridDim.x * blockDim.x) {
    float o = __bfloat162float(out[i]), r = r32[i];
    double d = (double)o - r; dsq += d * d; rsq += (double)r * r; ma = fmaxf(ma, fabsf(o - r));
    double d2 = (double)__bfloat162float(rc[i]) - r; bsq += d2 * d2;
    mm += (*(const unsigned short*)&out[i] != *(const unsigned short*)&rc[i]);
  }
  for (int o = 16; o; o >>= 1) {
    dsq += __shfl_xor_sync(~0u, dsq, o); rsq += __shfl_xor_sync(~0u, rsq, o); bsq += __shfl_xor_sync(~0u, bsq, o);
    ma = fmaxf(ma, __shfl_xor_sync(~0u, ma, o)); mm += __shfl_xor_sync(~0u, mm, o);
  }
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&e->dsq, dsq); atomicAdd(&e->rsq, rsq); atomicAdd(&e->bsq, bsq);
    atomicMax(&e->maxabs, __float_as_uint(ma)); atomicAdd(&e->mism, mm);
  }
}
// Bitwise mismatch count (16 B granules), accumulated across calls; ctr[1] counts calls with any mismatch.
__global__ void cmp_kernel(const uint4* out, const uint4* rc, size_t n16, unsigned long long* ctr) {
  unsigned long long mm = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    uint4 a = out[i], b = rc[i];
    mm += (a.x != b.x) | (a.y != b.y) | (a.z != b.z) | (a.w != b.w);
  }
  for (int o = 16; o; o >>= 1) mm += __shfl_xor_sync(~0u, mm, o);
  if ((threadIdx.x & 31) == 0 && mm) atomicAdd(&ctr[0], mm);
}
__global__ void mark_call(unsigned long long* ctr) {
  // runs after cmp_kernel: count a bad call when ctr[0] grew since the last mark
  if (ctr[0] != ctr[2]) { ctr[1]++; ctr[2] = ctr[0]; }
}
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__global__ void spin_kernel(unsigned long long ns) {
  const unsigned long long t0 = gtimer();
  while (gtimer() - t0 < ns) {}
}
__global__ void empty_kernel() {}
// ---- pre-quantized input (INPUT-FP4.md): per-tensor static scale gs = amax/2688, records in planar slot layout ----
__global__ void tensor_amax(const __nv_bfloat16* x, size_t N, float* am) {
  float m = 0.f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < N; i += (size_t)gridDim.x * blockDim.x) m = fmaxf(m, fabsf(__bfloat162float(x[i])));
  m = warp_amax(m);
  if ((threadIdx.x & 31) == 0) atomicMax((unsigned*)am, __float_as_uint(m));  // non-negative floats order as uints
}
// Q input rows (INPUT-FP4.md rev. 02:00): row r at base + r*QROW = 3072 E2M1 bytes then 384 E4M3 scales; gs per source rank.
constexpr int QROW = VB + SB;  // 3456
// Encodes with inv = 1/gs (spec-v2 block encoder, per-tensor scale), -0 canonicalized.
__global__ void q_encode(const __nv_bfloat16* x, char* recs, int R, float gs) {
  const int lane = threadIdx.x & 31;
  const float inv = __fdiv_rn(1.f, gs);
  for (int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); row < R; row += gridDim.x * (blockDim.x >> 5)) {
    const __nv_bfloat16* xr = x + (size_t)row * H; char* rr = recs + (size_t)row * QROW;
    for (int j = 0; j < 6; j++) {
      const int pp = lane + 32 * j; uint4 v[4];
      for (int k = 0; k < 4; k++) v[k] = ((const uint4*)xr)[4 * pp + k];
      uint4 q; uint16_t sf; enc_pair(v, inv, q, sf);
      ((uint4*)rr)[pp] = canon_neg0(q); ((uint16_t*)(rr + VB))[pp] = sf;
    }
  }
}
struct QSrc { const char* recs[8]; float gs[8]; };
__device__ __forceinline__ void q_sum_pair(const QSrc& Q, int n, int row, int pp, float* acc) {
  zero32(acc);
  for (int s = 0; s < n; s++) {
    const char* rr = Q.recs[s] + (size_t)row * QROW;
    dec_acc(acc, ((const uint4*)rr)[pp], ((const uint16_t*)(rr + VB))[pp], Q.gs[s]);
  }
}
__global__ void ref_q_oneshot(QSrc Q, int n, int R, __nv_bfloat16* out) {
  const int lane = threadIdx.x & 31;
  for (int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); row < R; row += gridDim.x * (blockDim.x >> 5))
    for (int j = 0; j < 6; j++) {
      const int pp = lane + 32 * j; float acc[32];
      q_sum_pair(Q, n, row, pp, acc);
      store_out(out + (size_t)row * H + 32 * pp, acc);
    }
}
// Two-shot: owner re-encodes the FP32 sum with its row amax (unchanged), output = decode of that re-encoding.
__global__ void ref_q_twoshot(QSrc Q, int n, int R, __nv_bfloat16* out) {
  const int lane = threadIdx.x & 31;
  for (int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); row < R; row += gridDim.x * (blockDim.x >> 5)) {
    float am = 0.f, acc[32];
    for (int j = 0; j < 6; j++) { q_sum_pair(Q, n, row, lane + 32 * j, acc); am = amax32(acc, am); }
    float inv, rs2; row_scales(warp_amax(am), inv, rs2);
    for (int j = 0; j < 6; j++) {
      const int pp = lane + 32 * j;
      q_sum_pair(Q, n, row, pp, acc);
      uint4 q; uint16_t sf; enc_pair_f(acc, inv, q, sf);
      zero32(acc); dec_acc(acc, q, sf, rs2);
      store_out(out + (size_t)row * H + 32 * pp, acc);
    }
  }
}
// "GEMM emulation": writes one call's records (R * ROW_BYTES, 16 B multiple) to the producer destination.
__global__ void copy16(const uint4* src, uint4* dst, size_t n16) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) dst[i] = src[i];
}
__global__ void flip_bit(uint16_t* p) { *p ^= 1; }

}  // namespace ref
