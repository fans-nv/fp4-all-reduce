// Codec unit test vs a CPU emulation of E2M1/E4M3 (independent oracle). Usage: codec_test (one GPU).
#include "fp4_codec.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
using namespace fp4c;
static float e2m1_val[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -.5f, -1, -1.5f, -2, -3, -4, -6};
static int e2m1_enc(float x) {  // RN-even, satfinite
  if (std::isnan(x)) return 7;  // hardware cvt maps NaN (amax < ~2^-116 gives 0*inf) to +6
  float a = fabsf(x); int s = (x < 0 || (x == 0 && signbit(x))) ? 8 : 0;
  static const float g[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  if (a >= 6) return s | 7;
  int best = 0;
  for (int i = 1; i < 8; i++) {
    float d0 = fabsf(a - g[best]), d1 = fabsf(a - g[i]);
    if (d1 < d0 || (d1 == d0 && (i % 2 == 0))) best = i;
  }
  return s | best;
}
// enc + dec of one pair, mirroring the two-shot owner pattern
__global__ void k(const float* in, float inv, uint4* q, uint16_t* sf, float* dec, float rs) {
  int t = threadIdx.x; float f[32];
  for (int i = 0; i < 32; i++) f[i] = in[32 * t + i];
  uint4 qq; uint16_t ss; enc_pair_f(f, inv, qq, ss); q[t] = qq; sf[t] = ss;
  float acc[32]; zero32(acc); dec_acc(acc, qq, ss, rs);
  for (int i = 0; i < 32; i++) dec[32 * t + i] = acc[i];
}
__global__ void canon(uint32_t* w, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) w[i] = canon_neg0(w[i]); }
int main() {
  const int T = 256; std::vector<float> in(32 * T);
  srand(1); for (auto& x : in) x = ((rand() / (float)RAND_MAX) * 2 - 1) * 5.f;
  float *din, *ddec; uint4* dq; uint16_t* dsf;
  cudaMalloc(&din, in.size() * 4); cudaMalloc(&ddec, in.size() * 4); cudaMalloc(&dq, T * 16); cudaMalloc(&dsf, T * 2);
  cudaMemcpy(din, in.data(), in.size() * 4, cudaMemcpyHostToDevice);
  const float inv = 2688.f / 5.f, rs = 5.f / 2688.f;
  k<<<1, T>>>(din, inv, dq, dsf, ddec, rs);
  std::vector<uint4> q(T); std::vector<uint16_t> sf(T); std::vector<float> dec(in.size());
  cudaMemcpy(q.data(), dq, T * 16, cudaMemcpyDeviceToHost); cudaMemcpy(sf.data(), dsf, T * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(dec.data(), ddec, in.size() * 4, cudaMemcpyDeviceToHost);
  if (cudaGetLastError() != cudaSuccess) { printf("cuda error\n"); return 1; }
  long bad_nib = 0, bad_dec = 0;
  for (int t = 0; t < T; t++) {
    uint32_t w[4] = {q[t].x, q[t].y, q[t].z, q[t].w};
    for (int b = 0; b < 2; b++) {
      // use the GPU's scale byte; CPU recomputes values from it
      int s = (sf[t] >> (8 * b)) & 0xff;
      __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)s, __NV_E4M3); float sff = __half2float(*(__half*)&hr);
      float m = sff > 0 ? inv / sff : 0;
      for (int i = 0; i < 16; i++) {
        int idx = 16 * b + i; int nib = (w[idx / 8] >> (4 * (idx % 8))) & 0xf;
        int want = e2m1_enc(in[32 * t + idx] * m);
        if ((nib & 7) != (want & 7) || ((nib & 7) && nib != want)) { if (bad_nib < 8) printf("nib t=%d i=%d got %x want %x x=%g\n", t, idx, nib, want, in[32 * t + idx] * m); bad_nib++; }
        float d = e2m1_val[nib] * (sff * rs);
        if (d != dec[32 * t + idx] && !(d == 0 && dec[32 * t + idx] == 0)) { if (bad_dec < 8) printf("dec t=%d i=%d got %g want %g\n", t, idx, dec[32 * t + idx], d); bad_dec++; }
      }
    }
  }
  // canonicalization: exhaustive over nibble patterns in 16-bit halves
  std::vector<uint32_t> cw(65536); for (int i = 0; i < 65536; i++) cw[i] = (uint32_t)i | ((uint32_t)(65535 - i) << 16);
  uint32_t* dc; cudaMalloc(&dc, 65536 * 4); cudaMemcpy(dc, cw.data(), 65536 * 4, cudaMemcpyHostToDevice);
  canon<<<256, 256>>>(dc, 65536); std::vector<uint32_t> co(65536); cudaMemcpy(co.data(), dc, 65536 * 4, cudaMemcpyDeviceToHost);
  long bad_c = 0;
  for (int i = 0; i < 65536; i++) {
    uint32_t want = 0;
    for (int k2 = 0; k2 < 8; k2++) { uint32_t nb = (cw[i] >> (4 * k2)) & 0xf; if (nb == 8) nb = 0; want |= nb << (4 * k2); }
    bad_c += co[i] != want;
  }
  printf("codec_test: bad_nibbles=%ld bad_decode=%ld bad_canon=%ld -> %s\n", bad_nib, bad_dec, bad_c, (bad_nib || bad_dec || bad_c) ? "FAIL" : "PASS");
  return (bad_nib || bad_dec || bad_c) ? 1 : 0;
}
