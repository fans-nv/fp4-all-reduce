// Standalone driver for lane NORM (fused NVFP4 AllReduce + residual + Gemma RMSNorm), adapted from kernels/lamport/drv.cu.
// --epi 0 (AR only) | 1 (norm, BF16 out) | 2 (norm, NVFP4 out). Modes: check (bitwise gate), fused|xport|compute (timing).
#include "fp4ar_api.h"
#include "fp4_codec.cuh"
#include "lnorm_epi.cuh"
#include <nccl_device/core.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <algorithm>
using namespace fp4c;

size_t lnorm_window_bytes(int max_rows, int hidden, int nranks);
int lnorm_init(Fp4ArCtx* c, int max_rows, int hidden);
int lnorm_launch(Fp4ArCtx* c, const Fp4ArArgs* a, const lne::NormArgs* n, cudaStream_t s);
void lnorm_fini(Fp4ArCtx* c);

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)
#define NK(x) do { ncclResult_t e_ = (x); if (e_ != ncclSuccess) { fprintf(stderr, "NCCL %s:%d %s\n", __FILE__, __LINE__, ncclGetErrorString(e_)); exit(1);} } while (0)

__device__ __forceinline__ uint32_t hsh(uint32_t a) { a ^= a >> 16; a *= 0x7feb352dU; a ^= a >> 15; a *= 0x846ca68bU; a ^= a >> 16; return a; }
__constant__ float MAGS[8] = {1.f, .5f, 2.f, .25f, 1.5f, .75f, 3.f, .125f};
__global__ void gen_input(__nv_bfloat16* x, int src, size_t N, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < N; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h1 = hsh(seed ^ hsh((uint32_t)i * 2 + 1 + (uint32_t)(i >> 31))), h2 = hsh(h1 ^ 0x9e3779b9U ^ src);
    float u1 = (h1 >> 8) * (1.f / 16777216.f) + 1e-7f, u2 = (h2 >> 8) * (1.f / 16777216.f);
    x[i] = __float2bfloat16(sqrtf(-2.f * logf(u1)) * cospif(2.f * u2) * MAGS[src]);
  }
}
__global__ void gen_outliers(__nv_bfloat16* x, int src, size_t N, uint32_t seed) {
  if (threadIdx.x < 32) {
    uint32_t h = hsh(seed * 7919u + 104729u * (threadIdx.x + 1));
    size_t pos = ((size_t)h * 2654435761u) % N;
    x[pos] = __float2bfloat16(((hsh(h) & 1) ? 40.f : -40.f) * MAGS[src]);
  }
}
__global__ void zero_row(__nv_bfloat16* x, int row) { for (int i = threadIdx.x; i < H; i += blockDim.x) x[(size_t)row * H + i] = __float2bfloat16(0.f); }
__global__ void ref_encode(const __nv_bfloat16* x, uint8_t* v, uint8_t* s, float* rsv, int R) {
  int lane = threadIdx.x & 31;
  for (int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); row < R; row += gridDim.x * (blockDim.x >> 5)) {
    const __nv_bfloat16* xr = x + (size_t)row * H;
    float inv, rs; row_scales(row_amax_warp(xr, lane), inv, rs);
    for (int j = 0; j < 6; j++) {
      int pp = lane + 32 * j; uint4 t[4];
      for (int k = 0; k < 4; k++) t[k] = ((const uint4*)xr)[4 * pp + k];
      uint4 q; uint16_t sf; enc_pair(t, inv, q, sf);
      ((uint4*)(v + (size_t)row * VB))[pp] = q; ((uint16_t*)(s + (size_t)row * SB))[pp] = sf;
    }
    if (lane == 0) rsv[row] = rs;
  }
}
// Pre-quantized input emulation (INPUT-FP4.md): static gs, spec-v2 block encode with inv = 1/gs, -0 canonicalized;
// row-interleaved 3456 B records into xq (if non-null) + reference arrays (if non-null).
__global__ void q_encode(const __nv_bfloat16* x, uint8_t* xq, uint8_t* v, uint8_t* s, float* rsv, int R, float gs) {
  int lane = threadIdx.x & 31; const float inv = __fdiv_rn(1.f, gs);
  for (int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); row < R; row += gridDim.x * (blockDim.x >> 5)) {
    for (int j = 0; j < 6; j++) {
      int pp = lane + 32 * j; uint4 t[4];
      for (int k = 0; k < 4; k++) t[k] = ((const uint4*)(x + (size_t)row * H))[4 * pp + k];
      uint4 q; uint16_t sf; enc_pair(t, inv, q, sf); q = canon_neg0(q);
      if (xq) { ((uint4*)(xq + (size_t)row * lne::QROW))[pp] = q; ((uint16_t*)(xq + (size_t)row * lne::QROW + VB))[pp] = sf; }
      if (v) { ((uint4*)(v + (size_t)row * VB))[pp] = q; ((uint16_t*)(s + (size_t)row * SB))[pp] = sf; }
    }
    if (lane == 0 && rsv) rsv[row] = gs;
  }
}
// Reference epilogue: rank-order fp4c decode-sum (FP32) -> z = sum + res (FP32) -> res_out = bf16(z);
// ss = FP32 sum of z^2 (lane order j, i; xor butterfly) -> rinv -> y = bf16(z) * rinv * (wbias + w) -> BF16 or vLLM NVFP4.
__global__ void ref_norm(const uint8_t* v, const uint8_t* s, const float* rsv, int n, int R, lne::NormArgs nm, __nv_bfloat16* out) {
  const int lane = threadIdx.x & 31; const size_t N = (size_t)R;
  for (int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); row < R; row += gridDim.x * (blockDim.x >> 5)) {
    uint4 hold[6][4]; float ss0 = 0.f, ss1 = 0.f;  // even / odd element partial sums of z^2
    for (int j = 0; j < 6; j++) {
      const int pp = lane + 32 * j; float acc[32]; zero32(acc);
      for (int k = 0; k < n; k++)
        dec_acc(acc, ((const uint4*)(v + (k * N + row) * VB))[pp], ((const uint16_t*)(s + (k * N + row) * SB))[pp], rsv[k * N + row]);
      if (nm.epi == 0) { store_out(out + (size_t)row * H + 32 * pp, acc); continue; }
      const __nv_bfloat16* rr = nm.res_in + (size_t)row * H + 32 * pp;
      for (int i = 0; i < 32; i++) { acc[i] = __fadd_rn(acc[i], __bfloat162float(rr[i])); if (i & 1) ss1 = __fmaf_rn(acc[i], acc[i], ss1); else ss0 = __fmaf_rn(acc[i], acc[i], ss0); }
      store_out(nm.res_out + (size_t)row * H + 32 * pp, acc);
      __nv_bfloat162* hb = (__nv_bfloat162*)hold[j];
      for (int i = 0; i < 16; i++) hb[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
    }
    if (nm.epi == 0) continue;
    float ss = __fadd_rn(ss0, ss1);
    for (int o = 16; o; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
    const float rinv = lne::norm_rinv(ss, nm.eps);
    for (int j = 0; j < 6; j++) {
      const int pp = lane + 32 * j; uint4 w4[4];
      for (int k = 0; k < 4; k++) w4[k] = ((const uint4*)nm.w)[4 * pp + k];
      float y[32]; lne::norm_pair(hold[j], w4, rinv, nm.wbias, y);
      if (nm.epi == 1) store_out(out + (size_t)row * H + 32 * pp, y);
      else if (nm.epi == 3) {
        uint4 q[2]; uint32_t e8 = lne::quant_pair_mxfp8(y, q);
        uint8_t* qr = nm.outq + (size_t)row * lne::XROW; ((uint4*)qr)[2 * pp] = q[0]; ((uint4*)qr)[2 * pp + 1] = q[1]; qr[H + pp] = (uint8_t)e8;
      } else {
        uint4 q; uint32_t sf = lne::quant_pair_vllm(y, nm.qscale, q);
        uint8_t* qr = nm.outq + (size_t)row * lne::QROW; ((uint4*)qr)[pp] = q; ((uint16_t*)(qr + VB))[pp] = (uint16_t)sf;
      }
    }
  }
}
// FP32 exact: y = norm(sum_s x_s + res) in FP32 (for rel-L2 of the BF16 normed output; not gated).
__global__ void ref_exact(const __nv_bfloat16* xall, int n, size_t stride, int R, lne::NormArgs nm, float* out) {
  const int lane = threadIdx.x & 31;
  for (int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); row < R; row += gridDim.x * (blockDim.x >> 5)) {
    double ss = 0;
    for (int i = lane; i < H; i += 32) {
      size_t e = (size_t)row * H + i; float a = __bfloat162float(nm.res_in[e]);
      for (int s = 0; s < n; s++) a += __bfloat162float(xall[s * stride + e]);
      out[e] = a; ss += (double)a * a;
    }
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(~0u, ss, o);
    const float r = (float)(1.0 / sqrt(ss / H + nm.eps));
    for (int i = lane; i < H; i += 32) { size_t e = (size_t)row * H + i; out[e] = nm.epi == 0 ? out[e] - __bfloat162float(nm.res_in[e]) : out[e] * r * (nm.wbias + __bfloat162float(nm.w[i])); }
  }
}
struct ErrAcc { double dsq, rsq; unsigned long long mism, mism_res; };
__global__ void cmp_kernel(const uint16_t* a, const uint16_t* b, size_t n, const uint16_t* ra, const uint16_t* rb, size_t nr,
                           const __nv_bfloat16* outb, const float* ex, size_t ne, ErrAcc* e) {
  unsigned long long m = 0, mr = 0; double dsq = 0, rsq = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) m += a[i] != b[i];
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nr; i += (size_t)gridDim.x * blockDim.x) mr += ra[i] != rb[i];
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < ne; i += (size_t)gridDim.x * blockDim.x) {
    double d = (double)__bfloat162float(outb[i]) - ex[i]; dsq += d * d; rsq += (double)ex[i] * ex[i];
  }
  for (int o = 16; o; o >>= 1) { m += __shfl_xor_sync(~0u, m, o); mr += __shfl_xor_sync(~0u, mr, o); dsq += __shfl_xor_sync(~0u, dsq, o); rsq += __shfl_xor_sync(~0u, rsq, o); }
  if ((threadIdx.x & 31) == 0) { atomicAdd(&e->mism, m); atomicAdd(&e->mism_res, mr); atomicAdd(&e->dsq, dsq); atomicAdd(&e->rsq, rsq); }
}
__global__ void spin(long long ns) { long long t0 = clock64(); while (clock64() - t0 < ns * 2) {} }

struct Barrier {
  std::mutex m; std::condition_variable cv; int n, cnt = 0, phase = 0;
  explicit Barrier(int n) : n(n) {}
  void wait() { std::unique_lock<std::mutex> l(m); int ph = phase; if (++cnt == n) { cnt = 0; phase++; cv.notify_all(); } else cv.wait(l, [&] { return phase != ph; }); }
};
struct Opt { int q = 0, n = 2, epi = 1; std::string mode = "check", tag = "", json = ""; std::vector<int> rows = {1024}; int tune[4] = {0, 0, 0, 0}; int graphK = 0, iters = 200, warmup = 20, reps = 300, greps = 300; float qscale = 2688.f / 16.f; };
static Opt O; static Barrier* BAR; static std::vector<std::vector<float>> TIMES; static std::vector<std::vector<uint8_t>> OUTS, RESS; static ncclUniqueId ID;
static int FAILS = 0; static std::mutex FM;

static void worker(int rank) {
  const int n = O.n; CK(cudaSetDevice(rank));
  ncclComm_t comm; NK(ncclCommInitRank(&comm, n, ID, rank));
  cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, rank));
  int RMAX = *std::max_element(O.rows.begin(), O.rows.end());
  Fp4ArCtx c{}; c.rank = rank; c.nranks = n; c.device = rank; c.sm_count = prop.multiProcessorCount; c.comm = comm;
  c.win_bytes = lnorm_window_bytes(RMAX, H, n);
  void* wbase; NK(ncclMemAlloc(&wbase, c.win_bytes));
  NK(ncclCommWindowRegister(comm, wbase, c.win_bytes, &c.win, NCCL_WIN_COLL_SYMMETRIC));
  for (int r = 0; r < n; r++) { void* q; NK(ncclGetLsaDevicePointer(c.win, 0, r, &q)); c.peer[r] = (char*)q; }
  if (lnorm_init(&c, RMAX, H) != 0) { fprintf(stderr, "init failed\n"); exit(1); }
  if (rank == 0) printf("# lnorm q=%d epi=%d n=%d SMs=%d win=%zu MB tune=%d,%d,%d,%d qscale=%g\n", O.q, O.epi, n, c.sm_count, c.win_bytes >> 20, O.tune[0], O.tune[1], O.tune[2], O.tune[3], O.qscale);
  const size_t NMAX = (size_t)RMAX * H;
  __nv_bfloat16 *xall, *out, *res, *reso, *w; uint8_t* outq;
  CK(cudaMalloc(&xall, n * NMAX * 2)); CK(cudaMalloc(&out, NMAX * 2)); CK(cudaMalloc(&res, NMAX * 2)); CK(cudaMalloc(&reso, NMAX * 2));
  CK(cudaMalloc(&w, H * 2)); CK(cudaMalloc(&outq, (size_t)RMAX * lne::XROW));
  uint8_t* xqb; CK(cudaMalloc(&xqb, (size_t)RMAX * lne::QROW));
  auto gsof = [](int s) { return 0.03f + 0.01f * s; };
  float* small; CK(cudaMalloc(&small, 64)); CK(cudaMemset(small, 0, 64));
  gen_input<<<24, 256, 0, st>>>(w, 7, H, 4242u);
  FILE* jf = (rank == 0 && !O.json.empty()) ? fopen(O.json.c_str(), "a") : nullptr;
  for (int R : O.rows) {
    const size_t N = (size_t)R * H;
    for (int s = 0; s < n; s++) { gen_input<<<1024, 256, 0, st>>>(xall + s * NMAX, s, N, 1000u + 17u * R); gen_outliers<<<1, 32, 0, st>>>(xall + s * NMAX, s, N, 1000u + 17u * R); zero_row<<<1, 256, 0, st>>>(xall + s * NMAX, R / 3); }
    gen_input<<<1024, 256, 0, st>>>(res, 5, N, 777u + R);
    CK(cudaStreamSynchronize(st));
    Fp4ArArgs a{}; a.x = xall + rank * NMAX; a.out = out; a.rows = R; a.hidden = H; a.algo = FP4AR_ONESHOT;
    for (int k = 0; k < 4; k++) a.tune[k] = O.tune[k];
    if (O.q) { a.xq = xqb; for (int s2 = 0; s2 < n; s2++) a.gs[s2] = gsof(s2);
      if (O.mode != "check") { q_encode<<<1024, 256, 0, st>>>(xall + rank * NMAX, xqb, nullptr, nullptr, nullptr, R, gsof(rank)); CK(cudaStreamSynchronize(st)); } }
    lne::NormArgs nm{res, reso, w, 1e-6f, 1.0f, O.epi, outq, O.qscale};
    auto call = [&](int mode, int rows) { Fp4ArArgs b = a; b.mode = mode; b.rows = rows; if (lnorm_launch(&c, &b, &nm, st) != 0) { fprintf(stderr, "launch failed\n"); exit(1); } };
    if (O.mode == "check") {
      uint8_t *rv, *rsf; float *rrs, *ex; __nv_bfloat16 *rc, *rres; uint8_t* rq;
      CK(cudaMalloc(&rv, n * (size_t)R * VB)); CK(cudaMalloc(&rsf, n * (size_t)R * SB)); CK(cudaMalloc(&rrs, n * (size_t)R * 4));
      CK(cudaMalloc(&rc, N * 2)); CK(cudaMalloc(&rres, N * 2)); CK(cudaMalloc(&rq, (size_t)R * lne::XROW)); CK(cudaMalloc(&ex, N * 4));
      for (int s = 0; s < n; s++)
        if (O.q) q_encode<<<1024, 256, 0, st>>>(xall + s * NMAX, s == rank ? xqb : nullptr, rv + s * (size_t)R * VB, rsf + s * (size_t)R * SB, rrs + s * (size_t)R, R, gsof(s));
        else ref_encode<<<1024, 256, 0, st>>>(xall + s * NMAX, rv + s * (size_t)R * VB, rsf + s * (size_t)R * SB, rrs + s * (size_t)R, R);
      lne::NormArgs rn = nm; rn.res_out = rres; rn.outq = rq;
      if (getenv("LN_NEGCTL")) rn.wbias = 1.0f + 0.0078125f;  // negative control: gate must FAIL (out only; res still matches)
      ref_norm<<<1024, 256, 0, st>>>(rv, rsf, rrs, n, R, rn, rc);
      ref_exact<<<1024, 256, 0, st>>>(xall, n, NMAX, R, nm, ex);
      ErrAcc* e; CK(cudaMalloc(&e, sizeof(ErrAcc)));
      const bool q = O.epi >= 2;
      const size_t ob = q ? (size_t)R * lne::orow_bytes(O.epi) : N * 2, rb = O.epi ? N * 2 : 0;
      auto check = [&](const char* what) {
        CK(cudaMemsetAsync(e, 0, sizeof(ErrAcc), st));
        cmp_kernel<<<1024, 256, 0, st>>>((const uint16_t*)(q ? (void*)outq : (void*)out), (const uint16_t*)(q ? (void*)rq : (void*)rc), ob / 2,
                                         (const uint16_t*)reso, (const uint16_t*)rres, rb / 2, out, ex, q ? 0 : N, e);
        ErrAcc h; CK(cudaMemcpyAsync(&h, e, sizeof(h), cudaMemcpyDeviceToHost, st)); CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
        OUTS[rank].resize(ob); CK(cudaMemcpy(OUTS[rank].data(), q ? (void*)outq : (void*)out, ob, cudaMemcpyDeviceToHost));
        RESS[rank].resize(rb); if (rb) CK(cudaMemcpy(RESS[rank].data(), reso, rb, cudaMemcpyDeviceToHost));
        BAR->wait();
        int same = 1; if (rank == 0) for (int r = 1; r < n; r++) same &= memcmp(OUTS[0].data(), OUTS[r].data(), ob) == 0 && memcmp(RESS[0].data(), RESS[r].data(), rb) == 0;
        bool ok = h.mism == 0 && h.mism_res == 0 && (rank != 0 || same);
        if (!ok) { std::lock_guard<std::mutex> l(FM); FAILS++; }
        printf("CHECK lnorm epi=%d %s n=%d R=%d rank=%d rel_l2_vs_fp32=%.6f mism_out=%llu/%zu mism_res=%llu%s %s\n", O.epi, what, n, R, rank,
               q ? 0.0 : sqrt(h.dsq / h.rsq), h.mism, ob / 2, h.mism_res, rank == 0 ? (same ? " ranks_identical=1" : " ranks_identical=0") : "", ok ? "PASS" : "FAIL");
        fflush(stdout); BAR->wait();
      };
      auto clr = [&]() { CK(cudaMemsetAsync(out, 0xff, N * 2, st)); CK(cudaMemsetAsync(reso, 0xff, N * 2, st)); CK(cudaMemsetAsync(outq, 0xff, (size_t)R * lne::XROW, st)); };
      clr(); call(FP4AR_FUSED, R); check("first");
      clr(); for (int i = 0; i < O.reps; i++) call(FP4AR_FUSED, R); check("reps");
      clr(); for (int i = 0; i < 60; i++) call(FP4AR_FUSED, (i & 1) ? R : std::max(1, R / 2 + 13)); call(FP4AR_FUSED, R); check("alt_rows");
      for (int i = 0; i < 7; i++) { call(FP4AR_XPORT, R); call(FP4AR_FUSED, R); } clr(); call(FP4AR_FUSED, R); check("after_xport");
      clr(); for (int i = 0; i < 200; i++) { if (rank == n - 1) spin<<<1, 1, 0, st>>>(20000 + 10000 * (i % 3)); call(FP4AR_FUSED, R); } check("skew_last");
      clr(); for (int i = 0; i < 200; i++) { if (rank == 0 && (i % 4) == 0) spin<<<1, 1, 0, st>>>(50000); call(FP4AR_FUSED, R); } check("skew_first");
      {
        cudaGraph_t g; cudaGraphExec_t gx; CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        for (int k = 0; k < 10; k++) call(FP4AR_FUSED, R);
        CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphInstantiate(&gx, g, 0)); CK(cudaGraphDestroy(g));
        clr(); for (int i = 0; i < O.greps; i++) CK(cudaGraphLaunch(gx, st)); check("graph");
        CK(cudaGraphExecDestroy(gx));
      }
      CK(cudaFree(rv)); CK(cudaFree(rsf)); CK(cudaFree(rrs)); CK(cudaFree(rc)); CK(cudaFree(rres)); CK(cudaFree(rq)); CK(cudaFree(ex)); CK(cudaFree(e));
      continue;
    }
    const int mode = O.mode == "fused" ? FP4AR_FUSED : O.mode == "xport" ? FP4AR_XPORT : O.mode == "compute" ? FP4AR_COMPUTE : -1;
    if (mode < 0) { fprintf(stderr, "bad mode\n"); exit(1); }
    auto op0 = [&]() { call(mode, R); };
    cudaGraphExec_t gx = nullptr;
    if (O.graphK > 0) {
      op0(); CK(cudaStreamSynchronize(st)); BAR->wait();
      cudaGraph_t g; CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed));
      for (int k = 0; k < O.graphK; k++) op0();
      CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphInstantiate(&gx, g, 0)); CK(cudaGraphDestroy(g));
    }
    auto op = [&]() { if (gx) CK(cudaGraphLaunch(gx, st)); else op0(); };
    std::vector<cudaEvent_t> ev(2 * O.iters); for (auto& x : ev) CK(cudaEventCreate(&x));
    for (int it = 0; it < O.warmup; it++) { NK(ncclAllReduce(small, small, 1, ncclFloat, ncclSum, comm, st)); op(); }
    CK(cudaStreamSynchronize(st)); BAR->wait();
    for (int it = 0; it < O.iters; it++) { NK(ncclAllReduce(small, small, 1, ncclFloat, ncclSum, comm, st)); CK(cudaEventRecord(ev[2 * it], st)); op(); CK(cudaEventRecord(ev[2 * it + 1], st)); }
    CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
    TIMES[rank].resize(O.iters);
    for (int it = 0; it < O.iters; it++) { float ms; CK(cudaEventElapsedTime(&ms, ev[2 * it], ev[2 * it + 1])); TIMES[rank][it] = ms * 1000.f / (O.graphK > 0 ? O.graphK : 1); }
    for (auto& x : ev) CK(cudaEventDestroy(x));
    if (gx) CK(cudaGraphExecDestroy(gx));
    BAR->wait();
    if (rank == 0) {
      std::vector<float> mx(O.iters, 0.f);
      for (int it = 0; it < O.iters; it++) for (int r = 0; r < n; r++) mx[it] = std::max(mx[it], TIMES[r][it]);
      std::sort(mx.begin(), mx.end());
      char line[512];
      snprintf(line, sizeof line, "{\"var\":\"LN\",\"q\":%d,\"epi\":%d,\"tag\":\"%s\",\"mode\":\"%s\",\"n\":%d,\"R\":%d,\"graphK\":%d,\"tune\":[%d,%d,%d,%d],\"iters\":%d,\"med_us\":%.2f,\"p10_us\":%.2f,\"p90_us\":%.2f}",
               O.q, O.epi, O.tag.c_str(), O.mode.c_str(), n, R, O.graphK, O.tune[0], O.tune[1], O.tune[2], O.tune[3], O.iters, mx[O.iters / 2], mx[O.iters / 10], mx[O.iters * 9 / 10]);
      puts(line); fflush(stdout); if (jf) { fprintf(jf, "%s\n", line); fflush(jf); }
    }
    BAR->wait();
  }
  if (jf) fclose(jf);
  CK(cudaDeviceSynchronize()); lnorm_fini(&c);
  NK(ncclCommWindowDeregister(comm, c.win)); NK(ncclMemFree(wbase)); NK(ncclCommDestroy(comm));
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i]; auto nx = [&]() { return std::string(argv[++i]); };
    auto ints = [](std::string s) { std::vector<int> v; size_t p = 0; while (p < s.size()) { size_t c = s.find(',', p); if (c == std::string::npos) c = s.size(); v.push_back(std::stoi(s.substr(p, c - p))); p = c + 1; } return v; };
    if (a == "--n") O.n = std::stoi(nx()); else if (a == "--mode") O.mode = nx(); else if (a == "--epi") O.epi = std::stoi(nx());
    else if (a == "--rows") O.rows = ints(nx());
    else if (a == "--tune") { auto v = ints(nx()); for (size_t k = 0; k < v.size() && k < 4; k++) O.tune[k] = v[k]; }
    else if (a == "--graph") O.graphK = std::stoi(nx()); else if (a == "--iters") O.iters = std::stoi(nx()); else if (a == "--warmup") O.warmup = std::stoi(nx());
    else if (a == "--reps") O.reps = std::stoi(nx()); else if (a == "--greps") O.greps = std::stoi(nx()); else if (a == "--qscale") O.qscale = std::stof(nx());
    else if (a == "--q") O.q = std::stoi(nx());
    else if (a == "--tag") O.tag = nx(); else if (a == "--json") O.json = nx();
  }
  NK(ncclGetUniqueId(&ID));
  Barrier b(O.n); BAR = &b; TIMES.resize(O.n); OUTS.resize(O.n); RESS.resize(O.n);
  std::vector<std::thread> th; for (int r = 0; r < O.n; r++) th.emplace_back(worker, r);
  for (auto& t : th) t.join();
  printf("# done fails=%d\n", FAILS);
  return FAILS ? 1 : 0;
}
