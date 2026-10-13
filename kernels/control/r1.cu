// fp4-allreduce: round-1 fused kernel behind Fp4ArVariant, variant r1 (control).
// Variant r1 (control): round-1 Lane B kernels (ref/fp4_ar_r1.cu, config TWOPASS=1 JU1=1 SFSTAGE=1) behind Fp4ArVariant.
// Changes vs round 1: codec v2 (FMA decode) from fp4_codec.cuh; two-shot gains XPORT/COMPUTE modes; pre-quantized input (a->xq) skips encode.
// Build knobs: -DMINB=<blocks/SM bound> (3 = r2_c3g, 4 = r2_c4), -DR1_NAME=<variant name> (default r1).
// tune[0] = lag (0 -> 1), tune[1] = CTAs/SM cap (0 -> occupancy), tune[2] = warps per CTA (0 -> 8, max 8).
#include "fp4ar_api.h"
#include "fp4_codec.cuh"
#include <cstdio>
#include <algorithm>

#ifndef MINB
#define MINB 3
#endif
#ifndef R1_NAME
#define R1_NAME r1
#endif
#define R1_CAT2(a, b) a##b
#define R1_CAT(a, b) R1_CAT2(a, b)
#define R1_STR2(a) #a
#define R1_STR(a) R1_STR2(a)

namespace {
using namespace fp4c;
constexpr int MAXW = 8;  // s_sf staging sized for 8 warps

struct P {
  const __nv_bfloat16* x; __nv_bfloat16* out;
  int R, me, ntiles, lag;
  char* win[4]; char* self;
  size_t S, flagOff, sfOff, rsOff; int ntmax; size_t s2Off, flag2Off;
  unsigned* genPtr; unsigned* doneCtr;
  const char* xq; size_t qOff; float gsme;  // pre-quantized input (3456 B rows), input slot offset, own static scale
};
// Loads pair pp of a pre-quantized row (INPUT-FP4.md: 3072 value bytes then 384 scales); r1 ships gs as its row scale.
__device__ __forceinline__ void load_q(const P& p, int row, int pp, uint4& q, uint16_t& sf) {
  const char* rr = p.xq + (size_t)row * (VB + SB);
  q = __ldg((const uint4*)rr + pp);
  sf = __ldg((const uint16_t*)(rr + VB) + pp);
}
__device__ __forceinline__ float load_q_rs(const P& p, int) { return p.gsme; }

__device__ __forceinline__ void st_release_sys(unsigned* a, unsigned v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(a), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* a) {
  unsigned v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v;
}
template <int NR>
__device__ __forceinline__ char* sel_win(const P& p, int r) {
  char* w = p.win[0];
#pragma unroll
  for (int k = 1; k < NR; k++) w = (r == k) ? p.win[k] : w;
  return w;
}

// ---------------- one-shot ----------------
template <int MODE, int NR>
__device__ __forceinline__ void produce_row(const P& p, char* const* dst, int row, int parity, int lane, unsigned gen) {
  __shared__ __align__(16) uint16_t s_sf[MAXW * PAIRS];
  const __nv_bfloat16* xr = p.x + (size_t)row * H;
  float inv = 0.f, rs = 1.f;
  if (MODE != FP4AR_XPORT) { if (p.xq) rs = load_q_rs(p, row); else row_scales(row_amax_warp(xr, lane), inv, rs); }
#pragma unroll 1
  for (int j = 0; j < 6; j++) {
    uint4 q; uint16_t sf;
    if (MODE != FP4AR_XPORT && p.xq) load_q(p, row, lane + 32 * j, q, sf);
    else if (MODE != FP4AR_XPORT) {
      uint4 v[4];
#pragma unroll
      for (int k = 0; k < 4; k++) v[k] = __ldg((const uint4*)xr + 4 * (lane + 32 * j) + k);
      enc_pair(v, inv, q, sf);
    } else { q = make_uint4(gen, row, lane, j); sf = (uint16_t)j; }
    const int pp = lane + 32 * j;
    const size_t off = (size_t)(parity * NR + p.me) * p.S;
    if (MODE == FP4AR_COMPUTE) {
      ((uint4*)(p.self + off + (size_t)row * VB))[pp] = q;
      ((uint16_t*)(p.self + off + p.sfOff + (size_t)row * SB))[pp] = sf;
    } else {
#pragma unroll
      for (int r = 0; r < NR; r++) ((uint4*)(dst[r] + off + (size_t)row * VB))[pp] = q;
      s_sf[(threadIdx.x >> 5) * PAIRS + pp] = sf;
    }
  }
  // scales leave as 24 x 16 B stores per row instead of 192 x 2 B
  if (MODE != FP4AR_COMPUTE) {
    __syncwarp();
    if (lane < 24) {
      uint4 t = ((const uint4*)(s_sf + (threadIdx.x >> 5) * PAIRS))[lane];
      const size_t off = (size_t)(parity * NR + p.me) * p.S + p.sfOff + (size_t)row * SB;
#pragma unroll
      for (int r = 0; r < NR; r++) ((uint4*)(dst[r] + off))[lane] = t;
    }
    __syncwarp();
  }
  if (lane == 0) {
    const size_t off = (size_t)(parity * NR + p.me) * p.S + p.rsOff;
    if (MODE == FP4AR_COMPUTE) ((float*)(p.self + off))[row] = rs;
    else {
#pragma unroll
      for (int r = 0; r < NR; r++) ((float*)(dst[r] + off))[row] = rs;
    }
  }
}

template <int MODE, int NR>
__device__ __forceinline__ void consume_row(const P& p, int row, int parity, int lane) {
  const char* sb[NR]; float rs[NR];
#pragma unroll
  for (int s = 0; s < NR; s++) {
    const int src = MODE == FP4AR_COMPUTE ? p.me : s;
    sb[s] = p.self + (size_t)(parity * NR + src) * p.S;
    rs[s] = __ldcg((const float*)(sb[s] + p.rsOff) + row);
  }
#pragma unroll 1
  for (int j = 0; j < 6; j++) {
    const int pp = lane + 32 * j;
    uint4 q[NR]; uint16_t sf[NR];
#pragma unroll
    for (int s = 0; s < NR; s++) {
      q[s] = __ldcg((const uint4*)(sb[s] + (size_t)row * VB) + pp);
      sf[s] = __ldcg((const unsigned short*)(sb[s] + p.sfOff + (size_t)row * SB) + pp);
    }
    float acc[32]; zero32(acc);
#pragma unroll
    for (int s = 0; s < NR; s++) dec_acc(acc, q[s], sf[s], rs[s]);
    store_out(p.out + (size_t)row * H + 32 * pp, acc);
  }
}

__device__ __forceinline__ void finish_gen(const P& p, unsigned gen) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    unsigned old = atomicAdd(p.doneCtr, 1u);
    if (old == gridDim.x - 1) { *p.doneCtr = 0; *(volatile unsigned*)p.genPtr = gen; __threadfence(); }
  }
}

template <int MODE, int NR>
__global__ void __launch_bounds__(256, MINB) os_kernel(P p) {
  __shared__ unsigned s_gen;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, TR = blockDim.x >> 5;
  if (tid == 0) s_gen = *(volatile unsigned*)p.genPtr + 1;
  __syncthreads();
  const unsigned gen = s_gen; const int parity = gen & 1;
  const int G = gridDim.x, bid = blockIdx.x;
  // destination order rotated by own rank so each rank's ingress sees one writer at a time
  char* dst[NR];
#pragma unroll
  for (int k = 0; k < NR; k++) {
    const int want = (p.me + k) % NR; dst[k] = p.win[0];
#pragma unroll
    for (int r = 1; r < NR; r++) if (r == want) dst[k] = p.win[r];
  }
  const int nmy = bid < p.ntiles ? (p.ntiles - bid + G - 1) / G : 0;
  for (int i = 0; i < nmy + p.lag; i++) {
    if (i < nmy) {
      const int tile = bid + i * G, row = tile * TR + warp;
      if (row < p.R) produce_row<MODE, NR>(p, dst, row, parity, lane, gen);
      if (MODE != FP4AR_COMPUTE) {
        __syncthreads();
        if (tid < NR && tid != p.me) st_release_sys((unsigned*)(sel_win<NR>(p, tid) + p.flagOff) + (size_t)p.me * p.ntmax + tile, gen);
      }
    }
    if (i >= p.lag) {
      const int tile = bid + (i - p.lag) * G, row = tile * TR + warp;
      if (MODE != FP4AR_COMPUTE && tid < NR && tid != p.me) {
        const unsigned* f = (const unsigned*)(p.self + p.flagOff) + (size_t)tid * p.ntmax + tile;
        while ((int)(ld_acquire_sys(f) - gen) < 0) {}
      }
      __syncthreads();
      if (MODE != FP4AR_XPORT && row < p.R) consume_row<MODE, NR>(p, row, parity, lane);
    }
  }
  if (MODE != FP4AR_COMPUTE) finish_gen(p, gen);
}

// ---------------- two-shot (owner reduce + FP4 all-gather of the re-encoded sum) ----------------
// Owner decodes all NR contributions in rank order, re-encodes the sum with its own row amax, pushes it
// to every rank and writes the decode of its own re-encoding, so all ranks are bitwise identical.
// XPORT: identical stores/flags/waits with placeholder payload, no loads or codec. COMPUTE: identical codec
// work against the local window only (sources all = own slot), no flags or waits.
template <int MODE, int NR>
__device__ __forceinline__ void ts_owner(const P& p, int parity, int row, int lane, char* const* d2, size_t d2off, unsigned gen) {
  const char* sb[NR]; float rs[NR];
#pragma unroll
  for (int s = 0; s < NR; s++) {
    sb[s] = p.self + (size_t)(parity * NR + (MODE == FP4AR_COMPUTE ? p.me : s)) * p.S;
    if (MODE != FP4AR_XPORT) rs[s] = __ldcg((const float*)(sb[s] + p.rsOff) + row);
  }
  float inv = 0.f, rs2 = 1.f;
  if (MODE != FP4AR_XPORT) {
    float am = 0.f;
#pragma unroll 1
    for (int j = 0; j < 6; j++) {
      const int pp = lane + 32 * j; float acc[32]; zero32(acc);
#pragma unroll
      for (int s = 0; s < NR; s++)
        dec_acc(acc, __ldcg((const uint4*)(sb[s] + (size_t)row * VB) + pp), __ldcg((const unsigned short*)(sb[s] + p.sfOff + (size_t)row * SB) + pp), rs[s]);
      am = amax32(acc, am);
    }
    row_scales(warp_amax(am), inv, rs2);
  }
#pragma unroll 1
  for (int j = 0; j < 6; j++) {
    const int pp = lane + 32 * j; uint4 q; uint16_t sf;
    float acc[32];
    if (MODE != FP4AR_XPORT) {
      zero32(acc);
#pragma unroll
      for (int s = 0; s < NR; s++)
        dec_acc(acc, __ldcg((const uint4*)(sb[s] + (size_t)row * VB) + pp), __ldcg((const unsigned short*)(sb[s] + p.sfOff + (size_t)row * SB) + pp), rs[s]);
      enc_pair_f(acc, inv, q, sf);
    } else { q = make_uint4(gen, row, lane, j); sf = (uint16_t)j; }
    if (MODE == FP4AR_COMPUTE) {
      ((uint4*)(p.self + d2off + (size_t)row * VB))[pp] = q;
      ((uint16_t*)(p.self + d2off + p.sfOff + (size_t)row * SB))[pp] = sf;
    } else {
#pragma unroll
      for (int r = 0; r < NR; r++) {
        ((uint4*)(d2[r] + d2off + (size_t)row * VB))[pp] = q;
        ((uint16_t*)(d2[r] + d2off + p.sfOff + (size_t)row * SB))[pp] = sf;
      }
    }
    if (MODE != FP4AR_XPORT) {
      zero32(acc); dec_acc(acc, q, sf, rs2);
      store_out(p.out + (size_t)row * H + 32 * pp, acc);
    }
  }
  if (lane == 0) {
    if (MODE == FP4AR_COMPUTE) ((float*)(p.self + d2off + p.rsOff))[row] = rs2;
    else {
#pragma unroll
      for (int r = 0; r < NR; r++) ((float*)(d2[r] + d2off + p.rsOff))[row] = rs2;
    }
  }
}

template <int MODE, int NR>
__global__ void __launch_bounds__(256, MINB) ts_kernel(P p) {
  __shared__ unsigned s_gen;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, TR = blockDim.x >> 5;
  if (tid == 0) s_gen = *(volatile unsigned*)p.genPtr + 1;
  __syncthreads();
  const unsigned gen = s_gen; const int parity = gen & 1;
  const int G = gridDim.x, bid = blockIdx.x, L = p.lag;
  const int nmy = bid < p.ntiles ? (p.ntiles - bid + G - 1) / G : 0;
  char* win[NR];
#pragma unroll
  for (int r = 0; r < NR; r++) win[r] = p.win[r];
  const size_t d2off = p.s2Off + (size_t)parity * p.S;
  for (int i = 0; i < nmy + 2 * L; i++) {
    if (i < nmy) {  // A: encode own contribution, send to the tile owner
      const int tile = bid + i * G, owner = (bid + i) % NR, row = tile * TR + warp;
      char* d = win[0];
#pragma unroll
      for (int r = 1; r < NR; r++) if (r == owner) d = win[r];
      if (MODE == FP4AR_COMPUTE) d = p.self;
      if (row < p.R) {
        const __nv_bfloat16* xr = p.x + (size_t)row * H;
        float inv = 0.f, rs = 1.f;
        if (MODE != FP4AR_XPORT) { if (p.xq) rs = load_q_rs(p, row); else row_scales(row_amax_warp(xr, lane), inv, rs); }
        char* base = d + (size_t)(parity * NR + p.me) * p.S;
#pragma unroll 1
        for (int j = 0; j < 6; j++) {
          uint4 q; uint16_t sf;
          if (MODE != FP4AR_XPORT && p.xq) load_q(p, row, lane + 32 * j, q, sf);
          else if (MODE != FP4AR_XPORT) {
            uint4 v[4];
#pragma unroll
            for (int k = 0; k < 4; k++) v[k] = __ldg((const uint4*)xr + 4 * (lane + 32 * j) + k);
            enc_pair(v, inv, q, sf);
          } else { q = make_uint4(gen, row, lane, j); sf = (uint16_t)j; }
          ((uint4*)(base + (size_t)row * VB))[lane + 32 * j] = q;
          ((uint16_t*)(base + p.sfOff + (size_t)row * SB))[lane + 32 * j] = sf;
        }
        if (lane == 0) ((float*)(base + p.rsOff))[row] = rs;
      }
      if (MODE != FP4AR_COMPUTE) {
        __syncthreads();
        if (tid == 0 && owner != p.me) st_release_sys((unsigned*)(d + p.flagOff) + (size_t)p.me * p.ntmax + tile, gen);
      }
    }
    if (i >= L && i - L < nmy) {  // B: owner reduces, re-encodes, all-gathers
      const int k = i - L, tile = bid + k * G, owner = (bid + k) % NR, row = tile * TR + warp;
      if (owner == p.me) {
        if (MODE != FP4AR_COMPUTE) {
          if (tid < NR && tid != p.me) {
            const unsigned* f = (const unsigned*)(p.self + p.flagOff) + (size_t)tid * p.ntmax + tile;
            while ((int)(ld_acquire_sys(f) - gen) < 0) {}
          }
          __syncthreads();
        }
        if (row < p.R) ts_owner<MODE, NR>(p, parity, row, lane, win, d2off, gen);
        if (MODE != FP4AR_COMPUTE) {
          __syncthreads();
          if (tid < NR && tid != p.me) st_release_sys((unsigned*)(sel_win<NR>(p, tid) + p.flag2Off) + tile, gen);
        }
      }
    }
    if (i >= 2 * L && i - 2 * L < nmy) {  // C: non-owners decode the gathered sum
      const int k = i - 2 * L, tile = bid + k * G, owner = (bid + k) % NR, row = tile * TR + warp;
      if (owner != p.me) {
        if (MODE != FP4AR_COMPUTE) {
          if (tid == 0) {
            const unsigned* f = (const unsigned*)(p.self + p.flag2Off) + tile;
            while ((int)(ld_acquire_sys(f) - gen) < 0) {}
          }
          __syncthreads();
        }
        if (MODE != FP4AR_XPORT && row < p.R) {
          const char* b = p.self + d2off;
          const float rs = __ldcg((const float*)(b + p.rsOff) + row);
#pragma unroll 1
          for (int j = 0; j < 6; j++) {
            const int pp = lane + 32 * j; float acc[32]; zero32(acc);
            dec_acc(acc, __ldcg((const uint4*)(b + (size_t)row * VB) + pp), __ldcg((const unsigned short*)(b + p.sfOff + (size_t)row * SB) + pp), rs);
            store_out(p.out + (size_t)row * H + 32 * pp, acc);
          }
        }
      }
    }
  }
  if (MODE != FP4AR_COMPUTE) finish_gen(p, gen);
}

// ---------------- host side ----------------
struct Priv {
  P base; unsigned* ctr;
  int occ[2][3][2];  // [ts][mode][nr==4] at 8 warps
};

size_t layout(int max_rows, int nranks, P& p) {
  p.S = ((size_t)max_rows * ROW_BYTES + 4095) / 4096 * 4096;
  p.sfOff = (size_t)max_rows * VB; p.rsOff = p.sfOff + (size_t)max_rows * SB;
  p.s2Off = 2 * (size_t)nranks * p.S; p.flagOff = p.s2Off + 2 * p.S; p.ntmax = max_rows;
  p.flag2Off = p.flagOff + (size_t)nranks * p.ntmax * 4;
  p.qOff = (p.flag2Off + (size_t)p.ntmax * 4 + 4095) / 4096 * 4096;
  return p.qOff + (size_t)max_rows * (VB + SB);
}

size_t win_bytes(int max_rows, int hidden, int nranks) {
  if (hidden != H) return 0;
  P p{}; return layout(max_rows, nranks, p);
}

template <int M, int NR> const void* kfn(bool ts) { return ts ? (const void*)ts_kernel<M, NR> : (const void*)os_kernel<M, NR>; }
const void* pick(bool ts, int mode, int nr) {
  if (nr == 2) return mode == FP4AR_XPORT ? kfn<FP4AR_XPORT, 2>(ts) : mode == FP4AR_COMPUTE ? kfn<FP4AR_COMPUTE, 2>(ts) : kfn<FP4AR_FUSED, 2>(ts);
  return mode == FP4AR_XPORT ? kfn<FP4AR_XPORT, 4>(ts) : mode == FP4AR_COMPUTE ? kfn<FP4AR_COMPUTE, 4>(ts) : kfn<FP4AR_FUSED, 4>(ts);
}

int init(Fp4ArCtx* c, int max_rows, int hidden) {
  if (hidden != H || (c->nranks != 2 && c->nranks != 4)) { fprintf(stderr, "r1: needs hidden 6144, TP2/TP4\n"); return 1; }
  Priv* v = new Priv{};
  P& p = v->base;
  layout(max_rows, c->nranks, p);
  p.me = c->rank;
  for (int r = 0; r < c->nranks; r++) p.win[r] = c->peer[r];
  p.self = c->peer[c->rank];
  if (cudaMalloc(&v->ctr, 8) != cudaSuccess || cudaMemset(v->ctr, 0, 8) != cudaSuccess) return 2;
  if (cudaMemset(c->peer[c->rank], 0, c->win_bytes) != cudaSuccess) return 2;
  p.genPtr = v->ctr; p.doneCtr = v->ctr + 1;
  for (int ts = 0; ts < 2; ts++)
    for (int m = 0; m < 3; m++)
      for (int q = 0; q < 2; q++)
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v->occ[ts][m][q], pick(ts, m, q ? 4 : 2), 256, 0) != cudaSuccess) return 3;
  if (cudaDeviceSynchronize() != cudaSuccess) return 4;
  c->priv = v;
  return 0;
}

int launch(Fp4ArCtx* c, const Fp4ArArgs* a, cudaStream_t s) {
  Priv* v = (Priv*)c->priv;
  P p = v->base;
  const bool ts = a->algo == FP4AR_TWOSHOT;
  const int warps = a->tune[2] > 0 ? std::min(a->tune[2], MAXW) : 8;
  p.x = a->x; p.out = a->out; p.R = a->rows; p.xq = (const char*)a->xq; p.gsme = a->gs[c->rank]; p.lag = a->tune[0] > 0 ? a->tune[0] : 1;
  p.ntiles = (a->rows + warps - 1) / warps;
  if (p.ntiles > p.ntmax) return 1;
  int occ = v->occ[ts][a->mode][c->nranks == 4];
  if (warps != 8 && cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, pick(ts, a->mode, c->nranks), warps * 32, 0) != cudaSuccess) return 2;
  const int cpsm = a->tune[1] > 0 ? std::min(a->tune[1], occ) : occ;
  const int grid = std::min(c->sm_count * cpsm, p.ntiles);
  void* args[] = {&p};
  return cudaLaunchKernel(pick(ts, a->mode, c->nranks), grid, warps * 32, args, 0, s) == cudaSuccess ? 0 : 3;
}

void fini(Fp4ArCtx* c) {
  Priv* v = (Priv*)c->priv;
  if (!v) return;
  cudaFree(v->ctr); delete v; c->priv = nullptr;
}

// Q-window: one fixed slot; the kernel copies it out to peers within the call, so no rotation or refill is needed.
uint8_t* input_slot(Fp4ArCtx* c, int) { Priv* v = (Priv*)c->priv; return (uint8_t*)(v->base.self + v->base.qOff); }

const Fp4ArVariant kVariant = {R1_STR(R1_NAME), true, win_bytes, init, launch, fini, input_slot};
}  // namespace

const Fp4ArVariant* R1_CAT(fp4ar_register_, R1_NAME)() { return &kVariant; }
