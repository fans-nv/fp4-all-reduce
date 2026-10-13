// fp4-allreduce: pull TP4 one-shot AllReduce of NVFP4 input, variant P4q (reference; not in the final pass).
// Variant P4q: TP4-lane pull AllReduce of pre-quantized NVFP4 input (INPUT-FP4.md), one-shot. Derived from Pq.
// Window mode: the producer writes straight into our symmetric slot; consumers TMA-pull peers' slots; no local encode/copy.
// Slot ring of NSQ = 5 (divides the harness graph K = 10): call N refills the slot of call N-2, which is next written at N+3.
#include "fp4ar_api.h"
#include "fp4_codec.cuh"
#include <algorithm>

#define PQ_CAT2(a, b) a##b
#define PQ_CAT(a, b) PQ_CAT2(a, b)
#ifndef FP4AR_NAME
#define FP4AR_NAME P4q
#endif
#ifndef FP4AR_REGISTER
#define FP4AR_REGISTER PQ_CAT(fp4ar_register_, FP4AR_NAME)
#endif

namespace {
using namespace fp4c;

constexpr int NTHR = 512;            // WG0 fetch, WG1 refill/copy, WG2-3 decode
constexpr int NDEC = 256, DEC0 = 256, NAUX = 128;
constexpr int HDR = 2048, MAXSLOT = 8, NSQ = 5;  // NSQ window input slots
constexpr int SMEM_MAX = 232448;
constexpr size_t RING_DEF = 113664;
constexpr int ROWQ = VB + SB;        // 3456 B input row: values then block scales (no row scale on the wire)
constexpr uint32_t FILL = 0x88888888u;  // sentinel fill: -0 nibbles; as a scale byte 0x88 > 0x7E

struct KP {
  __nv_bfloat16* out;
  char* win[FP4AR_MAX_RANKS];
  const char* xq;                      // local input rows when not in the window
  int2* hist;                          // per call % 4: {slot, rows}
  unsigned* epoch;                     // per-CTA call counter (replicated, never reset)
  float gs[FP4AR_MAX_RANKS];
  int rows, me, k, nstages, nslot, cur, inwin;
  size_t slot_bytes;
};

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mb_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mb_expect(uint32_t b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(tx) : "memory"); }
__device__ __forceinline__ void mb_wait(uint32_t b, uint32_t par) {
  asm volatile("{\n.reg .pred P;\nW%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra W%=;\n}" :: "r"(b), "r"(par) : "memory");
}
__device__ __forceinline__ void tma_pull(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ uint32_t ld_vol(const void* a) { uint32_t v; asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v; }
__device__ __forceinline__ bool sent_val(uint32_t w) { return (w & ~((w & 0x77777777u) + 0x77777777u) & 0x88888888u) != 0; }
__device__ __forceinline__ bool sent_sf(uint32_t w) { return (w & 0x80808080u) != 0; }  // valid scales are <= 0x7E
__device__ __forceinline__ void dec8(float* acc, uint32_t w, float scl) {
  __half2 h[4]; e2m1x8_dec(w, h);
#pragma unroll
  for (int i = 0; i < 4; i++) { float2 t = __half22float2(h[i]); acc[2 * i] = __fmaf_rn(t.x, scl, acc[2 * i]); acc[2 * i + 1] = __fmaf_rn(t.y, scl, acc[2 * i + 1]); }
}
__device__ __forceinline__ void st_bf16x8(__nv_bfloat16* o, const float* a) {
  uint4 u; __nv_bfloat162* b = (__nv_bfloat162*)&u;
#pragma unroll
  for (int i = 0; i < 4; i++) b[i] = __floats2bfloat162_rn(a[2 * i], a[2 * i + 1]);
  *(uint4*)o = u;
}
__host__ __device__ __forceinline__ int srcb(int k) { return k * ROWQ; }  // smem bytes per source

template <int NR, int MODE>
__global__ void __launch_bounds__(NTHR, 1) pq_kernel(KP p) {
  extern __shared__ __align__(1024) char smem[];
  uint64_t* bars = (uint64_t*)smem;
  const uint32_t full0 = sa(bars), empty0 = sa(bars + 8), rtry0 = sa(bars + 16);
  uint32_t* s_bad = (uint32_t*)(smem + 512);     // [2]
  uint32_t* s_first = (uint32_t*)(smem + 576);   // [2][8]
  unsigned* s_ep = (unsigned*)(smem + 704);
  char* ring = smem + HDR;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int c = blockIdx.x, G = gridDim.x, k = p.k, R = p.rows;
  const int nmy = c < p.nstages ? (p.nstages - c + G - 1) / G : 0;
  const size_t slot_b = (size_t)NR * srcb(k);
  if (tid == 0) {
    for (int i = 0; i < MAXSLOT; i++) { mb_init(full0 + 8 * i, 1); mb_init(empty0 + 8 * i, NDEC / 32); }
    mb_init(rtry0, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (tid == 0) *s_ep = p.epoch[c] + 1;
  __syncthreads();
  const unsigned ep = *s_ep;
  // window mode: slot from the producer's address (host sequence baked into graphs); local mode: device call count
  const int cur = p.inwin ? p.cur : (int)(ep % NSQ);
  if (tid == 0) for (int i = c; i < 1024; i += G) p.epoch[i] = ep;
  if (tid == 0 && c == 0) p.hist[ep & 3] = make_int2(cur, R);
  asm volatile("griddepcontrol.launch_dependents;");
  const size_t cur_off = (size_t)cur * p.slot_bytes;
  const bool peers = MODE != FP4AR_COMPUTE;

  if (warp < 4) {
    if (warp == 0) {
      // ---- fetch: lane s pulls source s; own rows from the input buffer, peers' slots unconditionally (checked later) ----
      for (int i = 0; i < nmy; i++) {
        const int slot = i % p.nslot, r0 = (c + G * i) * k, n = min(k, R - r0);
        if (lane == 0) { mb_wait(empty0 + 8 * slot, ((i / p.nslot) & 1) ^ 1); mb_expect(full0 + 8 * slot, NR * n * ROWQ); }
        __syncwarp();
        if (lane < NR) {
          const int src = peers ? lane : p.me;
          const char* b = src == p.me ? p.xq : p.win[src] + cur_off;
          tma_pull(sa(ring + slot * slot_b + (size_t)lane * srcb(k)), b + (size_t)r0 * ROWQ, n * ROWQ, full0 + 8 * slot);
        }
        __syncwarp();
      }
    }
    return;
  }
  if (warp < 8) {
    // ---- aux: (local mode) copy own rows into the window slot for peers; then refill the previous slot ----
    const int at = tid - 128;
    char* const self = p.win[p.me];
    if (!p.inwin) {
      const uint4* src = (const uint4*)p.xq; uint4* dst = (uint4*)(self + cur_off);
      for (int i = 0; i < nmy; i++) {
        const int r0 = (c + G * i) * k, n = min(k, R - r0);
        const size_t g0 = (size_t)r0 * (ROWQ / 16), ng = (size_t)n * (ROWQ / 16);
        for (size_t g = at; g < ng; g += NAUX) dst[g0 + g] = __ldg(src + g0 + g);
      }
    }
    // refill the slot of call ep-2: our call ep-1 validated every peer's ep-1 data, so all peers finished reading it
    const int2 h = p.hist[(ep + 2) & 3];
    const int prev = h.x, Rp = h.y == 0 || prev == cur ? 0 : h.y;
    uint4* pb = (uint4*)(self + (size_t)prev * p.slot_bytes);
    const size_t ng = (size_t)Rp * (ROWQ / 16), per = (ng + G - 1) / G, lo = min(ng, (size_t)c * per), hi = min(ng, lo + per);
    const uint4 f = make_uint4(FILL, FILL, FILL, FILL);
    for (size_t g = lo + at; g < hi; g += NAUX) pb[g] = f;
    bar_sync(3, NAUX);
    if (at == 0) asm volatile("fence.acq_rel.sys;" ::: "memory");  // refill performed before the slot's next producer writes
    return;
  }
  // ---- decode-reduce ----
  const int dt = tid - DEC0;
  uint32_t chk = 0, rp = 0;
  for (int i = 0; i < nmy; i++) {
    const int slot = i % p.nslot, r0 = (c + G * i) * k, n = min(k, R - r0);
    const char* sb = ring + slot * slot_b;
    mb_wait(full0 + 8 * slot, (i / p.nslot) & 1);
    if (peers) {
      for (;;) {
        const int cb = chk & 1;
        if (dt < 8) s_first[cb * 8 + dt] = ~0u;
        if (dt == 0) s_bad[cb] = 0;
        bar_sync(2, NDEC);
#pragma unroll
        for (int s = 0; s < NR; s++) {
          if (s == p.me) continue;
          const char* b = sb + (size_t)s * srcb(k);
          uint32_t off = ~0u;
          for (int rr = 0; rr < n && off == ~0u; rr++) {
            const uint32_t* w = (const uint32_t*)(b + rr * ROWQ);
#pragma unroll
            for (int u = 0; u < 3; u++) if (sent_val(w[u * 256 + dt])) { off = min(off, (uint32_t)(rr * ROWQ + 4 * (u * 256 + dt))); }
            if (dt < 96 && sent_sf(w[768 + dt])) off = min(off, (uint32_t)(rr * ROWQ + VB + 4 * dt));
          }
          if (off != ~0u) { atomicOr(&s_bad[cb], 1u << s); atomicMin(&s_first[cb * 8 + s], off); }
        }
        bar_sync(2, NDEC);
        const uint32_t bad = s_bad[cb];
        chk++;
        if (!bad) break;
        if (dt == 0) {
          uint32_t tot = 0;
#pragma unroll
          for (int s = 0; s < NR; s++) {
            if (!(bad >> s & 1)) continue;
            const uint32_t off = s_first[cb * 8 + s];
            const char* g = p.win[s] + cur_off + (size_t)r0 * ROWQ + off;
            const bool isv = off % ROWQ < VB; unsigned ns = 64;
            for (;;) {  // wait for the first missing word before re-pulling
              const uint32_t v = ld_vol(g);
              if (!(isv ? sent_val(v) : sent_sf(v))) break;
              __nanosleep(ns); ns = ns < 1024 ? 2 * ns : ns;
            }
            tot += n * ROWQ;
          }
          mb_expect(rtry0, tot);
#pragma unroll
          for (int s = 0; s < NR; s++)
            if (bad >> s & 1) tma_pull(sa(sb + (size_t)s * srcb(k)), p.win[s] + cur_off + (size_t)r0 * ROWQ, n * ROWQ, rtry0);
        }
        mb_wait(rtry0, rp); rp ^= 1;
      }
    }
    if (MODE != FP4AR_XPORT) {
      for (int rr = 0; rr < n; rr++) {
#pragma unroll
        for (int u = 0; u < 3; u++) {
          const int ch = u * 256 + dt;
          float acc[8];
#pragma unroll
          for (int e = 0; e < 8; e++) acc[e] = 0.f;
#pragma unroll
          for (int s = 0; s < NR; s++) {
            const char* b = sb + (size_t)s * srcb(k) + rr * ROWQ;
            dec8(acc, ((const uint32_t*)b)[ch], __fmul_rn(e4m3f(((const uint8_t*)b)[VB + ch / 2]), p.gs[peers ? s : p.me]));
          }
          st_bf16x8(p.out + (size_t)(r0 + rr) * H + 8 * ch, acc);
        }
      }
    }
    __syncwarp();
    if (lane == 0) mb_arrive(empty0 + 8 * slot);
  }
}

__global__ void fill_k(uint4* w, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) w[i] = make_uint4(FILL, FILL, FILL, FILL);
}

struct Priv { int2* hist; unsigned* epoch; int max_rows; size_t slot_bytes; long long calls; };
static size_t slot_bytes_of(int max_rows) { return ((size_t)max_rows * ROWQ + 4095) / 4096 * 4096; }
static size_t wbytes(int max_rows, int, int) { return NSQ * slot_bytes_of(max_rows) + 4096; }

template <int NR> static const void* pickm(int mode) {
  return mode == FP4AR_XPORT ? (const void*)pq_kernel<NR, FP4AR_XPORT> : mode == FP4AR_COMPUTE ? (const void*)pq_kernel<NR, FP4AR_COMPUTE> : (const void*)pq_kernel<NR, FP4AR_FUSED>;
}
static const void* pick(int nr, int mode) { return nr == 2 ? pickm<2>(mode) : nr == 4 ? pickm<4>(mode) : nr == 8 ? pickm<8>(mode) : nullptr; }

static int init(Fp4ArCtx* ctx, int max_rows, int hidden) {
  if (hidden != H || (ctx->nranks != 2 && ctx->nranks != 4 && ctx->nranks != 8)) return -1;
  Priv* pv = new Priv; pv->max_rows = max_rows; pv->slot_bytes = slot_bytes_of(max_rows); pv->calls = 0;
  if (NSQ * pv->slot_bytes > ctx->win_bytes) { delete pv; return -2; }
  cudaMalloc(&pv->hist, 4 * sizeof(int2)); cudaMemset(pv->hist, 0, 4 * sizeof(int2));
  cudaMalloc(&pv->epoch, 4096); cudaMemset(pv->epoch, 0, 4096);
  fill_k<<<1024, 256>>>((uint4*)ctx->peer[ctx->rank], NSQ * pv->slot_bytes / 16);
  for (int m = 0; m < 3; m++) cudaFuncSetAttribute(pick(ctx->nranks, m), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  float* d; cudaMalloc(&d, 4); cudaMemset(d, 0, 4);
  ncclAllReduce(d, d, 1, ncclFloat, ncclSum, ctx->comm, 0);
  cudaDeviceSynchronize(); cudaFree(d);
  ctx->priv = pv;
  return cudaGetLastError() == cudaSuccess ? 0 : -4;
}
// Slots rotate mod NSQ per call; captured graphs must hold a multiple of NSQ calls (harness: K = 10).
static uint8_t* input_slot(Fp4ArCtx* ctx, int) {
  Priv* pv = (Priv*)ctx->priv;
  return (uint8_t*)ctx->peer[ctx->rank] + (size_t)(pv->calls++ % NSQ) * pv->slot_bytes;
}
// tune[0] ring slots, tune[1] rows/stage (default 4 TP2, 2 TP4+), tune[2] grid, tune[3] 0x100 = no PDL
static int launch(Fp4ArCtx* ctx, const Fp4ArArgs* a, cudaStream_t s) {
  Priv* pv = (Priv*)ctx->priv; const int NR = ctx->nranks;
  if (!a->xq || a->algo != FP4AR_ONESHOT || a->hidden != H || a->rows > pv->max_rows) return -1;
  const int G0 = a->tune[2] > 0 ? std::min(a->tune[2], 1024) : ctx->sm_count;
  const int k = a->tune[1] > 0 ? std::min(a->tune[1], 8) : (NR == 2 ? 4 : 2);
  const int nst = (a->rows + k - 1) / k, G = std::min(G0, nst);
  const size_t slot = (size_t)NR * srcb(k);
  int nslot = a->tune[0] > 0 ? a->tune[0] : std::max(2, std::min(MAXSLOT, (int)(RING_DEF / slot)));
  nslot = std::min<int>(std::min(nslot, MAXSLOT), (SMEM_MAX - HDR) / slot);
  if (nslot < 2) return -4;
  KP p{};
  p.out = a->out; for (int r = 0; r < NR; r++) p.win[r] = ctx->peer[r];
  p.xq = (const char*)a->xq; p.hist = pv->hist; p.epoch = pv->epoch; for (int r = 0; r < NR; r++) p.gs[r] = a->gs[r]; p.rows = a->rows; p.me = ctx->rank; p.k = k; p.nstages = nst; p.nslot = nslot;
  p.inwin = a->xq_in_window; p.slot_bytes = pv->slot_bytes;
  if (p.inwin) {
    const size_t off = (const char*)a->xq - ctx->peer[ctx->rank];
    if (off % pv->slot_bytes || off / pv->slot_bytes >= NSQ) return -6;
    p.cur = (int)(off / pv->slot_bytes);
  } else {
    p.cur = 0;  // local mode: the kernel copies own rows into window slot (call count % NSQ)
  }
  cudaLaunchConfig_t cfg{}; cudaLaunchAttribute at[1];
  cfg.gridDim = dim3(G); cfg.blockDim = dim3(NTHR); cfg.dynamicSmemBytes = HDR + nslot * slot; cfg.stream = s;
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = (a->tune[3] & 0x100) ? 0 : 1;
  void* args[] = {&p};
  return cudaLaunchKernelExC(&cfg, pick(NR, a->mode), args) == cudaSuccess ? 0 : -5;
}
static void fini(Fp4ArCtx* ctx) { Priv* pv = (Priv*)ctx->priv; if (!pv) return; cudaFree(pv->hist); cudaFree(pv->epoch); delete pv; ctx->priv = nullptr; }
}  // namespace

#define PQ_STR2(x) #x
#define PQ_STR(x) PQ_STR2(x)
static const Fp4ArVariant kVariantP4q = {PQ_STR(FP4AR_NAME), false, wbytes, init, launch, fini, input_slot};
const Fp4ArVariant* FP4AR_REGISTER() { return &kVariantP4q; }
