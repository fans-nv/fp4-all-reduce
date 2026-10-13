// validity: Rubin TMA `cp.async.bulk...mbarrier::report::validity` probe (PTX ISA 9.4, sm_107a only).
// Part sem: which sentinel placements set isReportSeen, and how primary/conditional mbarrier phases behave on re-issue.
// Part xgpu: Lamport-in-TMA stress (remote push into a sentinel-filled local buffer, TMA validity load + retry).
// PTX ISA 9.4 (cp_async_bulk report_mechanism, mbarrier_test_wait
// phase_primary, mbarrier_init layout_v1) and src/trtllm/dev/CutlassPipeline.h PipelinePayloadLamport.
#include "micro_common.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1070
#define RUBIN 1
#else
#define RUBIN 0
#endif

static const char* VNAME[] = {"per_16bytes_80000000", "per_16bytes_8000", "per_16bytes_80", "per_16bytes_8", "per_element_ff"};
static const int VBITS[] = {32, 16, 8, 4, 8};
static const uint32_t VSENT[] = {0x80000000u, 0x8000u, 0x80u, 0x8u, 0xFFu};

#define VLD(dsp, mech) asm volatile("cp.async.bulk." dsp ".global.mbarrier::complete_tx::bytes.mbarrier::report::validity::" mech " [%0], [%1], %2, [%3];" \
                                    :: "r"(d), "l"(src), "r"(bytes), "r"(b) : "memory")

// V: 0..4 as VNAME; CL: shared::cluster destination form.
template <int V, int CL>
__device__ __forceinline__ void vload(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
#if RUBIN
  uint32_t d = smem_u32(dst), b = smem_u32(bar);
  if (CL == 0) {
    if (V == 0) VLD("shared::cta", "per_16bytes::80000000"); else if (V == 1) VLD("shared::cta", "per_16bytes::8000");
    else if (V == 2) VLD("shared::cta", "per_16bytes::80"); else if (V == 3) VLD("shared::cta", "per_16bytes::8");
    else VLD("shared::cta", "per_element::ff");
  } else {
    if (V == 0) VLD("shared::cluster", "per_16bytes::80000000"); else if (V == 1) VLD("shared::cluster", "per_16bytes::8000");
    else if (V == 2) VLD("shared::cluster", "per_16bytes::80"); else if (V == 3) VLD("shared::cluster", "per_16bytes::8");
    else VLD("shared::cluster", "per_element::ff");
  }
#endif
}

__device__ __forceinline__ void vbar_init(uint64_t* b, int layout) {
#if RUBIN
  if (layout == 1) asm volatile("mbarrier.init.layout::v1.shared.b64 [%0], 1;" :: "r"(smem_u32(b)) : "memory");
  else mbar_init(b, 1);
#endif
}
// Returns bit0 = primary phase complete, bit1 = isReportSeen (some 16 B / element held the sentinel).
__device__ __forceinline__ uint32_t vtest_primary(uint64_t* b, uint32_t parity) {
  uint32_t c = 0, r = 0;
#if RUBIN
  asm volatile("{\n\t.reg .pred p, q;\n\tmbarrier.test_wait.parity.phase_type::primary.acquire.cta.shared::cta.b64 p|q, [%2], %3;\n\t"
               "selp.b32 %0, 1, 0, p;\n\tselp.b32 %1, 1, 0, q;\n\t}" : "=r"(c), "=r"(r) : "r"(smem_u32(b)), "r"(parity) : "memory");
#endif
  return c | (r << 1);
}
// Conditional phase completes only when a load of that phase was valid (PipelinePayloadLamport consumer_try_wait).
__device__ __forceinline__ uint32_t vtry_conditional(uint64_t* b, uint32_t parity) {
  uint32_t c = 0;
#if RUBIN
  asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.phase_type::conditional.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
               "selp.b32 %0, 1, 0, p;\n\t}" : "=r"(c) : "r"(smem_u32(b)), "r"(parity) : "memory");
#endif
  return c;
}

constexpr unsigned long long SPIN_NS = 20000000ull;

// out[0..9]: rubin, done1, report1, cond1 (bitmask), ns1, done2, report2, cond2, ns2, smem_matches_src1
template <int V, int CL>
__global__ void sem_kernel(const char* src1, const char* src2, uint32_t bytes, int layout, unsigned long long* out) {
  __shared__ __align__(128) char buf[4096];
  __shared__ __align__(8) uint64_t bar;
  if (threadIdx.x != 0) return;
  out[0] = RUBIN;
  if (!RUBIN) return;
  vbar_init(&bar, layout); fence_mbar_init();
  for (int k = 0; k < 2; k++) {
    unsigned long long t0 = gtimer(), t;
    mbar_expect_tx(&bar, bytes);
    if (k == 0) vload<V, CL>(buf, src1, bytes, &bar); else vload<V, CL>(buf, src2, bytes, &bar);
    uint32_t s;
    do { s = vtest_primary(&bar, k); t = gtimer(); } while (!(s & 1) && t - t0 < SPIN_NS);
    uint32_t cond = 0;  // bit p = try_wait.parity(p) on the conditional phase returned true (fresh barrier: 0b10)
    for (int q = 0; q < 2; q++) { uint32_t c = 0; for (int i = 0; i < 16 && !c; i++) c = vtry_conditional(&bar, q); cond |= c << q; }
    out[1 + 4 * k] = s & 1; out[2 + 4 * k] = (s >> 1) & 1; out[3 + 4 * k] = cond; out[4 + 4 * k] = t - t0;
    if (k == 0) { uint32_t m = 1; for (uint32_t i = 0; i < bytes; i++) m &= buf[i] == src1[i]; out[9] = m; }
  }
}

// ---- cross-GPU stress ----
__host__ __device__ __forceinline__ uint32_t spread24(uint32_t x) {  // 24 bits -> low 3 bits of 8 nibbles (never 0x8)
  uint32_t w = 0;
  for (int i = 0; i < 8; i++) w |= ((x >> (3 * i)) & 7u) << (4 * i);
  return w;
}
__device__ __forceinline__ uint32_t enc(int V, uint32_t it, uint32_t widx) {
  uint32_t w = spread24((it * 4099u + widx) & 0xFFFFFF);
  return V == 4 ? (w & 0x7F7F7F7Fu) : w;  // ff variant: bytes < 0x80
}

struct XP {
  char* rbuf;        // consumer's receive buffer (peer ptr on producer, local on consumer)
  char* myflag; char* peerflag;
  int nchunk, chunk, iters, producer, pushTma, layout;
  unsigned long long* out;  // [0]=retries [1]=false_valid [2]=timeouts [3]=ns_total [4]=loads
};

template <int V>
__global__ void __launch_bounds__(256) xgpu_kernel(XP p) {
  __shared__ __align__(128) char sbuf[8][4096];
  __shared__ __align__(8) uint64_t bar;
  __shared__ uint32_t s_bad;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, T = blockDim.x, nw = T >> 5;
  const uint32_t SENT = V == 4 ? 0xFFFFFFFFu : 0x88888888u;
  const size_t total = (size_t)p.nchunk * p.chunk;
  if (!RUBIN) { if (tid == 0) p.out[2] = ~0ull; return; }
  unsigned long long retries = 0, fv = 0, to = 0, loads = 0, t0 = gtimer();
  if (p.producer) {
    for (uint32_t it = 1; it <= (uint32_t)p.iters; it++) {
      if (tid == 0) { unsigned long long s = gtimer(); while (ld_acq_sys(p.myflag) != it) if (gtimer() - s > 2000000000ull) { to++; break; } }
      __syncthreads();
      __nanosleep((it * 2654435761u >> 22) & 2047);  // jitter the arrival vs the consumer's first load
      if (!p.pushTma) {
        for (size_t o = (size_t)tid * 16; o < total; o += (size_t)T * 16) {
          uint32_t wi = (uint32_t)(o / 4);
          st_v4(p.rbuf + o, make_uint4(enc(V, it, wi), enc(V, it, wi + 1), enc(V, it, wi + 2), enc(V, it, wi + 3)));
        }
      } else {
        for (size_t base = (size_t)warp * 4096; base < total; base += (size_t)nw * 4096) {
          for (int i = lane * 4; i < 1024; i += 128) { uint32_t wi = (uint32_t)(base / 4) + i; *(uint4*)(sbuf[warp] + i * 4) = make_uint4(enc(V, it, wi), enc(V, it, wi + 1), enc(V, it, wi + 2), enc(V, it, wi + 3)); }
          fence_proxy_async_smem(); __syncwarp();
          if (lane == 0) { tma_s2g(p.rbuf + base, sbuf[warp], 4096); tma_commit(); tma_wait_read<0>(); }
          __syncwarp();
        }
        if (lane == 0) tma_wait_all<0>();
      }
      __syncthreads();
    }
  } else {
    if (tid == 0) { vbar_init(&bar, p.layout); fence_mbar_init(); }
    __syncthreads();
    uint32_t pphase = 0, cphase = 0;
    for (uint32_t it = 1; it <= (uint32_t)p.iters; it++) {
      for (int c = 0; c < p.nchunk; c++) {
        if (tid == 0) {
          const char* src = p.rbuf + (size_t)c * p.chunk;
          unsigned long long s0 = gtimer();
          for (;;) {
            mbar_expect_tx(&bar, p.chunk); vload<V, 0>(sbuf[0], src, p.chunk, &bar); loads++;
            uint32_t st; do { st = vtest_primary(&bar, pphase); } while (!(st & 1));
            pphase ^= 1;
            if (!(st & 2)) break;
            retries++;
            if (gtimer() - s0 > 2000000000ull) { to++; break; }
          }
          unsigned long long s1 = gtimer();
          while (!vtry_conditional(&bar, cphase)) if (gtimer() - s1 > 100000000ull) { to++; break; }
          cphase ^= 1;
          s_bad = 0;
        }
        __syncthreads();
        uint32_t bad = 0;
        for (int i = tid; i < p.chunk / 4; i += T) {
          uint32_t wi = (uint32_t)(((size_t)c * p.chunk) / 4) + i;
          bad |= ((const uint32_t*)sbuf[0])[i] != enc(V, it, wi);
        }
        if (bad) atomicOr(&s_bad, 1u);
        __syncthreads();
        if (tid == 0) fv += s_bad;
      }
      for (size_t o = (size_t)tid * 16; o < total; o += (size_t)T * 16) st_v4(p.rbuf + o, make_uint4(SENT, SENT, SENT, SENT));  // re-arm
      fence_proxy_async_global();
      __syncthreads();
      if (tid == 0) st_rel_sys(p.peerflag, it + 1);
    }
  }
  if (tid == 0) { atomicAdd(&p.out[0], retries); atomicAdd(&p.out[1], fv); atomicAdd(&p.out[2], to); p.out[3] = gtimer() - t0; atomicAdd(&p.out[4], loads); }
}

template <int V, int CL> static const void* semk() { return (const void*)sem_kernel<V, CL>; }
static const void* semfn(int v, int cl) {
  static const void* t[5][2] = {{semk<0, 0>(), semk<0, 1>()}, {semk<1, 0>(), semk<1, 1>()}, {semk<2, 0>(), semk<2, 1>()}, {semk<3, 0>(), semk<3, 1>()}, {semk<4, 0>(), semk<4, 1>()}};
  return t[v][cl];
}

// Writes element value `val` (VBITS[v] wide) at element index e of byte buffer b (little-endian, nibble-packed for 4 bit).
static void put_elem(std::vector<uint8_t>& b, int v, size_t byteOff, int e, uint32_t val) {
  int bits = VBITS[v];
  if (bits == 4) { uint8_t& x = b[byteOff + e / 2]; x = (e & 1) ? (uint8_t)((x & 0x0F) | (val << 4)) : (uint8_t)((x & 0xF0) | val); }
  else for (int k = 0; k < bits / 8; k++) b[byteOff + (size_t)e * (bits / 8) + k] = (uint8_t)(val >> (8 * k));
}

int main(int argc, char** argv) {
  Args A(argc, argv);
  const int n = (int)A.i("n", 2), layout = (int)A.i("layout", 1);
  const std::string part = A.s("part", "sem,xgpu"), tag = A.s("tag", "");
  auto vs = A.li("variant", "0,1,2,3,4");
  auto forms = A.li("cluster", "0,1");
  auto srcs = A.ls("src", "local,peer");
  const int iters = (int)A.i("iters", 20000), nchunk = (int)A.i("nchunk", 16), chunk = (int)std::min<long long>(4096, A.i("chunk", 4096));
  auto xvs = A.li("xvariant", "3,4");
  auto pushes = A.li("push_tma", "0,1");
  World W(n);
  run_ranks(n, [&](int rank) {
    Rank R; rank_setup(W, R, rank, 4 << 20, false);
    unsigned long long* out; CK(cudaMalloc(&out, 128));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, rank));
    if (rank == 0) Json().kv("prog", "validity").kv("event", "device").kv("cc", pr.major * 10 + pr.minor).kv("layout", layout).print();
    if (part.find("sem") != std::string::npos) {
      // rank 1 hosts the patterns for the peer-source case; rank 0 runs every probe
      const size_t B = 1024, OFF1 = 1 << 20, OFF2 = OFF1 + 4096;
      const char* cases[] = {"all_valid", "one_elem_first", "one_elem_last", "chunk_all", "all_sentinel", "near_miss", "nibble_hi", "nibble_lo"};
      for (long long v : vs) for (int ci = 0; ci < 8; ci++) {
        if (ci >= 6 && v != 3) continue;
        std::vector<uint8_t> h(B, 0x11), ok(B, 0x11);
        const size_t CH = 5 * 16; int epc = 128 / VBITS[v];
        if (ci == 1) put_elem(h, (int)v, CH, 0, VSENT[v]);
        if (ci == 2) put_elem(h, (int)v, CH, epc - 1, VSENT[v]);
        if (ci == 3) for (int e = 0; e < epc; e++) put_elem(h, (int)v, CH, e, VSENT[v]);
        if (ci == 4) for (size_t c = 0; c < B; c += 16) for (int e = 0; e < epc; e++) put_elem(h, (int)v, c, e, VSENT[v]);
        if (ci == 5) put_elem(h, (int)v, CH, 0, VSENT[v] ^ 1u);
        if (ci == 6) h[CH] = 0x81;
        if (ci == 7) h[CH] = 0x18;
        W.bar.wait();
        if (rank == 1 || n == 1) { CK(cudaMemcpy(R.base + OFF1, h.data(), B, cudaMemcpyHostToDevice)); CK(cudaMemcpy(R.base + OFF2, ok.data(), B, cudaMemcpyHostToDevice)); }
        if (rank == 0) { CK(cudaMemcpy(R.base + OFF1, h.data(), B, cudaMemcpyHostToDevice)); CK(cudaMemcpy(R.base + OFF2, ok.data(), B, cudaMemcpyHostToDevice)); }
        CK(cudaDeviceSynchronize()); W.bar.wait();
        if (rank == 0) for (auto& sname : srcs) for (long long cl : forms) {
          if (sname == "peer" && n < 2) continue;
          char* base = sname == "peer" ? R.peer[1] : R.base;
          CK(cudaMemset(out, 0, 128));
          void* fn = (void*)semfn((int)v, (int)cl);
          const char *s1 = base + OFF1, *s2 = base + OFF2; uint32_t bb = B; int lay = layout;
          void* args[] = {&s1, &s2, &bb, &lay, &out};
          cudaError_t e = cudaLaunchKernel(fn, dim3(1), dim3(32), args, 0, R.st);
          if (e == cudaSuccess) e = cudaStreamSynchronize(R.st);
          unsigned long long o[10] = {};
          if (e == cudaSuccess) CK(cudaMemcpy(o, out, 80, cudaMemcpyDeviceToHost));
          Json j; j.kv("prog", "validity").kv("part", "sem").kv("tag", tag).kv("variant", VNAME[v]).kv("case", cases[ci]).kv("src", sname)
            .kv("cluster_form", cl).kv("layout", layout).kv("cuda", cudaGetErrorString(e)).kv("rubin_path", o[0]);
          j.kv("load1_done", o[1]).kv("load1_report", o[2]).kv("load1_cond_phase", o[3]).kv("load1_ns", o[4]).kv("load1_smem_eq_src", o[9])
           .kv("reissue_valid_done", o[5]).kv("reissue_valid_report", o[6]).kv("reissue_valid_cond_phase", o[7]).kv("reissue_ns", o[8]).print();
          if (e != cudaSuccess) { fprintf(stderr, "sticky CUDA error, aborting sem part\n"); exit(3); }
        }
        W.bar.wait();
      }
    }
    if (part.find("xgpu") != std::string::npos && n >= 2) {
      for (long long v : xvs) for (long long pt : pushes) {
        const size_t RB = 2 << 20, FLAG = 0;
        XP p{}; p.nchunk = nchunk; p.chunk = chunk; p.iters = iters; p.pushTma = (int)pt; p.layout = layout;
        p.producer = rank == 0; p.out = out;
        p.rbuf = rank == 0 ? R.peer[1] + RB : R.base + RB;
        p.myflag = R.base + FLAG; p.peerflag = R.peer[rank == 0 ? 1 : 0] + FLAG;
        uint32_t sent = v == 4 ? 0xFFFFFFFFu : 0x88888888u;
        std::vector<uint32_t> fill((size_t)nchunk * chunk / 4, sent);
        if (rank == 1) CK(cudaMemcpy(R.base + RB, fill.data(), fill.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(R.base + FLAG, 0, 4)); CK(cudaMemset(out, 0, 128));
        if (rank == 0) { uint32_t one = 1; CK(cudaMemcpy(R.base + FLAG, &one, 4, cudaMemcpyHostToDevice)); }
        CK(cudaDeviceSynchronize());
        align(W, R);
        if (rank <= 1) {
          void* fn = v == 4 ? (void*)xgpu_kernel<4> : (void*)xgpu_kernel<3>;
          void* args[] = {&p};
          CK(cudaLaunchKernel(fn, dim3(1), dim3(256), args, 0, R.st));
        }
        CK(cudaStreamSynchronize(R.st)); CK(cudaGetLastError());
        unsigned long long o[5]; CK(cudaMemcpy(o, out, 40, cudaMemcpyDeviceToHost));
        W.vals[rank] = {(double)o[0], (double)o[1], (double)o[2], (double)o[3], (double)o[4]}; W.bar.wait();
        if (rank == 0) {
          auto& c = W.vals[1];
          Json().kv("prog", "validity").kv("part", "xgpu").kv("tag", tag).kv("variant", VNAME[v]).kv("push", pt ? "tma_st" : "st_v4").kv("layout", layout)
            .kv("iters", iters).kv("nchunk", nchunk).kv("chunk", chunk).kv("chunk_loads_valid", (long long)iters * nchunk)
            .kv("tma_loads", (unsigned long long)c[4]).kv("retries", (unsigned long long)c[0]).kv("false_valid", (unsigned long long)c[1])
            .kv("timeouts_consumer", (unsigned long long)c[2]).kv("timeouts_producer", (unsigned long long)W.vals[0][2])
            .kv("us_per_iter", c[3] / 1e3 / iters).print();
        }
        W.bar.wait();
      }
    }
    rank_teardown(W, R);
  });
  return 0;
}
