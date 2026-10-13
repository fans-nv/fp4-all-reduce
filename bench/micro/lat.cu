// lat: ranks 0<->1 ping-pong round trips (flag and payload protocols). One-way latency = RTT/2.
#include "micro_common.cuh"

enum LM { FLAG_RELACQ = 0, FLAG_WEAK = 1, FLAG_RELAXED = 2, LAMPORT_V4 = 3, TMAST_LAMPORT = 4, PULL_TMA = 5 };
static const char* LNAME[] = {"flag_relacq", "flag_weak", "flag_relaxed", "lamport_v4", "tmast_lamport", "pull_tma"};
constexpr size_t OFF_FA = 0, OFF_FB = 128, OFF_FC = 256, OFF_IN = 1 << 16;
constexpr unsigned long long TIMEOUT_NS = 2000000000ull;

struct LP {
  char* self; char* peer; int initiator, iters, S; uint32_t gbase; size_t outOff;
  unsigned long long* rtt; unsigned* err;
};

#define SPIN(cond) do { unsigned long long t_ = 0; int c_ = 0; while (!(cond)) { if (((++c_) & 255) == 0) { unsigned long long n_ = gtimer(); if (!t_) t_ = n_; else if (n_ - t_ > TIMEOUT_NS) { atomicAdd(p.err, 1u); break; } } } } while (0)

template <int M>
__global__ void __launch_bounds__(1024) lat_kernel(LP p) {
  extern __shared__ __align__(1024) char sm[];
  __shared__ __align__(8) uint64_t bar;
  const int tid = threadIdx.x, T = blockDim.x, nslot = p.S / 16;
  if (M == PULL_TMA && tid == 0) { mbar_init(&bar, 1); fence_mbar_init(); }
  __syncthreads();
  auto send = [&](uint32_t g) {
    if (M == FLAG_RELACQ) { if (tid == 0) st_rel_sys(p.peer + (p.initiator ? OFF_FA : OFF_FB), g); }
    else if (M == FLAG_WEAK) { if (tid == 0) st_u32(p.peer + (p.initiator ? OFF_FA : OFF_FB), g); }
    else if (M == FLAG_RELAXED) { if (tid == 0) st_rlx_sys(p.peer + (p.initiator ? OFF_FA : OFF_FB), g); }
    else if (M == LAMPORT_V4) { for (int j = tid; j < nslot; j += T) st_v4(p.peer + OFF_IN + (size_t)j * 16, make_uint4(g, g, g, g)); }
    else if (M == TMAST_LAMPORT) {
      for (int j = tid; j < nslot; j += T) *(uint4*)(sm + (size_t)j * 16) = make_uint4(g, g, g, g);
      fence_proxy_async_smem(); __syncthreads();
      if (tid == 0) { tma_s2g(p.peer + OFF_IN, sm, p.S); tma_commit(); tma_wait_read<0>(); }
      __syncthreads();
    } else {  // PULL_TMA: local outbox write, one sys-scope release covers the CTA's stores
      for (int j = tid; j < nslot; j += T) st_v4(p.self + p.outOff + (size_t)j * 16, make_uint4(g, g, g, g));
      __syncthreads();
      if (tid == 0) st_rel_sys(p.peer + OFF_FC, g);
    }
  };
  auto recv = [&](uint32_t g, int k) {
    if (M == FLAG_RELACQ) { if (tid == 0) SPIN(ld_acq_sys(p.self + (p.initiator ? OFF_FB : OFF_FA)) == g); }
    else if (M == FLAG_WEAK) { if (tid == 0) SPIN(ldv_u32(p.self + (p.initiator ? OFF_FB : OFF_FA)) == g); }
    else if (M == FLAG_RELAXED) { if (tid == 0) SPIN(ld_rlx_sys(p.self + (p.initiator ? OFF_FB : OFF_FA)) == g); }
    else if (M == LAMPORT_V4 || M == TMAST_LAMPORT) {
      for (int j = tid; j < nslot; j += T) { const char* a = p.self + OFF_IN + (size_t)j * 16; uint4 v; SPIN((v = ldv_v4(a), v.x == g && v.y == g && v.z == g && v.w == g)); }
    } else {
      if (tid == 0) {
        SPIN(ld_acq_sys(p.self + OFF_FC) == g);
        fence_proxy_async_global();
        mbar_expect_tx(&bar, p.S); tma_g2s(sm, p.peer + p.outOff, p.S, &bar); mbar_wait(&bar, k & 1);
      }
      __syncthreads();
      for (int j = tid; j < nslot; j += T) { uint4 v = *(uint4*)(sm + (size_t)j * 16); if (v.x != g || v.w != g) atomicAdd(p.err + 1, 1u); }
    }
    __syncthreads();
  };
  for (int i = 0; i < p.iters; i++) {
    uint32_t g = p.gbase + i + 1;
    if (p.initiator) {
      unsigned long long t0 = gtimer();
      send(g); recv(g, i);
      if (tid == 0) p.rtt[i] = gtimer() - t0;
    } else { recv(g, i); send(g); }
  }
}

static const void* kfn(int m) {
  switch (m) {
    case FLAG_RELACQ: return (const void*)lat_kernel<FLAG_RELACQ>;
    case FLAG_WEAK: return (const void*)lat_kernel<FLAG_WEAK>;
    case FLAG_RELAXED: return (const void*)lat_kernel<FLAG_RELAXED>;
    case LAMPORT_V4: return (const void*)lat_kernel<LAMPORT_V4>;
    case TMAST_LAMPORT: return (const void*)lat_kernel<TMAST_LAMPORT>;
    default: return (const void*)lat_kernel<PULL_TMA>;
  }
}

int main(int argc, char** argv) {
  Args A(argc, argv);
  const int n = (int)A.i("n", 2), iters = (int)A.i("iters", 2000), reps = (int)A.i("reps", 3);
  const int peer = (int)A.i("peer", 1);
  auto methods = A.ls("method", "flag_relacq,flag_weak,flag_relaxed,lamport_v4,tmast_lamport,pull_tma");
  auto sizes = A.li("size", "16,256,1K,4K,16K,64K");
  const std::string tag = A.s("tag", "");
  size_t maxS = *std::max_element(sizes.begin(), sizes.end());
  World W(n);
  run_ranks(n, [&](int rank) {
    Rank R; rank_setup(W, R, rank, OFF_IN + 2 * maxS + 4096, false);
    const bool on = rank == 0 || rank == peer;
    unsigned long long* rtt; unsigned* err;
    CK(cudaMalloc(&rtt, iters * 8)); CK(cudaMalloc(&err, 8));
    for (int m = 0; m < 6; m++) CK(cudaFuncSetAttribute(kfn(m), cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem_max(R)));
    uint32_t gbase = 0;
    for (auto& mname : methods) {
      int m = (int)(std::find(LNAME, LNAME + 6, mname) - LNAME);
      if (m >= 6) continue;
      for (long long S : sizes) {
        if (m <= FLAG_RELAXED && S != sizes[0]) continue;  // flags carry no payload
        int T = (int)std::min<long long>(1024, std::max<long long>(32, S / 16));
        int smem = (m == TMAST_LAMPORT || m == PULL_TMA) ? (int)S : 0;
        for (int rep = 0; rep < reps; rep++) {
          LP p{}; p.self = R.base; p.peer = R.peer[rank == 0 ? peer : 0]; p.initiator = rank == 0; p.iters = iters; p.S = (int)S;
          p.gbase = gbase; p.outOff = OFF_IN + maxS; p.rtt = rtt; p.err = err;
          gbase += iters + 1;
          CK(cudaMemsetAsync(err, 0, 8, R.st));
          align(W, R);
          if (on) { void* args[] = {&p}; CK(cudaLaunchKernel(kfn(m), dim3(1), dim3(T), args, smem, R.st)); }
          CK(cudaStreamSynchronize(R.st)); CK(cudaGetLastError());
          if (rank == 0) {
            std::vector<unsigned long long> h(iters); unsigned e[2];
            CK(cudaMemcpy(h.data(), rtt, iters * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(e, err, 8, cudaMemcpyDeviceToHost));
            std::vector<double> v; for (int i = iters / 10; i < iters; i++) v.push_back((double)h[i]);  // drop first 10%
            Json().kv("prog", "lat").kv("tag", tag).kv("method", mname).kv("n", n).kv("peer", peer).kv("size", S).kv("threads", T).kv("rep", rep)
              .kv("rtt_med_ns", pct(v, 0.5)).kv("rtt_p10_ns", pct(v, 0.1)).kv("rtt_p90_ns", pct(v, 0.9)).kv("oneway_med_ns", pct(v, 0.5) / 2)
              .kv("timeouts", (long long)e[0]).kv("stale", (long long)e[1]).print();
          }
          W.bar.wait();
        }
      }
    }
    rank_teardown(W, R);
  });
  return 0;
}
