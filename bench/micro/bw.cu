// bw: NVLink peer bandwidth vs method, chunk, size, CTAs/SM and SM count. Emits one JSON line per config.
#include "micro_common.cuh"

enum Method { PUSH_ST = 0, TMA_ST = 1, TMA_LD = 2, TMA_LD_CL = 3, PULL_LD = 4, MC_ST = 5 };
static const char* MNAME[] = {"push_st", "tma_st", "tma_ld", "tma_ld_cluster", "pull_ld", "mc_st"};
constexpr int MAXW = 32, MAXSTG = 8;

struct BwP {
  char* tgt[MAXR];      // push/mc: destination bases; pull: source bases
  int npeer, chunk, stages, active, filter, nsm, inner;
  size_t bytes;         // per peer per pass
  unsigned* ticket;
  unsigned long long* sink;
};

template <int M>
__global__ void __launch_bounds__(1024) bw_kernel(BwP p) {
  extern __shared__ __align__(1024) char sm[];
  __shared__ __align__(8) uint64_t bars[MAXW][MAXSTG];
  __shared__ int s_id;
  __shared__ char* tgt[MAXR];  // smem copy avoids a local-memory param array on dynamic peer index
  if (threadIdx.x < MAXR) tgt[threadIdx.x] = p.tgt[threadIdx.x];
  if (threadIdx.x == 0) s_id = p.filter ? (smid() < (uint32_t)p.nsm ? (int)atomicAdd(p.ticket, 1u) : -1) : (int)blockIdx.x;
  __syncthreads();
  const int id = s_id;
  if (id < 0 || id >= p.active) return;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const long long gw = (long long)id * nw + warp, GW = (long long)p.active * nw;
  const size_t nch = p.bytes / p.chunk, items = nch * p.npeer;
  const int C = p.chunk;

  if (M == PUSH_ST || M == MC_ST) {
    for (int rep = 0; rep < p.inner; rep++)
      for (long long it = gw; it < (long long)items; it += GW) {
        char* d = tgt[it % p.npeer] + (size_t)(it / p.npeer) * C;
#pragma unroll 4
        for (int off = lane * 16; off < C; off += 512) {
          uint4 v = make_uint4((uint32_t)it, off, rep, 0x3f800000u);
          if (M == PUSH_ST) st_v4(d + off, v); else mc_st_v4(d + off, v);
        }
      }
  } else if (M == PULL_LD) {
    uint32_t acc = 0;
    for (int rep = 0; rep < p.inner; rep++)
      for (long long it = gw; it < (long long)items; it += GW) {
        const char* s = tgt[it % p.npeer] + (size_t)(it / p.npeer) * C;
        for (int off = lane * 16; off < C; off += 4 * 512) {
          uint4 a[4];
#pragma unroll
          for (int j = 0; j < 4; j++) if (off + j * 512 < C) a[j] = ld_rlx_sys_v4(s + off + j * 512);  // sys-scope: never served from L1 on repeat passes else a[j] = make_uint4(0, 0, 0, 0);
#pragma unroll
          for (int j = 0; j < 4; j++) acc ^= a[j].x ^ a[j].y ^ a[j].z ^ a[j].w;
        }
      }
    if (acc == 0x9e3779b9u) atomicAdd(p.sink, 1ull);  // keeps loads live
  } else if (M == TMA_ST) {
    // one constant source tile shared by all issuing warps: measures link rate, not smem refill
    for (int i = threadIdx.x * 16; i < C; i += blockDim.x * 16) *(uint4*)(sm + i) = make_uint4(i, id, 1, 2);
    fence_proxy_async_smem();
    __syncthreads();
    if (lane == 0) {
      for (int rep = 0; rep < p.inner; rep++)
        for (long long it = gw; it < (long long)items; it += GW) {
          tma_s2g(tgt[it % p.npeer] + (size_t)(it / p.npeer) * C, sm, C);
          tma_commit();
          tma_wait_read<3>();
        }
      tma_wait_all<0>();
    }
  } else {  // TMA_LD / TMA_LD_CL: per-warp ring of `stages` chunk buffers
    char* buf = sm + (size_t)warp * p.stages * C;
    uint64_t* bar = bars[warp];
    if (lane == 0) {
      for (int s = 0; s < p.stages; s++) mbar_init(&bar[s], 1);
      fence_mbar_init();
      long long k = 0;
      for (int rep = 0; rep < p.inner; rep++)
        for (long long it = gw; it < (long long)items; it += GW, k++) {
          int s = (int)(k % p.stages);
          if (k >= p.stages) mbar_wait(&bar[s], (uint32_t)((k / p.stages - 1) & 1));
          mbar_expect_tx(&bar[s], C);
          const char* src = tgt[it % p.npeer] + (size_t)(it / p.npeer) * C;
          if (M == TMA_LD) tma_g2s(buf + (size_t)s * C, src, C, &bar[s]);
          else tma_g2s_cluster(buf + (size_t)s * C, src, C, &bar[s]);
        }
      for (long long j = k > p.stages ? k - p.stages : 0; j < k; j++) mbar_wait(&bar[j % p.stages], (uint32_t)((j / p.stages) & 1));
    }
    __syncwarp();
  }
}

static const void* kfn(int m) {
  switch (m) {
    case PUSH_ST: return (const void*)bw_kernel<PUSH_ST>;
    case TMA_ST: return (const void*)bw_kernel<TMA_ST>;
    case TMA_LD: return (const void*)bw_kernel<TMA_LD>;
    case TMA_LD_CL: return (const void*)bw_kernel<TMA_LD_CL>;
    case PULL_LD: return (const void*)bw_kernel<PULL_LD>;
    default: return (const void*)bw_kernel<MC_ST>;
  }
}

int main(int argc, char** argv) {
  Args A(argc, argv);
  const int n = (int)A.i("n", 2);
  auto methods = A.ls("method", "push_st,tma_st,tma_ld,tma_ld_cluster,pull_ld");
  auto patterns = A.ls("pattern", "uni,a2a");
  auto chunks = A.li("chunk", "1K,2K,4K,8K,16K,32K");
  auto sizes = A.li("bytes", "1M,4M,16M,64M");
  auto cpsms = A.li("cpsm", "1,2");
  auto nsms = A.li("nsm", "0");          // 0 = all SMs
  const int warps = (int)A.i("warps", 8), stages = (int)A.i("stages", 4), tmaw = (int)A.i("tma_warps", 4);
  const int iters = (int)A.i("iters", 50), warmup = (int)A.i("warmup", 10), inner = (int)A.i("inner", 1);
  const std::string tag = A.s("tag", "");
  bool want_mc = std::find(methods.begin(), methods.end(), "mc_st") != methods.end();
  size_t maxB = *std::max_element(sizes.begin(), sizes.end());
  World W(n);
  run_ranks(n, [&](int rank) {
    Rank R; rank_setup(W, R, rank, (size_t)n * maxB, want_mc);
    unsigned* ticket; unsigned long long* sink;
    CK(cudaMalloc(&ticket, 4)); CK(cudaMalloc(&sink, 8)); CK(cudaMemset(sink, 0, 8));
    for (int m = 0; m < 6; m++) CK(cudaFuncSetAttribute(kfn(m), cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem_max(R)));
    if (rank == 0) Json().kv("prog", "bw").kv("event", "device").kv("n", n).kv("sms", R.sms).kv("smem_per_sm", R.smemSM).kv("clock_khz", R.clockKHz).kv("mc", R.mc ? 1 : 0).print();
    std::vector<cudaEvent_t> ev(2 * iters); for (auto& e : ev) CK(cudaEventCreate(&e));
    for (auto& mname : methods) {
      int m = (int)(std::find(MNAME, MNAME + 6, mname) - MNAME);
      if (m >= 6) { if (rank == 0) fprintf(stderr, "unknown method %s\n", mname.c_str()); continue; }
      bool mcAvail = true;
      if (m == MC_ST) { int ok = R.mc ? 1 : 0; W.vals[rank] = {(double)ok}; W.bar.wait(); for (int r = 0; r < n; r++) mcAvail &= W.vals[r][0] > 0; W.bar.wait(); }
      if (!mcAvail) { if (rank == 0) Json().kv("prog", "bw").kv("method", mname).kv("error", "multimem_unavailable").print(); continue; }
      for (auto& pat : patterns) {
        if (m == MC_ST && pat != "uni" && pat != "a2a") continue;
        bool active = pat == "a2a" || pat == "self" || rank == 0;
        std::vector<char*> tg;
        if (m == MC_ST) tg.push_back(R.mc + (size_t)rank * maxB);
        else if (pat == "self") tg.push_back(R.base + (size_t)rank * maxB);
        else if (pat == "uni") tg.push_back(R.peer[1] + (size_t)rank * maxB);
        else for (int k = 1; k < n; k++) tg.push_back(R.peer[(rank + k) % n] + (size_t)rank * maxB);  // fanout / a2a
        for (long long C : chunks) for (long long B : sizes) for (long long cpsm : cpsms) for (long long nsm0 : nsms) {
          int nsm = nsm0 <= 0 || nsm0 > R.sms ? R.sms : (int)nsm0;
          if (B % C) continue;
          int thr = warps * 32, need = 0, wEff = warps, sEff = stages;
          if (m == TMA_ST) { wEff = tmaw; thr = wEff * 32; need = (int)C; }
          if (m == TMA_LD || m == TMA_LD_CL) {
            wEff = tmaw; int cap = R.smemSM / (int)cpsm - 2048;
            while (wEff > 1 && (long long)wEff * sEff * C > cap) wEff--;
            while (sEff > 1 && (long long)wEff * sEff * C > cap) sEff--;
            thr = wEff * 32; need = (int)(wEff * sEff * C);
          }
          int smem = smem_for_cpsm(R, (int)cpsm, need);
          if (smem < 0) { if (rank == 0) Json().kv("prog", "bw").kv("method", mname).kv("chunk", C).kv("cpsm", cpsm).kv("skip", "smem").print(); continue; }
          BwP p{}; for (size_t i = 0; i < tg.size(); i++) p.tgt[i] = tg[i];
          p.npeer = (int)tg.size(); p.chunk = (int)C; p.stages = sEff; p.bytes = B; p.inner = inner;
          p.filter = nsm < R.sms; p.nsm = nsm; p.active = nsm * (int)cpsm; p.ticket = ticket; p.sink = sink;
          int grid = R.sms * (int)cpsm;
          if (!p.filter) grid = p.active;
          auto launch = [&]() {
            if (!active) return;
            if (m == TMA_LD_CL) {
              cudaLaunchConfig_t cfg{}; cfg.gridDim = grid; cfg.blockDim = thr; cfg.dynamicSmemBytes = smem; cfg.stream = R.st;
              cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
              cfg.attrs = at; cfg.numAttrs = 1;
              CK(cudaLaunchKernelEx(&cfg, bw_kernel<TMA_LD_CL>, p));
            } else {
              void* args[] = {&p};
              CK(cudaLaunchKernel(kfn(m), dim3(grid), dim3(thr), args, smem, R.st));
            }
          };
          for (int it = 0; it < warmup; it++) { align(W, R); CK(cudaMemsetAsync(ticket, 0, 4, R.st)); launch(); }
          CK(cudaStreamSynchronize(R.st));
          for (int it = 0; it < iters; it++) {
            align(W, R); CK(cudaMemsetAsync(ticket, 0, 4, R.st));
            CK(cudaEventRecord(ev[2 * it], R.st)); launch(); CK(cudaEventRecord(ev[2 * it + 1], R.st));
          }
          CK(cudaStreamSynchronize(R.st)); CK(cudaGetLastError());
          std::vector<double> t(iters, 0.0);
          if (active) for (int it = 0; it < iters; it++) { float ms; CK(cudaEventElapsedTime(&ms, ev[2 * it], ev[2 * it + 1])); t[it] = ms * 1e3; }
          unsigned got = 0; if (p.filter && active) CK(cudaMemcpy(&got, ticket, 4, cudaMemcpyDeviceToHost));
          auto mx = max_over_ranks(W, R, t);
          if (rank == 0) {
            double med = pct(mx, 0.5), bytesMoved = (double)p.npeer * B * inner;
            Json j; j.kv("prog", "bw").kv("tag", tag).kv("method", mname).kv("pattern", pat).kv("n", n).kv("chunk", C).kv("bytes_per_peer", B)
              .kv("npeer", p.npeer).kv("inner", inner).kv("cpsm", cpsm).kv("nsm", nsm).kv("warps", wEff).kv("stages", sEff).kv("grid", grid).kv("smem", smem)
              .kv("med_us", med).kv("p10_us", pct(mx, 0.1)).kv("p90_us", pct(mx, 0.9))
              .kv("egress_GBps", bytesMoved / (med * 1e3)).kv("per_peer_GBps", (double)B * inner / (med * 1e3));
            if (m == MC_ST) j.kv("delivered_GBps", bytesMoved * (n - 1) / (med * 1e3));
            if (p.filter) j.kv("active_ctas_seen", (long long)got).kv("active_ctas_expected", p.active);
            j.print();
          }
        }
      }
    }
    for (auto& e : ev) CK(cudaEventDestroy(e));
    rank_teardown(W, R);
  });
  return 0;
}
