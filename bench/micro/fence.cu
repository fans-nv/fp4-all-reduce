// fence: (A) cost of a sys-scope release/fence after N outstanding remote vs local stores;
//        (B) throughput of a persistent push loop with per-tile publication variants (round-1 pattern vs none).
#include "micro_common.cuh"

enum FOp { OP_NONE = 0, OP_ST_REL_SYS, OP_FENCE_SYS, OP_FENCE_GPU, OP_RED_REL_SYS, OP_ST_RLX_SYS, OP_ST_REL_GPU };
static const char* FNAME[] = {"none", "st_release_sys", "fence_acq_rel_sys", "fence_acq_rel_gpu", "red_release_sys_add", "st_relaxed_sys", "st_release_gpu"};
enum TV { TV_NONE = 0, TV_BAR, TV_BAR_REL, TV_BAR_RED, TV_WARP_REL, TV_BAR_FENCE_RLX, TV_BAR_REL_GPU };
static const char* TNAME[] = {"none", "bar", "bar_rel_sys", "bar_red_rel_sys", "warp_rel_sys", "bar_fence_rlx", "bar_rel_gpu_local"};
constexpr size_t FLAG_OFF = 0, DATA_OFF = 1 << 20;

struct FA {
  char* data; char* flag; int nst, op, trials;
  unsigned long long* out;     // [trials] cycles of the publication op (CTA 0)
  unsigned long long* tot;     // [2] globaltimer ns, clock64 cycles over all trials (for MHz)
  volatile unsigned* stop; char* bgdst; size_t bgbytes;
};

__device__ __forceinline__ void do_op(int op, char* f, uint32_t v) {
  switch (op) {
    case OP_ST_REL_SYS: st_rel_sys(f, v); break;
    case OP_FENCE_SYS: fence_acqrel_sys(); break;
    case OP_FENCE_GPU: fence_acqrel_gpu(); break;
    case OP_RED_REL_SYS: red_rel_sys_add(f, 1); break;
    case OP_ST_RLX_SYS: st_rlx_sys(f, v); break;
    case OP_ST_REL_GPU: st_rel_gpu(f, v); break;
    default: break;
  }
}

// CTA 0 measures; CTAs >= 1 (if any) push background NVLink traffic until CTA 0 is done.
__global__ void __launch_bounds__(1024) fence_lat(FA p) {
  const int tid = threadIdx.x, T = blockDim.x;
  if (blockIdx.x > 0) {
    size_t off = ((size_t)(blockIdx.x - 1) * T + tid) * 16, step = (size_t)(gridDim.x - 1) * T * 16;
    while (!*p.stop) { for (size_t o = off; o < p.bgbytes; o += step) st_v4(p.bgdst + o, make_uint4(1, 2, 3, 4)); }
    return;
  }
  unsigned long long g0 = gtimer(), c0 = clk64();
  for (int t = 0; t < p.trials; t++) {
    for (int k = 0; k < p.nst; k++) st_v4(p.data + ((size_t)k * T + tid) * 16, make_uint4(t, k, tid, 0));
    __syncthreads();  // round-1 pattern: CTA barrier then one thread publishes
    if (tid == 0) {
      unsigned long long a = clk64();
      do_op(p.op, p.flag, t + 1);
      unsigned long long b = clk64();
      p.out[t] = b - a;
    }
    __syncthreads();
  }
  if (tid == 0) { p.tot[0] = gtimer() - g0; p.tot[1] = clk64() - c0; *p.stop = 1; }
}

struct FT {
  char* tgt[MAXR]; int npeer; size_t bytes; int tile, var;
  char* flag[MAXR];  // per-peer flag slot base (per-CTA offsets added)
  char* lflag;
};

__global__ void __launch_bounds__(1024) fence_tp(FT p) {
  __shared__ char* tgt[MAXR]; __shared__ char* flg[MAXR];
  if (threadIdx.x < MAXR) { tgt[threadIdx.x] = p.tgt[threadIdx.x]; flg[threadIdx.x] = p.flag[threadIdx.x]; }
  __syncthreads();
  const int tid = threadIdx.x, T = blockDim.x, lane = tid & 31;
  const size_t ntile = p.bytes / p.tile, items = ntile * p.npeer;
  uint32_t seq = 0;
  for (size_t it = blockIdx.x; it < items; it += gridDim.x) {
    const int pe = (int)(it % p.npeer);
    char* d = tgt[pe] + (it / p.npeer) * p.tile;
    for (int off = tid * 16; off < p.tile; off += T * 16) st_v4(d + off, make_uint4((uint32_t)it, off, 0, 0));
    seq++;
    char* f = flg[pe] + (size_t)blockIdx.x * 128;
    switch (p.var) {
      case TV_BAR: __syncthreads(); break;
      case TV_BAR_REL: __syncthreads(); if (tid == 0) st_rel_sys(f, seq); break;
      case TV_BAR_RED: __syncthreads(); if (tid == 0) red_rel_sys_add(f, 1); break;
      case TV_WARP_REL: __syncwarp(); if (lane == 0) st_rel_sys(f + 4 * (tid >> 5), seq); break;
      case TV_BAR_FENCE_RLX: __syncthreads(); if (tid == 0) { fence_acqrel_sys(); st_rlx_sys(f, seq); } break;
      case TV_BAR_REL_GPU: __syncthreads(); if (tid == 0) st_rel_gpu(p.lflag + (size_t)blockIdx.x * 128, seq); break;
      default: break;
    }
  }
}

int main(int argc, char** argv) {
  Args A(argc, argv);
  const int n = (int)A.i("n", 2);
  const std::string part = A.s("part", "lat,tp"), tag = A.s("tag", "");
  // part A
  auto nsts = A.li("nst", "0,1,4,16,64");
  auto ops = A.ls("op", "none,st_release_sys,fence_acq_rel_sys,fence_acq_rel_gpu,red_release_sys_add,st_relaxed_sys,st_release_gpu");
  auto dsts = A.ls("dst", "peer,local");
  auto flagdsts = A.ls("flagdst", "peer,local");
  auto lwarps = A.li("lwarps", "1,8");
  auto bgs = A.li("bg", "0,1");
  const int trials = (int)A.i("trials", 200);
  // part B
  auto vars = A.ls("var", "none,bar,bar_rel_sys,bar_red_rel_sys,warp_rel_sys,bar_fence_rlx,bar_rel_gpu_local");
  auto tiles = A.li("tile", "7K,28K");
  auto cpsms = A.li("cpsm", "1,2");
  const long long tbytes = A.i("bytes", 56LL << 20);
  const int twarps = (int)A.i("warps", 8), iters = (int)A.i("iters", 30), warmup = (int)A.i("warmup", 5);
  const std::string pattern = A.s("pattern", "a2a");
  size_t slot = std::max<size_t>(tbytes, (size_t)256 * 1024 * 16 * 2);
  World W(n);
  run_ranks(n, [&](int rank) {
    Rank R; rank_setup(W, R, rank, DATA_OFF + (size_t)n * slot, false);
    const char* ppeer = R.peer[(rank + 1) % n];
    if (part.find("lat") != std::string::npos) {
      unsigned long long *out, *tot; unsigned* stop;
      CK(cudaMalloc(&out, trials * 8)); CK(cudaMalloc(&tot, 16)); CK(cudaMalloc(&stop, 4));
      for (auto& dst : dsts) for (auto& fd : flagdsts) for (long long w : lwarps) for (long long bg : bgs) for (long long nst : nsts) for (auto& on : ops) {
        int op = (int)(std::find(FNAME, FNAME + 7, on) - FNAME); if (op >= 7) continue;
        FA p{}; int T = (int)w * 32;
        p.data = (dst == "peer" ? (char*)ppeer : R.base) + DATA_OFF + (size_t)rank * slot;
        p.flag = (fd == "peer" ? (char*)ppeer : R.base) + FLAG_OFF + (size_t)rank * 4096;
        p.nst = (int)nst; p.op = op; p.trials = trials; p.out = out; p.tot = tot; p.stop = stop;
        p.bgdst = (char*)ppeer + DATA_OFF + (size_t)rank * slot + slot / 2; p.bgbytes = slot / 2;
        CK(cudaMemsetAsync(stop, 0, 4, R.st));
        align(W, R);  // every rank measures simultaneously (all send to rank+1)
        fence_lat<<<bg ? R.sms : 1, T, 0, R.st>>>(p);
        CK(cudaStreamSynchronize(R.st)); CK(cudaGetLastError());
        std::vector<unsigned long long> h(trials); unsigned long long tt[2];
        CK(cudaMemcpy(h.data(), out, trials * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(tt, tot, 16, cudaMemcpyDeviceToHost));
        std::vector<double> v; for (int i = trials / 10; i < trials; i++) v.push_back((double)h[i]);
        double mhz = tt[0] ? (double)tt[1] / tt[0] * 1e3 : 0;
        auto mx = max_over_ranks(W, R, {pct(v, 0.5)});
        if (rank == 0)
          Json().kv("prog", "fence").kv("part", "lat").kv("tag", tag).kv("n", n).kv("op", on).kv("dst", dst).kv("flagdst", fd).kv("warps", w).kv("bg", bg)
            .kv("nst_per_thread", nst).kv("outstanding_bytes", (long long)nst * T * 16)
            .kv("rank0_med_cyc", pct(v, 0.5)).kv("rank0_p90_cyc", pct(v, 0.9)).kv("max_rank_med_cyc", mx[0]).kv("eff_mhz", mhz)
            .kv("rank0_med_ns", mhz > 0 ? pct(v, 0.5) / mhz * 1e3 : 0.0).print();
        W.bar.wait();
      }
    }
    if (part.find("tp") != std::string::npos) {
      std::vector<cudaEvent_t> ev(2 * iters); for (auto& e : ev) CK(cudaEventCreate(&e));
      for (long long tile : tiles) for (long long cpsm : cpsms) {
        double tnone = 0;
        for (auto& vn : vars) {
          int var = (int)(std::find(TNAME, TNAME + 7, vn) - TNAME); if (var >= 7) continue;
          FT p{}; p.var = var; p.tile = (int)tile; p.bytes = (size_t)tbytes / tile * tile;
          if (pattern == "uni") { p.npeer = 1; p.tgt[0] = R.peer[1] + DATA_OFF; p.flag[0] = R.peer[1] + FLAG_OFF + (1 << 19); }
          else { p.npeer = n - 1; for (int k = 1; k < n; k++) { int q = (rank + k) % n; p.tgt[k - 1] = R.peer[q] + DATA_OFF + (size_t)rank * slot; p.flag[k - 1] = R.peer[q] + FLAG_OFF + (1 << 19) + (size_t)rank * (1 << 16); } }
          p.lflag = R.base + FLAG_OFF + (1 << 18);
          bool active = pattern != "uni" || rank == 0;
          int grid = R.sms * (int)cpsm;
          auto launch = [&]() { if (active) fence_tp<<<grid, twarps * 32, 0, R.st>>>(p); };
          for (int i = 0; i < warmup; i++) { align(W, R); launch(); }
          for (int i = 0; i < iters; i++) { align(W, R); CK(cudaEventRecord(ev[2 * i], R.st)); launch(); CK(cudaEventRecord(ev[2 * i + 1], R.st)); }
          CK(cudaStreamSynchronize(R.st)); CK(cudaGetLastError());
          std::vector<double> t(iters, 0.0);
          if (active) for (int i = 0; i < iters; i++) { float ms; CK(cudaEventElapsedTime(&ms, ev[2 * i], ev[2 * i + 1])); t[i] = ms * 1e3; }
          auto mx = max_over_ranks(W, R, t);
          if (rank == 0) {
            double med = pct(mx, 0.5); if (var == TV_NONE) tnone = med;
            double tilesPerCta = (double)(p.bytes / tile) * p.npeer / grid;
            Json().kv("prog", "fence").kv("part", "tp").kv("tag", tag).kv("n", n).kv("pattern", pattern).kv("var", vn).kv("tile", tile).kv("cpsm", cpsm)
              .kv("warps", twarps).kv("grid", grid).kv("bytes_per_peer", (long long)p.bytes).kv("med_us", med).kv("p90_us", pct(mx, 0.9))
              .kv("egress_GBps", (double)p.bytes * p.npeer / (med * 1e3)).kv("tiles_per_cta", tilesPerCta)
              .kv("overhead_ns_per_tile", tnone > 0 ? (med - tnone) * 1e3 / tilesPerCta : 0.0).print();
          }
        }
      }
      for (auto& e : ev) CK(cudaEventDestroy(e));
    }
    rank_teardown(W, R);
  });
  return 0;
}
