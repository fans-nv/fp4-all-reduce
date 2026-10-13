// tear: remote writer pushes same-generation 16 B (or 8 B) packets into the peer's inbox while local readers
// poll it; counts torn observations (words of one packet from different generations). Ring: r -> (r+1)%n.
#include "micro_common.cuh"

enum WM { W_ST_V4 = 0, W_ST_V2, W_TMA_ST, W_SCALAR4, W_SPLIT };
static const char* WNAME[] = {"st_v4", "st_v2", "tma_st", "scalar4", "split"};
enum RM { R_LDV_V4 = 0, R_LDV_V2, R_TMA_LD };
static const char* RNAME[] = {"ldv_v4", "ldv_v2", "tma_ld"};
constexpr size_t INBOX = 1 << 20;
constexpr int TMA_CHUNK = 2048;  // per-warp TMA tile (writer and reader)

struct TP {
  char* dst; char* inbox; size_t bytes; int wctas; unsigned long long dur_ns;
  unsigned long long* cnt;  // [0]=reads [1]=torn [2]=transitions [3]=regress [4]=writes
};

template <int W, int R>
__global__ void __launch_bounds__(256) tear_kernel(TP p) {
  __shared__ __align__(128) char sbuf[8][TMA_CHUNK];
  __shared__ uint32_t slast[8][TMA_CHUNK / 16];
  __shared__ __align__(8) uint64_t bar[8];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
  const unsigned long long t0 = gtimer();
  if ((int)blockIdx.x < p.wctas) {  // ---- writer
    const int SL = W == W_ST_V2 ? 8 : 16;
    const size_t nslot = p.bytes / SL;
    const size_t gt = (size_t)blockIdx.x * blockDim.x + tid, GT = (size_t)p.wctas * blockDim.x;
    unsigned long long writes = 0;
    for (uint32_t g = 1;; g++) {
      if (W == W_TMA_ST) {
        const size_t nreg = p.bytes / TMA_CHUNK, wg = (size_t)blockIdx.x * nw + warp, WG = (size_t)p.wctas * nw;
        for (size_t c = wg; c < nreg; c += WG) {
          for (int i = lane * 16; i < TMA_CHUNK; i += 512) *(uint4*)(sbuf[warp] + i) = make_uint4(g, g, g, g);
          fence_proxy_async_smem(); __syncwarp();
          if (lane == 0) { tma_s2g(p.dst + c * TMA_CHUNK, sbuf[warp], TMA_CHUNK); tma_commit(); tma_wait_read<0>(); }
          __syncwarp(); writes += TMA_CHUNK / 16;
        }
      } else {
        for (size_t s = gt; s < nslot; s += GT) {
          char* a = p.dst + s * SL;
          if (W == W_ST_V4) st_v4(a, make_uint4(g, g, g, g));
          else if (W == W_ST_V2) st_v2(a, make_uint2(g, g));
          else if (W == W_SCALAR4) { st_u32(a, g); st_u32(a + 4, g); st_u32(a + 8, g); st_u32(a + 12, g); }
          else st_u32(a + 4 * (g & 3), g);  // split: one word per pass, so packets are torn by construction
          writes++;
        }
      }
      if (gtimer() - t0 > p.dur_ns) break;
    }
    if (W == W_TMA_ST && lane == 0) tma_wait_all<0>();
    if (W != W_TMA_ST || lane == 0) atomicAdd(&p.cnt[4], writes);  // TMA: lane-uniform count, add once per warp
    return;
  }
  // ---- reader
  unsigned long long reads = 0, torn = 0, trans = 0, regress = 0;
  const int rb = blockIdx.x - p.wctas, RB = gridDim.x - p.wctas;
  if (R == R_TMA_LD) {
    const size_t nreg = p.bytes / TMA_CHUNK, wg = (size_t)rb * nw + warp, WG = (size_t)RB * nw;
    if (lane == 0) { mbar_init(&bar[warp], 1); fence_mbar_init(); }
    for (int i = lane; i < TMA_CHUNK / 16; i += 32) slast[warp][i] = 0;
    __syncwarp();
    uint32_t ph = 0; const size_t c = wg % nreg;  // fixed region so slast tracks the same slots
    while (gtimer() - t0 < p.dur_ns) {
      if (lane == 0) { mbar_expect_tx(&bar[warp], TMA_CHUNK); tma_g2s(sbuf[warp], p.inbox + c * TMA_CHUNK, TMA_CHUNK, &bar[warp]); }
      mbar_wait(&bar[warp], ph); ph ^= 1;
      for (int i = lane; i < TMA_CHUNK / 16; i += 32) {
        uint4 v; asm volatile("ld.volatile.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(smem_u32(sbuf[warp] + i * 16)));
        reads++;
        if (v.x != v.y || v.x != v.z || v.x != v.w) torn++;
        uint32_t l = slast[warp][i]; if (v.x != l) { trans++; if (v.x < l) regress++; slast[warp][i] = v.x; }
      }
      __syncwarp();
      fence_proxy_async_smem();  // generic reads of the tile complete before the next TMA overwrites it
    }
  } else {
    const int SL = R == R_LDV_V2 ? 8 : 16;
    const size_t nslot = p.bytes / SL, gt = (size_t)rb * blockDim.x + tid, GT = (size_t)RB * blockDim.x;
    size_t s[4]; uint32_t last[4];
#pragma unroll
    for (int j = 0; j < 4; j++) { s[j] = (gt + j * GT) % nslot; last[j] = 0; }
    while (gtimer() - t0 < p.dur_ns) {
#pragma unroll 1
      for (int k = 0; k < 64; k++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const char* a = p.inbox + s[j] * SL;
          uint32_t x; bool bad;
          if (R == R_LDV_V2) { uint2 v = ldv_v2(a); x = v.x; bad = v.x != v.y; }
          else { uint4 v = ldv_v4(a); x = v.x; bad = v.x != v.y || v.x != v.z || v.x != v.w; }
          reads++; torn += bad;
          if (x != last[j]) { trans++; regress += x < last[j]; last[j] = x; }
        }
      }
    }
  }
  atomicAdd(&p.cnt[0], reads); atomicAdd(&p.cnt[1], torn); atomicAdd(&p.cnt[2], trans); atomicAdd(&p.cnt[3], regress);
}

template <int W, int R> static const void* kf() { return (const void*)tear_kernel<W, R>; }
static const void* kfn(int w, int r) {
  static const void* t[5][3] = {
    {kf<0, 0>(), kf<0, 1>(), kf<0, 2>()}, {kf<1, 0>(), kf<1, 1>(), kf<1, 2>()}, {kf<2, 0>(), kf<2, 1>(), kf<2, 2>()},
    {kf<3, 0>(), kf<3, 1>(), kf<3, 2>()}, {kf<4, 0>(), kf<4, 1>(), kf<4, 2>()}};
  return t[w][r];
}

int main(int argc, char** argv) {
  Args A(argc, argv);
  const int n = (int)A.i("n", 2);
  auto pairs = A.ls("pairs", "st_v4:ldv_v4,st_v4:tma_ld,tma_st:ldv_v4,tma_st:tma_ld,st_v2:ldv_v2,scalar4:ldv_v4,split:ldv_v4");
  auto sizes = A.li("bytes", "64K");
  const long long ms = A.i("ms", 2000);
  const std::string tag = A.s("tag", "");
  World W(n);
  run_ranks(n, [&](int rank) {
    Rank R; rank_setup(W, R, rank, INBOX + (1 << 21), false);
    unsigned long long* cnt; CK(cudaMalloc(&cnt, 64));
    const int wctas = (int)A.i("wctas", R.sms / 2), rctas = (int)A.i("rctas", R.sms - wctas);
    for (auto& pr : pairs) {
      auto c = pr.find(':'); std::string wn = pr.substr(0, c), rn = pr.substr(c + 1);
      int w = (int)(std::find(WNAME, WNAME + 5, wn) - WNAME), r = (int)(std::find(RNAME, RNAME + 3, rn) - RNAME);
      if (w >= 5 || r >= 3) { if (rank == 0) fprintf(stderr, "bad pair %s\n", pr.c_str()); continue; }
      for (long long B : sizes) {
        TP p{}; p.dst = R.peer[(rank + 1) % n] + INBOX; p.inbox = R.base + INBOX; p.bytes = B; p.wctas = wctas;
        p.dur_ns = (unsigned long long)ms * 1000000ull; p.cnt = cnt;
        CK(cudaMemsetAsync(R.base + INBOX, 0, B, R.st)); CK(cudaMemsetAsync(cnt, 0, 64, R.st));
        align(W, R);
        void* args[] = {&p};
        CK(cudaLaunchKernel(kfn(w, r), dim3(wctas + rctas), dim3(256), args, 0, R.st));
        CK(cudaStreamSynchronize(R.st)); CK(cudaGetLastError());
        unsigned long long h[5]; CK(cudaMemcpy(h, cnt, 40, cudaMemcpyDeviceToHost));
        std::vector<double> mine(h, h + 5);
        W.vals[rank] = mine; W.bar.wait();
        if (rank == 0) {
          double s[5] = {0, 0, 0, 0, 0};
          for (int q = 0; q < n; q++) for (int i = 0; i < 5; i++) s[i] += W.vals[q][i];
          Json().kv("prog", "tear").kv("tag", tag).kv("n", n).kv("writer", wn).kv("reader", rn).kv("bytes", B).kv("ms", ms)
            .kv("wctas", wctas).kv("rctas", rctas).kv("reads", (unsigned long long)s[0]).kv("torn", (unsigned long long)s[1])
            .kv("transitions", (unsigned long long)s[2]).kv("regress", (unsigned long long)s[3]).kv("writes", (unsigned long long)s[4])
            .kv("torn_per_1e9", s[0] > 0 ? s[1] / s[0] * 1e9 : 0.0).print();
        }
        W.bar.wait();
      }
    }
    rank_teardown(W, R);
  });
  return 0;
}
