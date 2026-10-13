// Shared host harness and PTX helpers for the round-2 NVLink microbenchmarks (one host thread per GPU).
#pragma once
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device/core.h>
#include <nccl_device/impl/comm__types.h>  // full ncclDevComm for ncclDevCommCreate
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <functional>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)
#define NK(x) do { ncclResult_t e_ = (x); if (e_ != ncclSuccess) { fprintf(stderr, "NCCL %s:%d %s\n", __FILE__, __LINE__, ncclGetErrorString(e_)); exit(1);} } while (0)

constexpr int MAXR = 8;

// ---------------- device PTX helpers ----------------
__device__ __forceinline__ uint64_t gtimer() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t smid() { uint32_t s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
__device__ __forceinline__ uint64_t clk64() { uint64_t t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory"); return t; }
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void st_v4(void* p, uint4 v) { asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ void st_v2(void* p, uint2 v) { asm volatile("st.global.v2.b32 [%0], {%1,%2};" :: "l"(p), "r"(v.x), "r"(v.y) : "memory"); }
__device__ __forceinline__ void st_u32(void* p, uint32_t v) { asm volatile("st.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ uint4 ld_v4(const void* p) { uint4 v; asm volatile("ld.global.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint4 ld_rlx_sys_v4(const void* p) { uint4 v; asm volatile("ld.relaxed.sys.global.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint4 ldv_v4(const void* p) { uint4 v; asm volatile("ld.volatile.global.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint2 ldv_v2(const void* p) { uint2 v; asm volatile("ld.volatile.global.v2.b32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint32_t ldv_u32(const void* p) { uint32_t v; asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rel_sys(void* p, uint32_t v) { asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_rlx_sys(void* p, uint32_t v) { asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ uint32_t ld_acq_sys(const void* p) { uint32_t v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint32_t ld_rlx_sys(const void* p) { uint32_t v; asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void red_rel_sys_add(void* p, uint32_t v) { asm volatile("red.release.sys.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void red_rlx_sys_add(void* p, uint32_t v) { asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void fence_acqrel_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ __forceinline__ void st_rel_gpu(void* p, uint32_t v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void fence_acqrel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
__device__ __forceinline__ void mc_st_v4(void* p, uint4 v) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

// mbarrier (CTA-local)
__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt) : "memory"); }
__device__ __forceinline__ void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(uint64_t* b, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n\tselp.b32 %0, 1, 0, p;\n\t}" : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
  return ok;
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) { while (!mbar_try_wait(b, parity)) {} }

// TMA 1-D bulk copies
__device__ __forceinline__ void tma_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void tma_g2s_cluster(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void tma_s2g(void* dst, const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void tma_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void tma_wait_all() { asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory"); }

// ---------------- host harness ----------------
struct Barrier {
  std::mutex m; std::condition_variable cv; int n, cnt = 0, phase = 0;
  explicit Barrier(int n) : n(n) {}
  void wait() { std::unique_lock<std::mutex> l(m); int ph = phase; if (++cnt == n) { cnt = 0; phase++; cv.notify_all(); } else cv.wait(l, [&] { return phase != ph; }); }
};

struct Args {
  std::map<std::string, std::string> kv;
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
      std::string a = argv[i];
      if (a.rfind("--", 0) != 0) continue;
      std::string v = (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) ? argv[++i] : "1";
      kv[a.substr(2)] = v;
    }
  }
  std::string s(const std::string& k, const std::string& d) const { auto it = kv.find(k); return it == kv.end() ? d : it->second; }
  long long i(const std::string& k, long long d) const { auto it = kv.find(k); return it == kv.end() ? d : parse(it->second); }
  static long long parse(const std::string& x) {  // accepts 4K, 2M suffixes
    long long m = 1; std::string t = x;
    if (!t.empty() && (t.back() == 'K' || t.back() == 'k')) { m = 1024; t.pop_back(); }
    else if (!t.empty() && (t.back() == 'M' || t.back() == 'm')) { m = 1 << 20; t.pop_back(); }
    return std::stoll(t) * m;
  }
  std::vector<long long> li(const std::string& k, const std::string& d) const {
    std::vector<long long> r; std::string x = s(k, d); size_t p = 0;
    while (p < x.size()) { size_t c = x.find(',', p); if (c == std::string::npos) c = x.size(); r.push_back(parse(x.substr(p, c - p))); p = c + 1; }
    return r;
  }
  std::vector<std::string> ls(const std::string& k, const std::string& d) const {
    std::vector<std::string> r; std::string x = s(k, d); size_t p = 0;
    while (p < x.size()) { size_t c = x.find(',', p); if (c == std::string::npos) c = x.size(); r.push_back(x.substr(p, c - p)); p = c + 1; }
    return r;
  }
};

struct Rank {
  int rank = 0, n = 0, sms = 0, smemSM = 0, smemOptin = 0, clockKHz = 0;
  ncclComm_t comm{};
  ncclWindow_t win{};
  char* base = nullptr;            // local window base
  char* peer[MAXR] = {};           // lsa pointers; peer[rank] is local
  char* mc = nullptr;              // NVLS multicast pointer (nullptr if unavailable)
  bool has_devcomm = false;
  ncclDevComm_t devcomm{};
  size_t win_bytes = 0;
  cudaStream_t st{};
  float* small = nullptr;
};

struct World {
  int n; Barrier bar; ncclUniqueId id; std::vector<std::vector<double>> vals;
  explicit World(int n) : n(n), bar(n), vals(n) { NK(ncclGetUniqueId(&id)); }
};

inline void rank_setup(World& W, Rank& R, int rank, size_t win_bytes, bool want_mc) {
  R.rank = rank; R.n = W.n;
  CK(cudaSetDevice(rank));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, rank));
  R.sms = p.multiProcessorCount;
  CK(cudaDeviceGetAttribute(&R.smemSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, rank));
  CK(cudaDeviceGetAttribute(&R.smemOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, rank));
  CK(cudaDeviceGetAttribute(&R.clockKHz, cudaDevAttrClockRate, rank));
  NK(ncclCommInitRank(&R.comm, W.n, W.id, rank));
  CK(cudaStreamCreateWithFlags(&R.st, cudaStreamNonBlocking));
  R.win_bytes = (win_bytes + (1 << 21) - 1) / (1 << 21) * (1 << 21);
  void* b; NK(ncclMemAlloc(&b, R.win_bytes));
  CK(cudaMemset(b, 0, R.win_bytes)); CK(cudaDeviceSynchronize());
  R.base = (char*)b;
  NK(ncclCommWindowRegister(R.comm, b, R.win_bytes, &R.win, NCCL_WIN_COLL_SYMMETRIC));
  ncclTeam_t lsa = ncclTeamLsa(R.comm);
  if (lsa.nRanks != W.n) { fprintf(stderr, "lsa team nRanks=%d != n=%d\n", lsa.nRanks, W.n); exit(1); }
  for (int r = 0; r < W.n; r++) { void* q; NK(ncclGetLsaDevicePointer(R.win, 0, r, &q)); R.peer[r] = (char*)q; }
  if (want_mc) {
    // devcomm with lsaMultimem binds the window to an NVLS multicast object (collective)
    ncclDevCommRequirements_t req = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    req.lsaMultimem = true;
    ncclResult_t e = ncclDevCommCreate(R.comm, &req, &R.devcomm);
    if (e == ncclSuccess) {
      R.has_devcomm = true;
      void* q = nullptr; e = ncclGetLsaMultimemDevicePointer(R.win, 0, &q);
      if (e == ncclSuccess) R.mc = (char*)q;
    }
    if (!R.mc) fprintf(stderr, "# rank %d: multimem unavailable (%s)\n", rank, ncclGetErrorString(e));
  }
  CK(cudaMalloc(&R.small, 64)); CK(cudaMemset(R.small, 0, 64));
  W.bar.wait();
}

inline void rank_teardown(World& W, Rank& R) {
  CK(cudaDeviceSynchronize());
  W.bar.wait();
  if (R.has_devcomm) NK(ncclDevCommDestroy(R.comm, &R.devcomm));
  NK(ncclCommWindowDeregister(R.comm, R.win));
  NK(ncclMemFree(R.base));
  NK(ncclCommDestroy(R.comm));
}

// Host barrier + tiny allreduce in-stream: GPUs start the next op within a few us of each other.
inline void align(World& W, Rank& R) {
  W.bar.wait();
  NK(ncclAllReduce(R.small, R.small, 1, ncclFloat, ncclSum, R.comm, R.st));
}

inline void run_ranks(int n, const std::function<void(int)>& f) {
  std::vector<std::thread> th;
  for (int r = 0; r < n; r++) th.emplace_back(f, r);
  for (auto& t : th) t.join();
}

inline double pct(std::vector<double> v, double q) {
  if (v.empty()) return 0; std::sort(v.begin(), v.end());
  size_t k = std::min(v.size() - 1, (size_t)(q * (v.size() - 1) + 0.5)); return v[k];
}

// Per-iteration max over ranks of a per-rank value (rank 0 returns the vector; others empty).
inline std::vector<double> max_over_ranks(World& W, Rank& R, const std::vector<double>& mine) {
  W.vals[R.rank] = mine; W.bar.wait();
  std::vector<double> mx;
  if (R.rank == 0) {
    mx.assign(mine.size(), 0.0);
    for (int r = 0; r < W.n; r++) for (size_t i = 0; i < mx.size() && i < W.vals[r].size(); i++) mx[i] = std::max(mx[i], W.vals[r][i]);
  }
  W.bar.wait();
  return mx;
}

// Minimal JSON line builder.
struct Json {
  std::string s = "{";
  Json& kv(const char* k, const std::string& v) { sep(); s += "\"" + std::string(k) + "\":\"" + v + "\""; return *this; }
  Json& kv(const char* k, const char* v) { return kv(k, std::string(v)); }
  Json& kv(const char* k, long long v) { sep(); s += "\"" + std::string(k) + "\":" + std::to_string(v); return *this; }
  Json& kv(const char* k, int v) { return kv(k, (long long)v); }
  Json& kv(const char* k, unsigned long long v) { sep(); s += "\"" + std::string(k) + "\":" + std::to_string(v); return *this; }
  Json& kv(const char* k, size_t v) { return kv(k, (unsigned long long)v); }
  Json& kv(const char* k, double v) { sep(); char b[64]; snprintf(b, sizeof b, "%.4f", v); s += "\"" + std::string(k) + "\":" + b; return *this; }
  void sep() { if (s.size() > 1) s += ","; }
  void print() { s += "}"; printf("%s\n", s.c_str()); fflush(stdout); }
};

// Dynamic smem that pins residency to exactly `cpsm` CTAs/SM (so an SM subset can be selected via %smid).
inline int dyn_smem_max(const Rank& R) { return R.smemOptin - 4096; }
inline int smem_for_cpsm(const Rank& R, int cpsm, int need) {
  int per = std::min(R.smemSM / cpsm - 1024, R.smemOptin - 4096);  // fits cpsm CTAs and leaves room for static smem
  int lo = R.smemSM / (cpsm + 1) - 1024 + 1;  // min that prevents cpsm+1
  int s = std::max(need, lo);
  return s <= per ? s : -1;
}
