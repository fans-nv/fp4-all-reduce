// Peer bandwidth: copy engines (cudaMemcpyPeerAsync) vs SM st.v4 push / ld.v4 pull, GPU0<->GPU1, one process.
// Usage: ce --mode ce|sm_push|sm_pull|ce+sm --dir uni|bidir --sizes 1M,..,4G --streams 1,2,4 --side src|dst
//           --ctas 16,32,64,212 --threads 512 --iters N --hold_s S (hold: loop for S seconds, one JSON line)
// bytes = payload per direction; GB/s = bytes / median time (1e9). bidir: both directions concurrently, GB/s is per direction.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

__global__ void kcopy(const int4* __restrict__ s, int4* __restrict__ d, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, st = (size_t)gridDim.x * blockDim.x;
  for (; i + 3 * st < n; i += 4 * st) {  // 4 independent 16 B ops in flight per thread
    int4 a = s[i], b = s[i + st], c = s[i + 2 * st], e = s[i + 3 * st];
    d[i] = a; d[i + st] = b; d[i + 2 * st] = c; d[i + 3 * st] = e;
  }
  for (; i < n; i += st) d[i] = s[i];
}

static std::map<std::string, std::string> A;
static std::string S(const char* k, const char* d) { auto it = A.find(k); return it == A.end() ? d : it->second; }
static long long P(std::string t) { long long m = 1; char c = t.back();
  if (c == 'K') m = 1LL << 10; else if (c == 'M') m = 1LL << 20; else if (c == 'G') m = 1LL << 30; if (m > 1) t.pop_back(); return std::stoll(t) * m; }
static std::vector<long long> L(const char* k, const char* d) { std::vector<long long> r; std::string x = S(k, d); size_t p = 0;
  while (p < x.size()) { size_t c = x.find(',', p); if (c == std::string::npos) c = x.size(); r.push_back(P(x.substr(p, c - p))); p = c + 1; } return r; }

struct Dir { int sd, dd; char *src, *dst; std::vector<cudaStream_t> cs; cudaStream_t ks; cudaEvent_t e0, e1, ek; std::vector<cudaEvent_t> ee; };

int main(int argc, char** argv) {
  for (int i = 1; i < argc; i++) { std::string a = argv[i]; if (a.rfind("--", 0) == 0 && i + 1 < argc) A[a.substr(2)] = argv[++i]; }
  std::string mode = S("mode", "ce"), dirm = S("dir", "uni"), side = S("side", "src"), tag = S("tag", "");
  auto sizes = L("sizes", "1M,4M,16M,64M,256M,1G,4G"), strs = L("streams", "1"), ctas = L("ctas", "212");
  int threads = (int)P(S("threads", "512")); int iters0 = (int)P(S("iters", "0")); double hold = atof(S("hold_s", "0").c_str());
  long long smfrac = P(S("smpct", "50"));  // ce+sm: percent of bytes moved by the SM kernel
  long long maxb = *std::max_element(sizes.begin(), sizes.end());
  int nd = dirm == "bidir" ? 2 : 1;
  CK(cudaSetDevice(0)); CK(cudaDeviceEnablePeerAccess(1, 0)); CK(cudaSetDevice(1)); CK(cudaDeviceEnablePeerAccess(0, 0));
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  printf("{\"event\":\"device\",\"name\":\"%s\",\"sms\":%d,\"asyncEngines\":%d,\"maxb\":%lld}\n", pr.name, pr.multiProcessorCount, pr.asyncEngineCount, maxb);
  Dir D[2];
  for (int k = 0; k < nd; k++) {
    D[k].sd = k; D[k].dd = 1 - k;
    CK(cudaSetDevice(D[k].sd)); CK(cudaMalloc(&D[k].src, maxb)); CK(cudaMemset(D[k].src, k + 1, maxb));
    CK(cudaSetDevice(D[k].dd)); CK(cudaMalloc(&D[k].dst, maxb)); CK(cudaMemset(D[k].dst, 0, maxb));
  }
  for (int k = 0; k < nd; k++) {  // streams live on the issuing device: CE side for ce, push=src / pull=dst for sm
    int cdev = side == "dst" ? D[k].dd : D[k].sd; CK(cudaSetDevice(cdev));
    for (int s = 0; s < 16; s++) { cudaStream_t t; CK(cudaStreamCreateWithFlags(&t, cudaStreamNonBlocking)); D[k].cs.push_back(t);
      cudaEvent_t e; CK(cudaEventCreate(&e)); D[k].ee.push_back(e); }
    CK(cudaEventCreate(&D[k].e0)); CK(cudaEventCreate(&D[k].e1));
    int kdev = mode == "sm_pull" ? D[k].dd : D[k].sd; CK(cudaSetDevice(kdev));
    CK(cudaStreamCreateWithFlags(&D[k].ks, cudaStreamNonBlocking)); CK(cudaEventCreate(&D[k].ek));
  }
  for (auto B : sizes) for (auto ns : strs) for (auto nc : ctas) {
    bool useCE = mode == "ce" || mode == "ce+sm", useSM = mode != "ce";
    if (!useCE && ns != strs[0]) continue; if (!useSM && nc != ctas[0]) continue;
    long long bsm = mode == "ce+sm" ? (B * smfrac / 100) & ~4095LL : (useSM ? B : 0), bce = B - bsm;
    auto launch = [&]() {
      for (int k = 0; k < nd; k++) {
        Dir& d = D[k]; int cdev = side == "dst" ? d.dd : d.sd;
        CK(cudaSetDevice(cdev)); CK(cudaEventRecord(d.e0, d.cs[0]));
        if (useCE) { long long per = (bce / ns) & ~4095LL;
          for (int s = 0; s < ns; s++) { long long off = s * per, len = s == ns - 1 ? bce - off : per;
            if (s) CK(cudaStreamWaitEvent(d.cs[s], d.e0, 0));
            CK(cudaMemcpyPeerAsync(d.dst + off, d.dd, d.src + off, d.sd, len, d.cs[s])); } }
        if (useSM) { int kdev = mode == "sm_pull" ? d.dd : d.sd; CK(cudaSetDevice(kdev));
          if (kdev != cdev) { /* cross-device ordering not needed: e0 is a local timestamp; both start ASAP */ }
          else CK(cudaStreamWaitEvent(d.ks, d.e0, 0));
          kcopy<<<(int)nc, threads, 0, d.ks>>>((const int4*)(d.src + bce), (int4*)(d.dst + bce), (size_t)bsm / 16);
          CK(cudaGetLastError()); CK(cudaEventRecord(d.ek, d.ks)); CK(cudaSetDevice(cdev)); CK(cudaStreamWaitEvent(d.cs[0], d.ek, 0)); }
        if (useCE) for (int s = 1; s < ns; s++) { CK(cudaEventRecord(d.ee[s], d.cs[s])); CK(cudaStreamWaitEvent(d.cs[0], d.ee[s], 0)); }
        CK(cudaEventRecord(d.e1, d.cs[0]));
      }
    };
    auto sync = [&]() { for (int k = 0; k < nd; k++) { CK(cudaSetDevice(D[k].sd)); CK(cudaDeviceSynchronize()); CK(cudaSetDevice(D[k].dd)); CK(cudaDeviceSynchronize()); } };
    for (int w = 0; w < 3; w++) launch(); sync();
    if (hold > 0) {  // back-to-back for hold_s seconds; wall clock includes everything
      long long n = 0; auto t0 = std::chrono::steady_clock::now(); double el = 0;
      while (el < hold) { for (int j = 0; j < 8; j++) launch(); n += 8; sync(); el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
      printf("{\"tag\":\"%s\",\"mode\":\"%s\",\"dir\":\"%s\",\"side\":\"%s\",\"bytes\":%lld,\"streams\":%lld,\"ctas\":%lld,\"hold_s\":%.3f,\"reps\":%lld,\"payload_per_dir_bytes\":%lld,\"GBps_per_dir\":%.1f}\n",
             tag.c_str(), mode.c_str(), dirm.c_str(), side.c_str(), B, ns, nc, el, n, B * n, B * n / el / 1e9);
      fflush(stdout); continue;
    }
    int iters = iters0 ? iters0 : (int)std::max(10LL, std::min(400LL, (16LL << 30) / B));
    std::vector<float> t;
    for (int i = 0; i < iters; i++) {
      launch(); sync(); float mx = 0;
      for (int k = 0; k < nd; k++) { float ms; CK(cudaEventElapsedTime(&ms, D[k].e0, D[k].e1)); mx = std::max(mx, ms); }
      t.push_back(mx);
    }
    std::sort(t.begin(), t.end()); float med = t[t.size() / 2];
    printf("{\"tag\":\"%s\",\"mode\":\"%s\",\"dir\":\"%s\",\"side\":\"%s\",\"bytes\":%lld,\"sm_bytes\":%lld,\"streams\":%lld,\"ctas\":%lld,\"threads\":%d,\"iters\":%d,\"med_us\":%.2f,\"p10_us\":%.2f,\"GBps_per_dir\":%.1f,\"GBps_best\":%.1f}\n",
           tag.c_str(), mode.c_str(), dirm.c_str(), side.c_str(), B, bsm, ns, nc, threads, iters, med * 1e3, t[t.size() / 10] * 1e3, B / (med * 1e-3) / 1e9, B / (t[0] * 1e-3) / 1e9);
    fflush(stdout);
  }
  return 0;
}
