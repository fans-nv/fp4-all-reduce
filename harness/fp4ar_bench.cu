// Round-2 host driver: one thread per GPU, NCCL symmetric window, any linked Fp4ArVariant, check gate, timing, BF16 baseline.
// Variants are linked in via -DFP4AR_VARIANT_LIST='V(r1) V(r1c4) ...'; each provides const Fp4ArVariant* fp4ar_register_<name>().
#include <cuda_profiler_api.h>
#include "fp4ar_api.h"
#include "fp4_codec.cuh"
#include "ref.cuh"
#include <nccl_device/core.h>
#include <dlfcn.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <functional>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); fflush(stdout); _exit(1);} } while (0)
#define NK(x) do { ncclResult_t e_ = (x); if (e_ != ncclSuccess) { fprintf(stderr, "NCCL %s:%d %s\n", __FILE__, __LINE__, ncclGetErrorString(e_)); fflush(stdout); _exit(1);} } while (0)

#ifndef FP4AR_VARIANT_LIST
#define FP4AR_VARIANT_LIST V(r1)
#endif
#define V(n) const Fp4ArVariant* fp4ar_register_##n();
FP4AR_VARIANT_LIST
#undef V
static std::vector<const Fp4ArVariant*> all_variants() {
  std::vector<const Fp4ArVariant*> v;
#define V(n) v.push_back(fp4ar_register_##n());
  FP4AR_VARIANT_LIST
#undef V
  return v;
}

using fp4c::H;
constexpr int NSETS = 3;  // input sets cycled across calls so stale-buffer bugs cannot pass

// ---------------- NVML via dlopen (no link dependency) ----------------
struct Nvml {
  typedef int (*init_t)();
  typedef int (*byPci_t)(const char*, void**);
  typedef int (*clk_t)(void*, int, unsigned*);
  init_t init = nullptr; byPci_t byPci = nullptr; clk_t clk = nullptr; bool ok = false;
  Nvml() {
    void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!h) return;
    init = (init_t)dlsym(h, "nvmlInit_v2"); byPci = (byPci_t)dlsym(h, "nvmlDeviceGetHandleByPciBusId_v2");
    clk = (clk_t)dlsym(h, "nvmlDeviceGetClockInfo");
    ok = init && byPci && clk && init() == 0;
  }
  // type 1 = SM, 2 = MEM
  unsigned get(const char* bus, int type) {
    void* d; unsigned v = 0;
    if (!ok || byPci(bus, &d) != 0 || clk(d, type, &v) != 0) return 0;
    return v;
  }
};
static Nvml* NVML;

// ---------------- options ----------------
static std::vector<int> ilist(const std::string& s) {
  std::vector<int> v; size_t p = 0;
  while (p < s.size()) { size_t c = s.find(',', p); if (c == std::string::npos) c = s.size(); v.push_back(std::stoi(s.substr(p, c - p))); p = c + 1; }
  return v;
}
static std::vector<std::string> slist(const std::string& s) {
  std::vector<std::string> v; size_t p = 0;
  while (p < s.size()) { size_t c = s.find(',', p); if (c == std::string::npos) c = s.size(); v.push_back(s.substr(p, c - p)); p = c + 1; }
  return v;
}
struct Opt {
  std::vector<std::string> variants = {"r1"}, modes = {"fused"}, protos = {"LL128", "Simple"};
  std::vector<int> rows = {1024}, algos = {1}, graphs = {0}, bf16_reg = {0};
  int tp = 2, iters = 200, warmup = 20, tune[4] = {0, 0, 0, 0};
  bool check = false, bf16 = false, bf16_inherit = false;
  int reps = 1000, graph_reps = 1000, check_graphK = 10, skew_reps = 200, skew_us = 50;
  std::string tag, out;
  int input = 0;  // 0 bf16, 1 nvfp4 (xq local buffer), 2 nvfp4-window (xq written into input_slot)
} O;

static void usage() {
  fprintf(stderr,
    "fp4ar_bench --tp 2|4 --variant r1[,..]|all|none --rows 1024,.. --mode fused,xport,compute,empty --algo 1[,2]\n"
    "  --graph 0[,10] (calls per graph; 0 = eager) --iters 200 --warmup 20 --tune a,b,c,d\n"
    "  --check [--reps 1000 --graph-reps 1000 --check-graph 10 --skew-reps 200 --skew-us 50]\n"
    "  --input bf16|nvfp4|nvfp4-window (pre-quantized input, INPUT-FP4.md)\n"
    "  --bf16 [--bf16-protos LL128,Simple | --bf16-inherit] [--bf16-reg 0,1] --tag T --out file.jsonl\n");
}

// ---------------- shared host state ----------------
struct Barrier {
  std::mutex m; std::condition_variable cv; int n, cnt = 0, phase = 0;
  explicit Barrier(int n) : n(n) {}
  void wait() { std::unique_lock<std::mutex> l(m); int ph = phase; if (++cnt == n) { cnt = 0; phase++; cv.notify_all(); } else cv.wait(l, [&] { return phase != ph; }); }
};
static Barrier* BAR;
static ncclUniqueId ID_MAIN, ID_BF[4];
static std::vector<std::vector<float>> TIMES;
static std::vector<std::vector<uint16_t>> OUTS;
static std::vector<std::vector<double>> RVAL;  // per-rank scalar results for the current phase
static std::vector<std::string> BUS;
static std::vector<unsigned> CLK0, CLK1, CLKSM;
static std::string NODE, GPUNAME;
static int NCCLVER = 0;
static FILE* OUTF = nullptr;
static std::mutex PMX;
static const char* INPUT_NAME[3] = {"bf16", "nvfp4", "nvfp4-window"};
static std::vector<std::vector<float>> GS;  // per rank thread: gs[set * n + src] (all ranks compute identical values)

static void emit(const std::string& s) {
  std::lock_guard<std::mutex> l(PMX);
  printf("%s\n", s.c_str()); fflush(stdout);
  if (OUTF) { fprintf(OUTF, "%s\n", s.c_str()); fflush(OUTF); }
}
static std::string jarr_u(const std::vector<unsigned>& v) {
  std::string s = "["; for (size_t i = 0; i < v.size(); i++) s += (i ? "," : "") + std::to_string(v[i]); return s + "]";
}
static std::string jarr_f(const std::vector<double>& v) {
  std::string s = "["; char b[32];
  for (size_t i = 0; i < v.size(); i++) { snprintf(b, sizeof b, "%s%.3f", i ? "," : "", v[i]); s += b; }
  return s + "]";
}
static std::string jarr_s(const std::vector<std::string>& v) {
  std::string s = "["; for (size_t i = 0; i < v.size(); i++) s += (i ? ",\"" : "\"") + v[i] + "\""; return s + "]";
}
static std::string common_fields() {
  char b[512]; time_t t = time(nullptr); char ts[32]; strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%S", localtime(&t));
  const char* cvd = getenv("CUDA_VISIBLE_DEVICES");
  snprintf(b, sizeof b, "\"tag\":\"%s\",\"node\":\"%s\",\"gpu\":\"%s\",\"cvd\":\"%s\",\"nccl\":%d,\"time\":\"%s\"",
           O.tag.c_str(), NODE.c_str(), GPUNAME.c_str(), cvd ? cvd : "", NCCLVER, ts);
  std::string q = ",\"input\":\"" + std::string(INPUT_NAME[O.input]) + "\"";
  if (O.input) { std::vector<double> g(GS[0].begin(), GS[0].begin() + std::min(GS[0].size(), BUS.size())); char t[64]; q += ",\"gs\":["; for (size_t i = 0; i < g.size(); i++) { snprintf(t, sizeof t, "%s%.6e", i ? "," : "", g[i]); q += t; } q += "]"; }
  return std::string(b) + q + ",\"bus\":" + jarr_s(BUS) + ",\"clk_mem_start\":" + jarr_u(CLK0) + ",\"clk_mem_end\":" + jarr_u(CLK1) + ",\"clk_sm\":" + jarr_u(CLKSM);
}
static void stats(std::vector<float> v, double& med, double& p10, double& p90) {
  std::sort(v.begin(), v.end()); size_t k = v.size();
  med = v[k / 2]; p10 = v[k / 10]; p90 = v[(k * 9) / 10];
}
static const char* rule_proto(int tp, int rows) { return rows <= (tp == 2 ? 2048 : 4096) ? "LL128" : "Simple"; }

// ---------------- worker ----------------
struct Rank {
  int rank, n; cudaStream_t st; ncclComm_t comm; float* small; cudaDeviceProp prop;
  void align() { NK(ncclAllReduce(small, small, 1, ncclFloat, ncclSum, comm, st)); }
  void clocks(std::vector<unsigned>& dst) { dst[rank] = NVML->get(BUS[rank].c_str(), 2); }
};

// Times op (eager or K-call graph) with per-iteration rank alignment; rank 0 returns stats via fill().
static void time_op(Rank& R, int graphK, const std::function<void()>& op0, const std::function<void(double, double, double, std::vector<double>&)>& fill,
                    const std::function<void()>& pre = nullptr) {
  cudaGraphExec_t gx = nullptr;
  if (graphK > 0) {
    if (pre) pre();
    op0(); R.align(); CK(cudaStreamSynchronize(R.st)); BAR->wait();
    cudaGraph_t g; CK(cudaStreamBeginCapture(R.st, cudaStreamCaptureModeRelaxed));
    for (int k = 0; k < graphK; k++) { if (pre) pre(); op0(); }
    CK(cudaStreamEndCapture(R.st, &g)); CK(cudaGraphInstantiate(&gx, g, 0)); CK(cudaGraphDestroy(g));
  }
  auto op = [&]() { if (gx) CK(cudaGraphLaunch(gx, R.st)); else op0(); };
  auto pre1 = [&]() { if (!gx && pre) pre(); };
  std::vector<cudaEvent_t> ev(2 * O.iters);
  for (auto& e : ev) CK(cudaEventCreate(&e));
  R.clocks(CLK0);
  for (int it = 0; it < O.warmup; it++) { R.align(); pre1(); op(); }
  CK(cudaStreamSynchronize(R.st)); BAR->wait();
  for (int it = 0; it < O.iters; it++) { R.align(); pre1(); CK(cudaEventRecord(ev[2 * it], R.st)); op(); CK(cudaEventRecord(ev[2 * it + 1], R.st)); }
  CK(cudaStreamSynchronize(R.st)); CK(cudaGetLastError());
  R.clocks(CLK1); CLKSM[R.rank] = NVML->get(BUS[R.rank].c_str(), 1);
  TIMES[R.rank].resize(O.iters);
  for (int it = 0; it < O.iters; it++) { float ms; CK(cudaEventElapsedTime(&ms, ev[2 * it], ev[2 * it + 1])); TIMES[R.rank][it] = ms * 1000.f / (graphK > 0 ? graphK : 1); }
  for (auto& e : ev) CK(cudaEventDestroy(e));
  if (gx) CK(cudaGraphExecDestroy(gx));
  BAR->wait();
  if (R.rank == 0) {
    std::vector<float> mx(O.iters, 0.f); std::vector<double> rm(R.n);
    for (int it = 0; it < O.iters; it++) for (int r = 0; r < R.n; r++) mx[it] = std::max(mx[it], TIMES[r][it]);
    for (int r = 0; r < R.n; r++) { double a, b, c; stats(TIMES[r], a, b, c); rm[r] = a; }
    double med, p10, p90; stats(mx, med, p10, p90);
    fill(med, p10, p90, rm);
  }
  BAR->wait();
}

struct Bufs {
  __nv_bfloat16 *xin = nullptr, *rc = nullptr, *out = nullptr; float* r32 = nullptr; char* scratch = nullptr;
  ref::ErrAcc* e = nullptr; unsigned long long* ctr = nullptr; size_t Nmax = 0;
  char* xq = nullptr; char* dummy = nullptr; float* am = nullptr;  // NVFP4 records [set][src] (slot_bytes(rows) stride), copy sink, tensor amax
};

// Inputs for all sets and sources (stride N), same-codec references per set (for algo), FP32 sum for set 0.
static void make_inputs(Rank& R, Bufs& B, int rows, int algo, bool refs) {
  const size_t N = (size_t)rows * H; const int n = R.n;
  for (int set = 0; set < NSETS; set++)
    for (int s = 0; s < n; s++) {
      __nv_bfloat16* x = B.xin + ((size_t)set * n + s) * N; uint32_t seed = 1000u + 17u * rows + 7777u * set;
      ref::gen_input<<<1024, 256, 0, R.st>>>(x, s, rows, seed);
      ref::gen_outliers<<<1, 32, 0, R.st>>>(x, s, rows, seed);
    }
  const size_t SB_ = (size_t)rows * ref::QROW;
  if (O.input) {
    // calibration emulation outside timing: gs[r] = amax(partial_r)/2688, per set; every rank derives all ranks' gs
    CK(cudaMemsetAsync(B.am, 0, NSETS * n * sizeof(float), R.st));
    for (int i = 0; i < NSETS * n; i++) ref::tensor_amax<<<1024, 256, 0, R.st>>>(B.xin + (size_t)i * N, N, B.am + i);
    std::vector<float> am(NSETS * n); CK(cudaMemcpyAsync(am.data(), B.am, am.size() * sizeof(float), cudaMemcpyDeviceToHost, R.st)); CK(cudaStreamSynchronize(R.st));
    GS[R.rank].resize(NSETS * n);
    for (int i = 0; i < NSETS * n; i++) {
      GS[R.rank][i] = am[i] / 2688.f;
      ref::q_encode<<<2048, 256, 0, R.st>>>(B.xin + (size_t)i * N, B.xq + (size_t)i * SB_, rows, GS[R.rank][i]);
    }
  }
  if (refs && O.input) {  // reference = decode-sum of the same input records
    for (int set = 0; set < NSETS; set++) {
      ref::QSrc Q{};
      for (int s = 0; s < n; s++) { Q.recs[s] = B.xq + ((size_t)set * n + s) * SB_; Q.gs[s] = GS[R.rank][set * n + s]; }
      if (algo == FP4AR_ONESHOT) ref::ref_q_oneshot<<<2048, 256, 0, R.st>>>(Q, n, rows, B.rc + set * N);
      else ref::ref_q_twoshot<<<2048, 256, 0, R.st>>>(Q, n, rows, B.rc + set * N);
    }
    ref::ref_fp32<<<2048, 256, 0, R.st>>>(B.xin, n, N, B.r32);
  } else if (refs) {
    for (int set = 0; set < NSETS; set++) {
      for (int s = 0; s < n; s++) ref::ref_encode<<<2048, 256, 0, R.st>>>(B.xin + ((size_t)set * n + s) * N, B.scratch + s * ref::slot_bytes(rows), rows);
      if (algo == FP4AR_ONESHOT) ref::ref_oneshot<<<2048, 256, 0, R.st>>>(B.scratch, n, rows, B.rc + set * N);
      else ref::ref_twoshot<<<2048, 256, 0, R.st>>>(B.scratch, n, rows, B.rc + set * N);
    }
    ref::ref_fp32<<<2048, 256, 0, R.st>>>(B.xin, n, N, B.r32);
  }
  CK(cudaStreamSynchronize(R.st)); CK(cudaGetLastError());
}

// Points a at the call's NVFP4 records; window mode runs the "GEMM emulation" copy into input_slot() first (or into a sink).
static void set_input(Rank& R, Bufs& B, const Fp4ArVariant* V, Fp4ArCtx& ctx, Fp4ArArgs& a, int rows, int set, bool sink) {
  if (!O.input) { a.xq = nullptr; a.xq_in_window = 0; return; }
  const size_t SB_ = (size_t)rows * ref::QROW;
  for (int r = 0; r < R.n; r++) a.gs[r] = GS[R.rank][set * R.n + r];
  const char* src = B.xq + ((size_t)set * R.n + R.rank) * SB_;
  if (O.input == 1) { a.xq = (const uint8_t*)src; a.xq_in_window = 0; return; }
  uint8_t* dst = sink ? (uint8_t*)B.dummy : V->input_slot(&ctx, rows);
  ref::copy16<<<2 * R.prop.multiProcessorCount, 512, 0, R.st>>>((const uint4*)src, (uint4*)dst, SB_ / 16);
  if (!sink) { a.xq = dst; a.xq_in_window = 1; }
}

// Correctness gate for one (variant, algo, rows); emits one JSON line.
static bool run_check(Rank& R, Bufs& B, const Fp4ArVariant* V, Fp4ArCtx& ctx, int rows, int algo) {
  const size_t N = (size_t)rows * H; const int n = R.n, me = R.rank;
  Fp4ArArgs a{}; a.out = B.out; a.rows = rows; a.hidden = H; a.mode = FP4AR_FUSED; a.algo = algo;
  for (int i = 0; i < 4; i++) a.tune[i] = O.tune[i];
  // negative controls (must FAIL): FP4AR_NEGCTL=stale feeds the next set's input, =flip corrupts one value on the last rank
  const char* nc = getenv("FP4AR_NEGCTL");
  const int negctl = !nc ? 0 : !strcmp(nc, "stale") ? 1 : !strcmp(nc, "flip") ? 2 : 0;
  auto call = [&](int set) {
    const int in = negctl == 1 ? (set + 1) % NSETS : set;
    a.x = B.xin + ((size_t)in * n + me) * N;
    set_input(R, B, V, ctx, a, rows, in, false);
    int rc = V->launch(&ctx, &a, R.st);
    if (rc) { fprintf(stderr, "launch rc=%d variant %s\n", rc, V->name); _exit(1); }
    if (negctl == 2 && me == n - 1) ref::flip_bit<<<1, 1, 0, R.st>>>((uint16_t*)B.out + 12345 % N);
  };
  auto cmp = [&](int set) {
    ref::cmp_kernel<<<1024, 256, 0, R.st>>>((const uint4*)B.out, (const uint4*)(B.rc + set * N), N / 8, B.ctr);
    ref::mark_call<<<1, 1, 0, R.st>>>(B.ctr);
  };
  auto ctr_reset = [&]() { CK(cudaMemsetAsync(B.ctr, 0, 4 * sizeof(unsigned long long), R.st)); };
  auto ctr_get = [&](unsigned long long* h) { CK(cudaMemcpyAsync(h, B.ctr, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost, R.st)); CK(cudaStreamSynchronize(R.st)); };
  // identical-across-ranks test on the current output
  auto ranks_same = [&]() -> int {
    OUTS[me].resize(N); CK(cudaStreamSynchronize(R.st)); CK(cudaMemcpy(OUTS[me].data(), B.out, N * 2, cudaMemcpyDeviceToHost));
    BAR->wait(); int same = 1;
    for (int r = 1; r < n; r++) same &= memcmp(OUTS[0].data(), OUTS[r].data(), N * 2) == 0;
    BAR->wait(); return same;
  };
  std::vector<double>& rv = RVAL[me]; rv.assign(16, 0.0);
  unsigned long long h[4];
  // 1. first call + accuracy on set 0
  CK(cudaMemsetAsync(B.out, 0, N * 2, R.st)); R.align(); call(0);
  CK(cudaMemsetAsync(B.e, 0, sizeof(ref::ErrAcc), R.st));
  ref::err_kernel<<<1024, 256, 0, R.st>>>(B.out, B.r32, B.rc, N, B.e);
  ref::ErrAcc e; CK(cudaMemcpyAsync(&e, B.e, sizeof e, cudaMemcpyDeviceToHost, R.st)); CK(cudaStreamSynchronize(R.st));
  float ma; memcpy(&ma, &e.maxabs, 4);
  rv[0] = (double)e.mism; rv[1] = sqrt(e.dsq / e.rsq); rv[2] = sqrt(e.bsq / e.rsq); rv[3] = ma;
  if (getenv("FP4AR_DEBUG") && me == 0) {  // print first differing elements of the first call
    std::vector<uint16_t> o(N), c(N); CK(cudaMemcpy(o.data(), B.out, N * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(c.data(), B.rc, N * 2, cudaMemcpyDeviceToHost));
    int shown = 0;
    for (size_t i = 0; i < N && shown < 24; i++) if (o[i] != c[i]) {
      uint32_t a = (uint32_t)o[i] << 16, b = (uint32_t)c[i] << 16; float fa, fb; memcpy(&fa, &a, 4); memcpy(&fb, &b, 4);
      fprintf(stderr, "# diff row %zu col %zu out %g ref %g\n", i / H, i % H, fa, fb); shown++;
    }
  }
  const int same_first = ranks_same();
  // 2. back-to-back eager, no intermediate host sync or compare; final output vs its set
  CK(cudaMemsetAsync(B.out, 0, N * 2, R.st)); R.align(); CK(cudaStreamSynchronize(R.st)); BAR->wait();
  for (int k = 0; k < O.reps; k++) call(k % NSETS);
  ctr_reset(); cmp((O.reps - 1) % NSETS); ctr_get(h); rv[4] = (double)h[0];
  const int same_b2b = ranks_same();
  // 3. eager with per-call compare
  ctr_reset(); R.align();
  for (int k = 0; k < O.reps; k++) { call(k % NSETS); cmp(k % NSETS); }
  ctr_get(h); rv[5] = (double)h[0]; rv[6] = (double)h[1];
  // 4. CUDA graph: check_graphK calls (+ compares) per graph, graph_reps replays
  {
    R.align(); CK(cudaStreamSynchronize(R.st)); BAR->wait();
    cudaGraph_t g; cudaGraphExec_t gx;
    CK(cudaStreamBeginCapture(R.st, cudaStreamCaptureModeRelaxed));
    for (int k = 0; k < O.check_graphK; k++) { call(k % NSETS); cmp(k % NSETS); }
    CK(cudaStreamEndCapture(R.st, &g)); CK(cudaGraphInstantiate(&gx, g, 0)); CK(cudaGraphDestroy(g));
    ctr_reset();
    for (int k = 0; k < O.graph_reps; k++) CK(cudaGraphLaunch(gx, R.st));
    ctr_get(h); rv[7] = (double)h[0]; rv[8] = (double)h[1];
    CK(cudaGraphExecDestroy(gx));
  }
  const int same_graph = ranks_same();
  // 5. rank skew: one rank (rotating) spins skew_us before the call
  ctr_reset();
  for (int k = 0; k < O.skew_reps; k++) {
    R.align();
    if (k % n == me) ref::spin_kernel<<<1, 1, 0, R.st>>>((unsigned long long)O.skew_us * 1000ull);
    call(k % NSETS); cmp(k % NSETS);
  }
  ctr_get(h); rv[9] = (double)h[0]; rv[10] = (double)h[1];
  CK(cudaStreamSynchronize(R.st)); CK(cudaGetLastError());
  BAR->wait();
  bool pass = true;
  if (me == 0) {
    double tot[11] = {0};
    for (int r = 0; r < n; r++) for (int i = 0; i < 11; i++) tot[i] = (i >= 1 && i <= 3) ? std::max(tot[i], RVAL[r][i]) : tot[i] + RVAL[r][i];
    pass = tot[0] == 0 && tot[4] == 0 && tot[5] == 0 && tot[7] == 0 && tot[9] == 0 && same_first && same_b2b && same_graph;
    char b[1024];
    snprintf(b, sizeof b,
      "{\"kind\":\"check\",\"variant\":\"%s\",\"algo\":%d,\"tp\":%d,\"rows\":%d,\"hidden\":%d,\"tune\":[%d,%d,%d,%d],\"pass\":%s,"
      "\"mism_first\":%.0f,\"ranks_identical\":[%d,%d,%d],\"b2b_calls\":%d,\"mism_b2b_final\":%.0f,"
      "\"percall_calls\":%d,\"mism_percall\":%.0f,\"bad_calls_percall\":%.0f,"
      "\"graph_calls\":%d,\"mism_graph\":%.0f,\"bad_calls_graph\":%.0f,"
      "\"skew_calls\":%d,\"skew_us\":%d,\"mism_skew\":%.0f,\"bad_calls_skew\":%.0f,"
      "\"rel_l2_vs_fp32\":%.6f,\"codecref_rel_l2_vs_fp32\":%.6f,\"max_abs_vs_fp32\":%.4f,",
      V->name, algo + 1, n, rows, H, O.tune[0], O.tune[1], O.tune[2], O.tune[3], pass ? "true" : "false",
      tot[0], same_first, same_b2b, same_graph, O.reps, tot[4], O.reps, tot[5], tot[6],
      O.check_graphK * O.graph_reps, tot[7], tot[8], O.skew_reps, O.skew_us, tot[9], tot[10], tot[1], tot[2], tot[3]);
    emit(std::string(b) + common_fields() + "}");
    fprintf(stderr, "# CHECK %s %s algo=%d tp=%d rows=%d\n", pass ? "PASS" : "FAIL", V->name, algo + 1, n, rows);
  }
  BAR->wait();
  return pass;
}

static int FAILS = 0;

static void worker(int rank) {
  Rank R; R.rank = rank; R.n = O.tp;
  CK(cudaSetDevice(rank));
  CK(cudaGetDeviceProperties(&R.prop, rank));
  { char bus[32]; CK(cudaDeviceGetPCIBusId(bus, sizeof bus, rank)); BUS[rank] = bus; }
  if (rank == 0) GPUNAME = R.prop.name;
  NK(ncclCommInitRank(&R.comm, R.n, ID_MAIN, rank));
  CK(cudaStreamCreateWithFlags(&R.st, cudaStreamNonBlocking));
  CK(cudaMalloc(&R.small, 64)); CK(cudaMemset(R.small, 0, 64));
  const int maxRows = *std::max_element(O.rows.begin(), O.rows.end());
  Bufs B; B.Nmax = (size_t)maxRows * H; const int n = R.n;
  CK(cudaMalloc(&B.xin, (size_t)NSETS * n * B.Nmax * 2)); CK(cudaMalloc(&B.out, B.Nmax * 2));
  if (O.check) {
    CK(cudaMalloc(&B.rc, NSETS * B.Nmax * 2)); CK(cudaMalloc(&B.r32, B.Nmax * 4));
    CK(cudaMalloc(&B.scratch, n * ref::slot_bytes(maxRows))); CK(cudaMalloc(&B.e, sizeof(ref::ErrAcc)));
    CK(cudaMalloc(&B.ctr, 4 * sizeof(unsigned long long)));
  }
  if (O.input) {
    CK(cudaMalloc(&B.xq, (size_t)NSETS * n * ref::slot_bytes(maxRows))); CK(cudaMalloc(&B.am, NSETS * n * sizeof(float)));
    CK(cudaMalloc(&B.dummy, ref::slot_bytes(maxRows)));
  }
  auto vars = all_variants();
  std::vector<const Fp4ArVariant*> run;
  for (auto& name : O.variants) {
    if (name == "none") continue;
    for (auto* v : vars)
      if (name == "all" || name == v->name) {
        if (O.input == 2 && !v->input_slot) { if (rank == 0) fprintf(stderr, "# skip %s: no input_slot (nvfp4-window)\n", v->name); continue; }
        run.push_back(v);
      }
  }
  for (const Fp4ArVariant* V : run) {
    // symmetric window sized by the variant
    Fp4ArCtx ctx{}; ctx.rank = rank; ctx.nranks = n; ctx.device = rank; ctx.sm_count = R.prop.multiProcessorCount; ctx.comm = R.comm;
    ctx.win_bytes = V->window_bytes(maxRows, H, n);
    if (ctx.win_bytes == 0) { if (rank == 0) fprintf(stderr, "# skip %s: window_bytes=0\n", V->name); continue; }
    const size_t alloc = (ctx.win_bytes + (1 << 21) - 1) / (1 << 21) * (1 << 21);
    void* wbase; NK(ncclMemAlloc(&wbase, alloc)); CK(cudaMemset(wbase, 0, alloc)); CK(cudaDeviceSynchronize());
    NK(ncclCommWindowRegister(R.comm, wbase, alloc, &ctx.win, NCCL_WIN_COLL_SYMMETRIC));
    ncclTeam_t lsa = ncclTeamLsa(R.comm);
    if (lsa.nRanks != n) { fprintf(stderr, "lsa nRanks=%d != tp=%d\n", lsa.nRanks, n); _exit(1); }
    for (int r = 0; r < n; r++) { void* q; NK(ncclGetLsaDevicePointer(ctx.win, 0, r, &q)); ctx.peer[r] = (char*)q; }
    if (int rc = V->init(&ctx, maxRows, H)) { fprintf(stderr, "init %s rc=%d\n", V->name, rc); _exit(1); }
    BAR->wait();
    for (int algoCli : O.algos) {
      const int algo = algoCli == 2 ? FP4AR_TWOSHOT : FP4AR_ONESHOT;
      if (algo == FP4AR_TWOSHOT && !V->supports_twoshot) { if (rank == 0) fprintf(stderr, "# skip %s: no two-shot\n", V->name); continue; }
      for (int rows : O.rows) {
        const size_t N = (size_t)rows * H;
        make_inputs(R, B, rows, algo, O.check);
        BAR->wait();
        if (O.check && !run_check(R, B, V, ctx, rows, algo)) FAILS++;
        if (O.iters <= 0) continue;
        for (auto& ms : O.modes) {
          const int mode = ms == "fused" ? FP4AR_FUSED : ms == "xport" ? FP4AR_XPORT : ms == "compute" ? FP4AR_COMPUTE : -1;
          Fp4ArArgs a{}; a.x = B.xin + (size_t)rank * N; a.out = B.out; a.rows = rows; a.hidden = H; a.mode = mode < 0 ? 0 : mode; a.algo = algo;
          for (int i = 0; i < 4; i++) a.tune[i] = O.tune[i];
          set_input(R, B, V, ctx, a, rows, 0, true);  // nvfp4: fixed xq; window: refreshed per call by pre
          auto op0 = [&]() {
            if (mode < 0) { ref::empty_kernel<<<R.prop.multiProcessorCount, 256, 0, R.st>>>(); return; }
            if (int rc = V->launch(&ctx, &a, R.st)) { fprintf(stderr, "launch rc=%d\n", rc); _exit(1); }
          };
          // window mode: the GEMM-emulation copy precedes every call and is excluded (eager: outside the events;
          // graph: K x (copy + AR) minus K x copy-to-sink, both timed with the same protocol)
          std::function<void()> pre = nullptr;
          if (O.input == 2 && mode >= 0) pre = [&]() { set_input(R, B, V, ctx, a, rows, 0, false); };
          for (int g : O.graphs) {
            double copy_med = 0;
            if (pre && g > 0)
              time_op(R, g, [&]() { set_input(R, B, V, ctx, a, rows, 0, true); }, [&](double med, double, double, std::vector<double>&) { copy_med = med; });
            time_op(R, g, op0, [&](double med, double p10, double p90, std::vector<double>& rm) {
              if (copy_med > 0) { med -= copy_med; p10 -= copy_med; p90 -= copy_med; for (auto& x : rm) x -= copy_med; }
              char b[512];
              snprintf(b, sizeof b, "{\"kind\":\"timing\",\"variant\":\"%s\",\"mode\":\"%s\",\"algo\":%d,\"tp\":%d,\"rows\":%d,\"hidden\":%d,\"graphK\":%d,"
                       "\"iters\":%d,\"warmup\":%d,\"tune\":[%d,%d,%d,%d],\"med_us\":%.3f,\"p10_us\":%.3f,\"p90_us\":%.3f,",
                       V->name, ms.c_str(), algo + 1, n, rows, H, g, O.iters, O.warmup, O.tune[0], O.tune[1], O.tune[2], O.tune[3], med, p10, p90);
              char c[64]; snprintf(c, sizeof c, "\"copy_us\":%.3f,", copy_med);
              emit(std::string(b) + c + "\"rank_med_us\":" + jarr_f(rm) + "," + common_fields() + "}");
            }, pre);
          }
        }
      }
    }
    CK(cudaStreamSynchronize(R.st)); BAR->wait();
    V->fini(&ctx);
    NK(ncclCommWindowDeregister(R.comm, ctx.win));
    NK(ncclMemFree(wbase));
    BAR->wait();
  }
  // tuned BF16 NCCL baseline: one comm per protocol (NCCL_PROTO is read at comm init), out-of-place
  if (O.bf16) {
    std::vector<std::string> protos = O.bf16_inherit ? std::vector<std::string>{"inherit"} : O.protos;
    __nv_bfloat16* bo; CK(cudaMalloc(&bo, B.Nmax * 2));
    for (size_t pi = 0; pi < protos.size(); pi++) {
      BAR->wait();
      if (rank == 0 && !O.bf16_inherit) {
        if (protos[pi] == "default") unsetenv("NCCL_PROTO"); else setenv("NCCL_PROTO", protos[pi].c_str(), 1);
      }
      BAR->wait();
      ncclComm_t cb; NK(ncclCommInitRank(&cb, n, ID_BF[pi], rank));
      BAR->wait();
      const char* envp = getenv("NCCL_PROTO"); std::string eff = envp ? envp : "default";
      // reg=1: ncclMemAlloc (cuMem) in/out buffers registered with ncclCommRegister (registered out-of-place)
      __nv_bfloat16 *rin = nullptr, *rout = nullptr; void *hin = nullptr, *hout = nullptr;
      if (std::find(O.bf16_reg.begin(), O.bf16_reg.end(), 1) != O.bf16_reg.end()) {
        NK(ncclMemAlloc((void**)&rin, B.Nmax * 2)); NK(ncclMemAlloc((void**)&rout, B.Nmax * 2));
        NK(ncclCommRegister(cb, rin, B.Nmax * 2, &hin)); NK(ncclCommRegister(cb, rout, B.Nmax * 2, &hout));
      }
      for (int rows : O.rows) for (int reg : O.bf16_reg) {
        const size_t N = (size_t)rows * H;
        make_inputs(R, B, rows, 0, false);
        const __nv_bfloat16* x = B.xin + (size_t)rank * N;
        __nv_bfloat16* dst = bo;
        if (reg) { CK(cudaMemcpyAsync(rin, x, N * 2, cudaMemcpyDeviceToDevice, R.st)); CK(cudaStreamSynchronize(R.st)); x = rin; dst = rout; }
        auto op0 = [&]() { NK(ncclAllReduce(x, dst, N, ncclBfloat16, ncclSum, cb, R.st)); };
        for (int g : O.graphs) {
          time_op(R, g, op0, [&](double med, double p10, double p90, std::vector<double>& rm) {
            char b[512]; const char* al = getenv("NCCL_ALGO"); const char* mc = getenv("NCCL_MIN_NCHANNELS");
            snprintf(b, sizeof b, "{\"kind\":\"timing\",\"variant\":\"nccl_bf16\",\"mode\":\"bf16\",\"proto\":\"%s\",\"nccl_algo\":\"%s\",\"min_nch\":\"%s\","
                     "\"rule_proto\":\"%s\",\"algo\":0,\"tp\":%d,\"rows\":%d,\"hidden\":%d,\"graphK\":%d,\"iters\":%d,\"warmup\":%d,"
                     "\"reg\":%d,\"med_us\":%.3f,\"p10_us\":%.3f,\"p90_us\":%.3f,",
                     eff.c_str(), al ? al : "", mc ? mc : "", rule_proto(n, rows), n, rows, H, g, O.iters, O.warmup, reg, med, p10, p90);
            emit(std::string(b) + "\"rank_med_us\":" + jarr_f(rm) + "," + common_fields() + "}");
          });
        }
      }
      CK(cudaStreamSynchronize(R.st));
      // graph-captured NCCL comms hung at destroy in round 1: leak them, the process exits via _exit
      bool graphed = std::any_of(O.graphs.begin(), O.graphs.end(), [](int g) { return g > 0; });
      if (!graphed) {
        if (hin) { NK(ncclCommDeregister(cb, hin)); NK(ncclCommDeregister(cb, hout)); NK(ncclMemFree(rin)); NK(ncclMemFree(rout)); }
        NK(ncclCommDestroy(cb));
      }
    }
    BAR->wait();
  }
  CK(cudaDeviceSynchronize());
  BAR->wait();
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto nx = [&]() { if (i + 1 >= argc) { usage(); exit(2); } return std::string(argv[++i]); };
    if (a == "--tp" || a == "--n") O.tp = std::stoi(nx());
    else if (a == "--variant") O.variants = slist(nx());
    else if (a == "--rows") O.rows = ilist(nx());
    else if (a == "--mode") O.modes = slist(nx());
    else if (a == "--algo") O.algos = ilist(nx());
    else if (a == "--graph") O.graphs = ilist(nx());
    else if (a == "--iters") O.iters = std::stoi(nx());
    else if (a == "--warmup") O.warmup = std::stoi(nx());
    else if (a == "--tune") { auto t = ilist(nx()); for (size_t k = 0; k < t.size() && k < 4; k++) O.tune[k] = t[k]; }
    else if (a == "--check") O.check = true;
    else if (a == "--reps") O.reps = std::stoi(nx());
    else if (a == "--graph-reps") O.graph_reps = std::stoi(nx());
    else if (a == "--check-graph") O.check_graphK = std::stoi(nx());
    else if (a == "--skew-reps") O.skew_reps = std::stoi(nx());
    else if (a == "--skew-us") O.skew_us = std::stoi(nx());
    else if (a == "--bf16") O.bf16 = true;
    else if (a == "--bf16-protos") O.protos = slist(nx());
    else if (a == "--bf16-inherit") O.bf16_inherit = true;
    else if (a == "--bf16-reg") O.bf16_reg = ilist(nx());
    else if (a == "--tag") O.tag = nx();
    else if (a == "--input") { std::string v = nx(); O.input = v == "bf16" ? 0 : v == "nvfp4" ? 1 : v == "nvfp4-window" ? 2 : -1; if (O.input < 0) { usage(); return 2; } }
    else if (a == "--out") O.out = nx();
    else if (a == "--list") { for (auto* v : all_variants()) printf("%s twoshot=%d\n", v->name, (int)v->supports_twoshot); return 0; }
    else { usage(); return 2; }
  }
  for (auto& nm : O.variants) {  // an unknown --variant must fail loudly, not silently time nothing
    bool ok = nm == "none" || nm == "all";
    for (auto* v : all_variants()) ok = ok || nm == v->name;
    if (!ok) { fprintf(stderr, "unknown variant '%s' (not linked in this binary; see --list)\n", nm.c_str()); return 2; }
  }
  int ndev = 0; CK(cudaGetDeviceCount(&ndev));
  if (O.tp < 2 || O.tp > FP4AR_MAX_RANKS || O.tp > ndev) { fprintf(stderr, "tp=%d but %d visible GPUs\n", O.tp, ndev); return 1; }
  if (O.protos.size() > 4) { fprintf(stderr, "at most 4 bf16 protocols\n"); return 1; }
  char hn[256] = {0}; gethostname(hn, sizeof hn - 1); NODE = hn;
  NVML = new Nvml();
  NK(ncclGetVersion(&NCCLVER));
  if (!O.out.empty()) OUTF = fopen(O.out.c_str(), "a");
  NK(ncclGetUniqueId(&ID_MAIN));
  for (int i = 0; i < 4; i++) NK(ncclGetUniqueId(&ID_BF[i]));
  printf("# fp4ar_bench node=%s tp=%d nccl=%d nvml=%d NCCL_ALGO=%s NCCL_PROTO=%s NCCL_MIN_NCHANNELS=%s NCCL_MAX_NCHANNELS=%s\n", hn, O.tp, NCCLVER, (int)NVML->ok,
         getenv("NCCL_ALGO") ? getenv("NCCL_ALGO") : "-", getenv("NCCL_PROTO") ? getenv("NCCL_PROTO") : "-",
         getenv("NCCL_MIN_NCHANNELS") ? getenv("NCCL_MIN_NCHANNELS") : "-", getenv("NCCL_MAX_NCHANNELS") ? getenv("NCCL_MAX_NCHANNELS") : "-");
  fflush(stdout);
  Barrier b(O.tp); BAR = &b;
  TIMES.resize(O.tp); OUTS.resize(O.tp); GS.resize(O.tp); RVAL.resize(O.tp); BUS.resize(O.tp); CLK0.assign(O.tp, 0); CLK1.assign(O.tp, 0); CLKSM.assign(O.tp, 0);
  std::vector<std::thread> th;
  for (int r = 0; r < O.tp; r++) th.emplace_back(worker, r);
  for (auto& t : th) t.join();
  printf("# done fails=%d\n", FAILS); fflush(stdout);
  if (OUTF) fclose(OUTF);
  for (int d = 0; d < O.tp; d++) { cudaSetDevice(d); cudaDeviceSynchronize(); cudaProfilerStop(); }  // flush CUPTI (nsys) before _exit
  _exit(FAILS ? 3 : 0);  // skip NCCL/driver teardown (round-1 hang)
}
