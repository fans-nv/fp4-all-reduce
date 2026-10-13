// fp4-allreduce: host interface (Fp4ArCtx/Fp4ArArgs/Fp4ArVariant) shared by the harness and every variant.
// Shared host interface between the round-2 harness and every fused NVFP4 AllReduce variant.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nccl.h>

constexpr int FP4AR_MAX_RANKS = 8;

enum Fp4ArMode : int {
  FP4AR_FUSED = 0,      // full operator
  FP4AR_XPORT = 1,      // same bytes and sync, codec skipped
  FP4AR_COMPUTE = 2,    // same codec work, no peer traffic or waits
};

enum Fp4ArAlgo : int { FP4AR_ONESHOT = 0, FP4AR_TWOSHOT = 1 };

// Owned by the harness; one per rank (one host thread per GPU).
struct Fp4ArCtx {
  int rank, nranks, device, sm_count;
  ncclComm_t comm;
  ncclWindow_t win;                  // NCCL_WIN_COLL_SYMMETRIC window from ncclMemAlloc
  char* peer[FP4AR_MAX_RANKS];       // ncclGetLsaDevicePointer(win, 0, r, &peer[r]); peer[rank] is local
  size_t win_bytes;                  // as requested by the variant's window_bytes()
  void* priv;                        // variant-private state
};

struct Fp4ArArgs {
  const __nv_bfloat16* x;            // [rows, hidden] local partial
  __nv_bfloat16* out;                // [rows, hidden] reduced result (same on all ranks)
  int rows, hidden;                  // hidden == 6144 for M3
  int mode;                          // Fp4ArMode
  int algo;                          // Fp4ArAlgo
  int tune[4];                       // variant-defined knobs (stages, CTAs/SM, lag, ...); 0 = default
  // Pre-quantized input (GEMM epilogue emits NVFP4; see INPUT-FP4.md). When xq != nullptr the kernel skips encode and x is unused.
  const uint8_t* xq;                 // [rows] x 3456 B rows: 3072 E2M1 bytes then 384 E4M3 scales (16 B aligned, no row scale)
  int xq_in_window;                  // 1: xq is the slot returned by input_slot() (producer wrote straight into the window)
  float gs[FP4AR_MAX_RANKS];         // static per-tensor scale of each rank's partial (calibrated, known to all ranks; never transferred)
};

// Each variant exports one of these via fp4ar_register_<name>().
struct Fp4ArVariant {
  const char* name;
  bool supports_twoshot;
  size_t (*window_bytes)(int max_rows, int hidden, int nranks);
  int (*init)(Fp4ArCtx* ctx, int max_rows, int hidden);   // collective: all ranks call; may zero/seed the window
  int (*launch)(Fp4ArCtx* ctx, const Fp4ArArgs* a, cudaStream_t s);  // graph-capturable, no host sync
  void (*fini)(Fp4ArCtx* ctx);
  // Optional (window input mode): where the producer must write the next call's NVFP4 input; nullptr if unsupported.
  uint8_t* (*input_slot)(Fp4ArCtx* ctx, int rows);
};
