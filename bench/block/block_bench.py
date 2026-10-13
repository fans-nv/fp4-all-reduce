# M3 TP2 AR + residual + RMSNorm (+NVFP4) block timing; torchrun --nproc-per-node 2 block_bench.py
import os, sys, json, time, statistics
import torch, torch.distributed as dist

H = 6144; EPS = 1e-6
ROWS = [int(r) for r in os.environ.get("ROWS", "1024,2048,4096,8192,16384").split(",")]
SETS = os.environ.get("SETS", "norm,nccl,fi").split(",")
NREP = 10; WARM = 5; TRIALS = int(os.environ.get("TRIALS", "40"))
OUT = os.environ.get("OUT", "/tmp/block.jsonl")
TAG = os.environ.get("TAG", "")

dist.init_process_group("nccl")
rank = dist.get_rank(); ws = dist.get_world_size()
torch.cuda.set_device(rank % torch.cuda.device_count()); dev = torch.device("cuda")
import vllm._custom_ops as ops
from flashinfer import add_rmsnorm_fp4quant
import flashinfer.comm as fic

def fp4_bufs(m):
    return ops.create_fp4_output_tensors(m, H, dev, True)

def timeit(fn):
    for _ in range(WARM): fn()
    torch.cuda.synchronize(); dist.barrier()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(NREP): fn()
    torch.cuda.synchronize(); dist.barrier()
    for _ in range(3): g.replay()
    ts = []
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    for _ in range(TRIALS):
        torch.cuda.synchronize(); dist.barrier()
        e0.record(); g.replay(); e1.record(); e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / NREP)
    med = torch.tensor([statistics.median(ts)], device=dev)
    mx = med.clone(); dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    del g
    return mx.item(), med.item()

def emit(name, m, fn):
    try:
        mx, me = timeit(fn)
        rec = dict(tag=TAG, variant=name, rows=m, us=round(mx, 2), us_rank=round(me, 2),
                   proto=os.environ.get("NCCL_PROTO", ""), host=os.uname()[1])
    except Exception as e:
        rec = dict(tag=TAG, variant=name, rows=m, err=repr(e)[:300])
        torch.cuda.synchronize()
    if rank == 0:
        print(json.dumps(rec), flush=True)
        with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")

torch.manual_seed(1234 + rank)
gs = torch.tensor([1.0], device=dev, dtype=torch.float32)
w = torch.ones(H, device=dev, dtype=torch.bfloat16)
fiws = None
if "fi" in SETS:
    try:
        kw = {}
        try:
            from flashinfer.comm.mnnvl import TorchDistBackend  # noqa
            kw["comm_backend"] = TorchDistBackend(group=dist.group.WORLD)
        except Exception:
            pass
        fiws = fic.create_allreduce_fusion_workspace(backend=os.environ.get("FIB","trtllm"), world_size=ws, rank=rank,
                                                     max_token_num=max(ROWS), hidden_dim=H,
                                                     dtype=torch.bfloat16, **({"force_oneshot_support": True} if os.environ.get("FIB")=="mnnvl" else {}), **kw)
    except Exception as e:
        if rank == 0: print("FI workspace failed", repr(e)[:400], flush=True)

for m in ROWS:
    x = torch.randn(m, H, device=dev, dtype=torch.bfloat16)
    res = torch.randn(m, H, device=dev, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    q, sf = fp4_bufs(m)
    if "norm" in SETS:
        emit("norm_bf16_fused_add_rms_norm", m, lambda: ops.fused_add_rms_norm(x, res, w, EPS))
        emit("norm_fp4_fi_add_rmsnorm_fp4quant", m, lambda: add_rmsnorm_fp4quant(
            x, res, w, y_fp4=q.view(torch.float4_e2m1fn_x2), block_scale=sf.view(torch.float8_e4m3fn),
            global_scale=gs, eps=EPS, block_size=16, scale_format="e4m3", is_sf_swizzled_layout=True))
        def nq():
            ops.fused_add_rms_norm(x, res, w, EPS); ops.scaled_fp4_quant(x, gs)
        emit("norm_fp4_vllm_add_rms_norm+scaled_fp4_quant", m, nq)
        emit("quant_alone_scaled_fp4_quant", m, lambda: ops.scaled_fp4_quant(x, gs))
    if "nccl" in SETS:
        emit("nccl_ar", m, lambda: dist.all_reduce(x))
        def b_bf16():
            dist.all_reduce(x); ops.fused_add_rms_norm(x, res, w, EPS)
        emit("nccl_ar+norm_bf16", m, b_bf16)
        def b_fp4():
            dist.all_reduce(x)
            add_rmsnorm_fp4quant(x, res, w, y_fp4=q.view(torch.float4_e2m1fn_x2),
                                 block_scale=sf.view(torch.float8_e4m3fn), global_scale=gs, eps=EPS,
                                 block_size=16, scale_format="e4m3", is_sf_swizzled_layout=True)
        emit("nccl_ar+norm_fp4", m, b_fp4)
        x.normal_(); res.normal_()
    if "fi" in SETS and fiws is not None:
        for one in (True, False):
            sfx = "1shot" if one else "2shot"
            def fb(one=one):
                fic.allreduce_fusion(input=x, workspace=fiws, pattern=fic.AllReduceFusionPattern.kARResidualRMSNorm,
                                     residual_in=res, residual_out=res, norm_out=y, rms_gamma=w, rms_eps=EPS,
                                     use_oneshot=one, launch_with_pdl=True, trigger_completion_at_end=True, fp32_acc=True)
            emit(f"fi_{os.environ.get('FIB','trtllm')}_fused_bf16_{sfx}", m, fb)
            def ff(one=one):
                fic.allreduce_fusion(input=x, workspace=fiws, pattern=fic.AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
                                     residual_in=res, residual_out=res, norm_out=y, rms_gamma=w, rms_eps=EPS,
                                     quant_out=q, scale_out=sf, layout_code=fic.QuantizationSFLayout.SWIZZLED_128x4,
                                     scale_factor=gs, use_oneshot=one, launch_with_pdl=True,
                                     trigger_completion_at_end=True, fp32_acc=True)
            emit(f"fi_{os.environ.get('FIB','trtllm')}_fused_fp4_{sfx}", m, ff)
    if "normrot" in SETS:
        R = int(os.environ.get("ROT", "4"))
        xs = [torch.randn(m, H, device=dev, dtype=torch.bfloat16) for _ in range(R)]
        rs = [torch.randn(m, H, device=dev, dtype=torch.bfloat16) for _ in range(R)]
        cnt = [0]
        def nr():
            i = cnt[0] % R; cnt[0] += 1; ops.fused_add_rms_norm(xs[i], rs[i], w, EPS)
        emit(f"normrot{R}_bf16_fused_add_rms_norm", m, nr)
        def nqr():
            i = cnt[0] % R; cnt[0] += 1
            ops.fused_add_rms_norm(xs[i], rs[i], w, EPS); ops.scaled_fp4_quant(xs[i], gs)
        emit(f"normrot{R}_fp4_vllm_add_rms_norm+scaled_fp4_quant", m, nqr)
        def cr():
            i = cnt[0] % R; cnt[0] += 1; xs[i].copy_(rs[i])
        emit(f"copyrot{R}_bf16_copy", m, cr)
        del xs, rs
    del x, res, y, q, sf; torch.cuda.empty_cache()
dist.barrier(); dist.destroy_process_group()
