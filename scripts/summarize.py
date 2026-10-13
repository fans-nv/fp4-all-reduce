#!/usr/bin/env python3
"""Summarize fp4ar_bench JSON lines into Markdown tables.

Usage: summarize.py runs/*.jsonl [--graph 0|10] [--variants r1,L,...]
Tuned BF16 = fastest measured NCCL config for the same (tp, rows, graphK) (protocol and channel envs recorded);
the brief's fixed rule (LL128 TP2<=2K / TP4<=4K else Simple) is shown alongside when measured.
Overlap efficiency = (X + C - F) / min(X, C) with X = xport, C = compute, F = fused (1 = fully hidden, 0 = serial).
Input modes (bf16 / nvfp4 / nvfp4-window) are separate tables; in nvfp4-window graph mode the GEMM-emulation copy is
already subtracted by the harness (copy_us field). Corrected subtracts the fixed cost E (empty kernel + rank align, mode "empty"): (X + C - F - E) / min(X - E, C - E).
"""
import json
import sys
from collections import defaultdict


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line.startswith("{"):
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return rows


def fmt(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"


def main():
    args = [a for a in sys.argv[1:]]
    graph = None
    only = None
    if "--graph" in args:
        i = args.index("--graph"); graph = int(args[i + 1]); del args[i:i + 2]
    if "--variants" in args:
        i = args.index("--variants"); only = set(args[i + 1].split(",")); del args[i:i + 2]
    recs = load(args)
    timing = [r for r in recs if r.get("kind") == "timing"]
    checks = [r for r in recs if r.get("kind") == "check"]

    # latest record wins per key
    fp4 = {}
    bf16 = defaultdict(list)
    for r in timing:
        if r["variant"] == "nccl_bf16":
            bf16[(r["tp"], r["rows"], r["graphK"])].append(r)
        else:
            fp4[(r["variant"], r["algo"], r["tp"], r["rows"], r["graphK"], tuple(r.get("tune", [])), r["mode"], r.get("input", "bf16"))] = r

    def best_bf16(tp, rows, g):
        c = bf16.get((tp, rows, g), [])
        if not c:
            return None, None, None
        b = min(c, key=lambda r: r["med_us"])
        rule = [r for r in c if r.get("proto") == r.get("rule_proto") and r.get("min_nch") == "64"]
        cfg = f"{b.get('proto')}/{b.get('nccl_algo') or 'def'}/{b.get('min_nch') or 'def'}ch"
        return b["med_us"], cfg, (min(rule, key=lambda r: r["med_us"])["med_us"] if rule else None)

    groups = sorted({(k[7], k[0], k[1], k[4], k[5]) for k in fp4})
    for (inp, v, algo, g, tune) in groups:
        if only and v not in only:
            continue
        if graph is not None and g != graph:
            continue
        print(f"\n**{v} input={inp} algo={'one-shot' if algo == 1 else 'two-shot'} {'graph x' + str(g) if g else 'eager'} tune={list(tune)}** (µs, median of max-over-ranks)\n")
        print("| TP | Rows | Fused (p10/p90) | XPORT | COMPUTE | Empty | Tuned BF16 (cfg) | Rule BF16 | Speedup | Overlap raw | Overlap corr. |")
        print("| --: | --: | --- | --: | --: | --: | --- | --: | --: | --: | --: |")
        keys = sorted({(k[2], k[3]) for k in fp4 if (k[7], k[0], k[1], k[4], k[5]) == (inp, v, algo, g, tune)})
        for tp, rows in keys:
            get = lambda m: fp4.get((v, algo, tp, rows, g, tune, m, inp))
            F, X, C, E = (get(m) for m in ("fused", "xport", "compute", "empty"))
            f = F["med_us"] if F else None
            x = X["med_us"] if X else None
            c = C["med_us"] if C else None
            e = E["med_us"] if E else None
            bb, cfg, rule = best_bf16(tp, rows, g)
            sp = f"{bb / f:.2f}x" if (bb and f) else ""
            ov = ovc = None
            if f and x and c:
                ov = (x + c - f) / min(x, c)
                if e is not None and min(x, c) - e > 0:
                    ovc = (x + c - f - e) / (min(x, c) - e)
            fcol = f"{fmt(f)} ({fmt(F['p10_us'])}/{fmt(F['p90_us'])})" if F else ""
            print(f"| {tp} | {rows // 1024}K | {fcol} | {fmt(x)} | {fmt(c)} | {fmt(e)} | {fmt(bb)} ({cfg or ''}) | {fmt(rule)} | {sp} | {fmt(ov)} | {fmt(ovc)} |")

    if checks:
        print("\n**Correctness gate** (bitwise vs same-codec v2 reference; calls = b2b / per-call / graph / skew)\n")
        print("| Variant | Input | Algo | TP | Rows | Pass | Mismatch (first/b2b/per-call/graph/skew) | Ranks identical | rel-L2 vs FP32 | Codec-ref rel-L2 |")
        print("| --- | --- | --: | --: | --: | --- | --- | --- | --: | --: |")
        last = {}
        for r in checks:
            last[(r.get("input", "bf16"), r["variant"], r["algo"], r["tp"], r["rows"], tuple(r.get("tune", [])))] = r
        for k in sorted(last):
            r = last[k]
            if only and r["variant"] not in only:
                continue
            mm = "/".join(str(int(r[m])) for m in ("mism_first", "mism_b2b_final", "mism_percall", "mism_graph", "mism_skew"))
            print(f"| {r['variant']} | {r.get('input', 'bf16')} | {r['algo']} | {r['tp']} | {r['rows'] // 1024}K | {'PASS' if r['pass'] else 'FAIL'} | {mm} "
                  f"(calls {r['b2b_calls']}/{r['percall_calls']}/{r['graph_calls']}/{r['skew_calls']}) | {r['ranks_identical']} | "
                  f"{r['rel_l2_vs_fp32']:.4f} | {r['codecref_rel_l2_vs_fp32']:.4f} |")


if __name__ == "__main__":
    main()
