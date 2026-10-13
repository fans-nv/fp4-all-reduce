#!/usr/bin/env python3
"""Summarize micro JSONL runs into markdown tables. Usage: summarize.py <runs_dir> [platform_label]"""
import json, os, sys, glob

D = sys.argv[1]
SMS_ALL = None


def load(name):
    p = os.path.join(D, name + ".jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.startswith("{")]


def bw_table(rows, pattern, B, methods, chunks, nsms):
    rows = [r for r in rows if "med_us" in r and r["pattern"] == pattern and r["bytes_per_peer"] == B]
    if not rows:
        return
    allsm = max(r["nsm"] for r in rows)
    nsms = [allsm if n == 0 else n for n in nsms]
    print(f"\n**{pattern}, {B >> 20 if B >= 1 << 20 else B >> 10}{'M' if B >= 1 << 20 else 'K'} per peer, inner={rows[0]['inner']}** — egress GB/s (best of CTAs/SM)\n")
    print("| method | chunk | " + " | ".join(f"{n} SMs" for n in nsms) + " |")
    print("|---|---|" + "---|" * len(nsms))
    for m in methods:
        for C in chunks:
            v = []
            for n in nsms:
                x = [r["egress_GBps"] for r in rows if r["method"] == m and r["chunk"] == C and r["nsm"] == n]
                v.append(f"{max(x):.0f}" if x else "-")
            if any(s != "-" for s in v):
                print(f"| {m} | {C >> 10}K | " + " | ".join(v) + " |")


def bw_rows(f):
    # a <f>_pull rerun (sys-scope loads, tag *rlx*) supersedes pull_ld rows in <f>
    rows, over = load(f), load(f + "_pull")
    if over:
        rows = [r for r in rows if r.get("method") != "pull_ld"] + over
    return rows


def main():
    nsms = [16, 32, 64, 0]
    for f, pats in [("bw_tp2", ["uni", "a2a", "self"]), ("bw_tp4", ["uni", "fanout", "a2a"])]:
        rows = bw_rows(f)
        if rows:
            print(f"\n### {f}")
            for p in pats:
                bw_table(rows, p, 64 << 20, ["push_st", "tma_st", "tma_ld", "tma_ld_cluster", "pull_ld", "mc_st"], [1024, 4096, 16384, 32768], nsms)
    for f in ["bw_tp2_inner", "bw_tp4_inner"]:
        rows = bw_rows(f)
        if rows:
            print(f"\n### {f}")
            for B in [512 << 10, 1 << 20, 4 << 20]:
                bw_table(rows, "a2a", B, ["push_st", "tma_st", "tma_ld", "pull_ld", "mc_st"], [4096, 16384], nsms)
    for f in ["lat_tp2", "lat_tp4"]:
        rows = load(f)
        if rows:
            print(f"\n### {f} — one-way latency = RTT/2, ns (median)\n")
            sizes = sorted({r["size"] for r in rows})
            print("| method | " + " | ".join(f"{s} B" for s in sizes) + " |")
            print("|---|" + "---|" * len(sizes))
            for m in dict.fromkeys(r["method"] for r in rows):
                v = []
                for s in sizes:
                    x = [r["oneway_med_ns"] for r in rows if r["method"] == m and r["size"] == s]
                    v.append(f"{min(x):.0f}" if x else "")
                print(f"| {m} | " + " | ".join(v) + " |")
            bad = [r for r in rows if r["timeouts"] or r["stale"]]
            print(f"\ntimeouts/stale rows: {len(bad)}")
    for f in ["fence_tp2", "fence_tp4"]:
        rows = load(f)
        lat = [r for r in rows if r["part"] == "lat"]
        if lat:
            print(f"\n### {f} — publication op cost, cycles (rank 0 median), 8 warps, data and flag on peer unless noted\n")
            nst = sorted({r["nst_per_thread"] for r in lat})
            print("| op | bg | dst/flag | " + " | ".join(f"N={n}" for n in nst) + " |")
            print("|---|---|---|" + "---|" * len(nst))
            for bg in [0, 1]:
                for dst, fd in [("peer", "peer"), ("local", "local")]:
                    for op in ["st_release_sys", "fence_acq_rel_sys", "red_release_sys_add", "fence_acq_rel_gpu", "st_release_gpu", "st_relaxed_sys"]:
                        v = []
                        for n in nst:
                            x = [r["rank0_med_cyc"] for r in lat if r["op"] == op and r["bg"] == bg and r["dst"] == dst and r["flagdst"] == fd and r["warps"] == 8 and r["nst_per_thread"] == n]
                            v.append(f"{x[0]:.0f}" if x else "")
                        print(f"| {op} | {bg} | {dst}/{fd} | " + " | ".join(v) + " |")
            print(f"\neffective SM clock {lat[0]['eff_mhz']:.0f} MHz")
        tp = [r for r in rows if r["part"] == "tp"]
        if tp:
            print(f"\n### {f} — persistent push loop, {tp[0]['bytes_per_peer'] >> 20} MB per peer, {tp[0]['pattern']}\n")
            print("| variant | tile | CTAs/SM | GB/s | overhead ns/tile |")
            print("|---|---|---|---|---|")
            for r in tp:
                print(f"| {r['var']} | {r['tile'] >> 10}K | {r['cpsm']} | {r['egress_GBps']:.0f} | {r['overhead_ns_per_tile']:.0f} |")
    for f in ["tear_tp2", "tear_tp4", "tear_tp2_tmald_fix"]:
        rows = load(f)
        if rows:
            print(f"\n### {f}\n")
            print("| writer | reader | bytes | reads | torn | transitions | regress |")
            print("|---|---|---|---|---|---|---|")
            for r in rows:
                print(f"| {r['writer']} | {r['reader']} | {r['bytes'] >> 10}K | {r['reads']:.2e} | {r['torn']} | {r['transitions']:.2e} | {r['regress']} |")
    for f in glob.glob(os.path.join(D, "validity*.jsonl")):
        rows = [json.loads(l) for l in open(f) if l.startswith("{")]
        sem = [r for r in rows if r.get("part") == "sem"]
        if sem:
            print(f"\n### {os.path.basename(f)} (sem)\n")
            print("| variant | case | src | cluster | done | report | cond bits | ns | reissue report | reissue cond bits | reissue ns |")
            print("|---|---|---|---|---|---|---|---|---|---|---|")
            for r in sem:
                print(f"| {r['variant']} | {r['case']} | {r['src']} | {r['cluster_form']} | {r['load1_done']} | {r['load1_report']} | {r['load1_cond_phase']} | {r['load1_ns']} | {r['reissue_valid_report']} | {r['reissue_valid_cond_phase']} | {r['reissue_ns']} |")
        xg = [r for r in rows if r.get("part") == "xgpu"]
        if xg:
            print(f"\n### {os.path.basename(f)} (xgpu)\n")
            for r in xg:
                print(f"- {r['variant']} push={r['push']}: valid chunk loads {r['chunk_loads_valid']}, TMA loads {r['tma_loads']}, retries {r['retries']}, false_valid {r['false_valid']}, timeouts {r['timeouts_consumer']}/{r['timeouts_producer']}, {r['us_per_iter']:.2f} us/iter")


main()
