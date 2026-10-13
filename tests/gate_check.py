#!/usr/bin/env python3
"""Verdict for gate.sh / negative_control.sh manifests.

gate:   every (variant, tp, input) run must exit 0 with >=1 check line, all pass; full gate counts (no lowered reps).
        Cells the harness skips (window_bytes=0, no two-shot, no input_slot) are listed as SKIP, never counted as PASS.
negctl: per variant, clean must pass, stale and flip must each produce >=1 check line and all of them fail (rc 3).
"""
import json
import sys

FULL = {"b2b_calls": 1000, "percall_calls": 1000, "graph_calls": 10000, "skew_calls": 200}


def checks(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("{"):
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if d.get("kind") == "check":
                        out.append(d)
    except FileNotFoundError:
        pass
    return out


def gate(rows):
    bad, skipped, cells, by_var = [], [], 0, {}
    for v, tp, inp, rc, path in rows:
        cs = checks(path)
        by_var.setdefault(v, 0)
        if not cs:
            (skipped if rc == "0" else bad).append(f"{v} tp{tp} {inp}: rc={rc}, no check lines")
            continue
        for c in cs:
            cells += 1
            key = f"{v} tp{tp} {inp} algo{c['algo']} rows{c['rows']}"
            low = [k for k, n in FULL.items() if c.get(k, 0) < n]
            if not c["pass"]:
                bad.append(f"{key}: FAIL mism_first={c['mism_first']} b2b={c['mism_b2b_final']} percall={c['mism_percall']} "
                           f"graph={c['mism_graph']} skew={c['mism_skew']} ranks_identical={c['ranks_identical']}")
            elif low:
                bad.append(f"{key}: passed with lowered counts {low}")
            else:
                by_var[v] += 1
        if rc != "0" and all(c["pass"] for c in cs):
            bad.append(f"{v} tp{tp} {inp}: rc={rc} despite passing check lines (crash after the gate?)")
    for v, n in by_var.items():
        if n == 0:
            bad.append(f"{v}: zero passing cells at any tier (vacuous)")
    for s in skipped:
        print("SKIP", s)
    for b in bad:
        print("RED ", b)
    print(f"gate: {cells} cells, {sum(by_var.values())} pass, {len(skipped)} skipped runs, {len(bad)} problems -> "
          f"{'GREEN' if not bad else 'RED'}")
    return 0 if not bad else 1


def negctl(rows):
    bad = []
    for v, nc, rc, path in rows:
        cs = checks(path)
        if not cs:
            bad.append(f"{v} {nc}: no check lines (rc={rc}); the control did not exercise the gate")
        elif nc == "clean" and not (rc == "0" and all(c["pass"] for c in cs)):
            bad.append(f"{v} clean: did not pass (rc={rc}); the controls below prove nothing")
        elif nc != "clean" and (rc != "3" or any(c["pass"] for c in cs)):
            bad.append(f"{v} {nc}: gate stayed green (rc={rc})")
        else:
            print(f"ok   {v} {nc}: rc={rc} {'pass' if nc == 'clean' else 'red as required'}")
    for b in bad:
        print("BAD ", b)
    print(f"negative_control: {len(rows)} runs, {len(bad)} problems -> {'GATE CAN GO RED' if not bad else 'FAILED'}")
    return 0 if not bad else 1


def main():
    mode, manifest = sys.argv[1], sys.argv[2]
    rows = [l.rstrip("\n").split("\t") for l in open(manifest) if l.strip()]
    sys.exit(gate(rows) if mode == "gate" else negctl(rows))


if __name__ == "__main__":
    main()
