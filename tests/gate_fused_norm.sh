#!/bin/bash
# Container: fused-norm driver gate (bench/fused_norm) + its negative control. Exit 0 only if every clean cell PASSES
# and every NEG cell (reference weight bias + 2^-7) FAILS. Env: FN_TPS (default 2), FN_EPIS, FN_QS, FN_ROWS.
. "$(dirname "$0")/../scripts/env.sh"
R=$(cd "$(dirname "$0")/.." && pwd); RUN=$R/bench/fused_norm/run.sh; T=gate-$(date +%m%d-%H%M%S)
bad=0; cells=0
for n in ${FN_TPS:-2}; do for q in ${FN_QS:-0 1}; do for epi in ${FN_EPIS:-0 1 2}; do
  for neg in "" NEG; do
    a="--n $n --q $q --epi $epi --rows ${FN_ROWS:-1024,16384} --mode check --tag ${T}_n${n}q${q}e${epi}$neg"
    out=$(bash $RUN $T - "$neg$a" 2>&1); np=$(grep -c '^CHECK.* PASS$' <<< "$out"); nf=$(grep -c '^CHECK.* FAIL$' <<< "$out")
    cells=$((cells + 1))
    if [ -z "$neg" ]; then ok=$([ $np -gt 0 ] && [ $nf = 0 ] && grep -q '^rc=0' <<< "$out" && echo 1)
    else ok=$([ $nf -gt 0 ] && ! grep -q '^rc=0' <<< "$out" && echo 1); fi
    echo "${ok:+ok  }${ok:-RED } n=$n q=$q epi=$epi ${neg:-clean}: $np pass $nf fail"
    [ -z "$ok" ] && { bad=$((bad + 1)); grep -E 'build failed|error|^rc=' <<< "$out" | head -5; }
  done
done; done; done
echo "gate_fused_norm: $cells runs, $bad problems -> $([ $bad = 0 ] && echo GREEN || echo RED)"; [ $bad = 0 ]
