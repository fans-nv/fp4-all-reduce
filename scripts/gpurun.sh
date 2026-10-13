#!/bin/bash
# Login node: per-GPU mkdir locks (atomic on Lustre), then ONE `srun --overlap` step in the existing job $JOB.
# Usage: gpurun.sh <lane> <node> <gpus e.g. 0,1 | -> <timeout_s> <script.sh> [args...]   (gpus "-" = no GPU lock, no CVD)
set -u
. "$(dirname "$0")/env.sh"
lane=$1 node=$2 gpus=$3 to=$4; shift 4
LK=$WORK/locks; mkdir -p "$LK"
STALE=${LOCK_STALE_S:-$(( to + 600 ))}  # a holder older than its own timeout + margin is dead
locks=(); [ "$gpus" != - ] && for g in $(echo $gpus | tr , '\n' | sort -n); do locks+=("$LK/$node.gpu$g"); done
got=()
release() { for d in "${got[@]}"; do rm -rf "$d"; done; }
trap release EXIT
trap 'release; exit 143' INT TERM  # a signalled waiter must not keep polling and launch later
for d in "${locks[@]}"; do
  n=0
  until mkdir "$d" 2>/dev/null; do
    if [ -f "$d/owner" ] && [ $(( $(date +%s) - $(stat -c %Y "$d/owner") )) -gt $(( $(cat "$d/stale" 2>/dev/null || echo 2400) )) ]; then rm -rf "$d"; continue; fi
    n=$((n+1)); [ $((n % 30)) -eq 0 ] && echo "# waiting for $d held by $(cat $d/owner 2>/dev/null)" >&2
    sleep 2
  done
  echo "$lane $(date +%T) $$" > "$d/owner"; echo $STALE > "$d/stale"; got+=("$d")
done
echo "# $lane got $node gpus $gpus at $(date +%T)" >&2
CVD=$gpus; [ "$gpus" = - ] && CVD=
timeout $to srun --jobid=$JOB --overlap -N1 -n1 -w $node --container-image=$IMG --container-mounts=$MOUNTS \
  bash -c 'export CLUSTER='"$CLUSTER"' HOME='"$WORK"'/home; [ -n "'"$CVD"'" ] && export CUDA_VISIBLE_DEVICES='"$CVD"'; exec bash "$@"' _ "$@" 2>&1 | grep -v '^pyxis'
rc=${PIPESTATUS[0]}
echo "# $lane released $node gpus $gpus rc=$rc at $(date +%T)" >&2
exit $rc
