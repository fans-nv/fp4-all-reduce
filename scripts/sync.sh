#!/bin/bash
# Workstation: copies the repo sources (incl. scripts/cluster.env) to $LOGIN:$SRC. Usage: CLUSTER=<name> sync.sh
. "$(dirname "$0")/env.sh"
R=$(cd "$(dirname "$0")/.." && pwd)
ssh -o BatchMode=yes $LOGIN "mkdir -p $SRC $BUILD $RUNS $LOGS"
tar -C $R --exclude='*.o' --exclude=build -czf - build.sh Makefile include harness kernels scripts tests bench | ssh -o BatchMode=yes $LOGIN "tar -C $SRC -xzf -"
echo "synced $R -> $LOGIN:$SRC"
