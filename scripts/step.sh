#!/bin/bash
# Login node: one container step through gpurun.sh. Usage: step.sh <node> <gpus|-> <timeout_s> <script> [args]
# A bare script name resolves under $SRC/scripts, then $SRC/tests; LANE=<name> labels the lock.
. "$(dirname "$0")/env.sh"
N=$1 G=$2 T=$3; shift 3
S=$1; shift
if [ "${S:0:1}" != / ]; then for d in scripts tests; do [ -e "$(dirname "$0")/../$d/$S" ] && { S=$SRC/$d/$S; break; }; done; fi
exec bash "$(dirname "$0")/gpurun.sh" ${LANE:-harness} $N $G $T $S "$@"
