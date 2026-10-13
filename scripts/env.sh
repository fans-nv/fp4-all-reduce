# Sourced by every script: loads the cluster profile and derives the work layout. See cluster.env.example.
_R=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
_CE=${FP4AR_CLUSTER_ENV:-$_R/scripts/cluster.env}
[ -f "$_CE" ] || { echo "missing $_CE (copy scripts/cluster.env.example)" >&2; return 2 2>/dev/null || exit 2; }
. "$_CE"
for _v in JOB IMG WORK ARCH NVCC NCCL_HOME; do [ -n "${!_v}" ] || { echo "cluster.env: $_v unset for CLUSTER=$CLUSTER" >&2; return 2 2>/dev/null || exit 2; }; done
export CLUSTER
SRC=${SRC:-$WORK/src}                       # sync.sh target (repo copy on the cluster)
NCCL_LIBDIR=${NCCL_LIBDIR:-$NCCL_HOME/lib}
MOUNTS=${MOUNTS:?set MOUNTS in cluster.env}; NGPU=${NGPU:-4}
BUILD=$WORK/build RUNS=$WORK/runs LOGS=$WORK/logs
