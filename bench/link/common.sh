# Sourced by the link scripts: NVLink counter snapshots around a workload. NT = nccl-tests bin dir.
. "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"
NT=${NT:-/usr/local/bin}; CE=${CE:-$BUILD/link/ce}; BW=${BW:-$BUILD/micro/bw}
G=$(echo $CUDA_VISIBLE_DEVICES | tr , ' '); NG=$(echo $G | wc -w)
snap() { echo "# SNAP $1 $(date +%T) $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,clocks.mem,clocks.sm --format=csv,noheader | tr '\n' ';')"; }
cnt() { for g in $G; do echo "## CNT $1 gpu$g $(date +%s.%N)"; nvidia-smi nvlink -gt d -i $g; nvidia-smi nvlink -gt r -i $g; nvidia-smi nvlink -e -i $g | grep -E 'Tx packets|Tx bytes|Rx packets|Rx bytes'; done; }
