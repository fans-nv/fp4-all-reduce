#!/bin/bash
# Container: AG + A2A sweeps for one TP. Args: <n> <rep>
D=$(cd "$(dirname "$0")" && pwd)
$D/nt_sweep.sh $1 ag $2 "Ring NVLS PAT"
$D/nt_sweep.sh $1 a2a $2 "Ring" "LL LL128 Simple" "16 32 64"
$D/nt_extra.sh $1 ag $2
$D/nt_extra.sh $1 a2a $2
