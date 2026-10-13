#!/bin/bash
# Container, CVD=0,1,2,3, quiet whole node: TP4 nccl-tests + counters, TP2 sweep + counters, link facts -> $RUNS/link/win_<host>.
. "$(dirname "$0")/common.sh"; S=$(cd "$(dirname "$0")" && pwd); O=$RUNS/link/win_$(hostname -s)
mkdir -p $O; T0=$(date +%s); el() { echo "# T+$(( $(date +%s) - T0 ))s $*"; }
el start; $S/facts2.sh > $O/facts2.log 2>&1
el tp4cells; $S/nc.sh $S/cells_tp4.txt $O/nc_tp4 > $O/nc_tp4.log 2>&1
el tp4cnt; $S/ncnt.sh 1G 1000 > $O/ncnt_tp4.log 2>&1
el tp2cnt; CUDA_VISIBLE_DEVICES=0,1 $S/ncnt.sh 1G 2000 > $O/ncnt_tp2.log 2>&1
el tp2cells; CUDA_VISIBLE_DEVICES=0,1 $S/nc.sh $S/cells_tp2.txt $O/nc_tp2 > $O/nc_tp2.log 2>&1
el tp2cellsb; CUDA_VISIBLE_DEVICES=0,1 $S/nc.sh $S/cells_tp2b.txt $O/nc_tp2 > $O/nc_tp2b.log 2>&1
el end
