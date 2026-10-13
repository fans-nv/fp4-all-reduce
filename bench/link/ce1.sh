#!/bin/bash
# Container, CVD=<2 gpus>: CE vs SM peer bandwidth + NVLink counter holds. Usage: ce1.sh [parts=ce,sm,mix,hold]  (build.sh first)
. "$(dirname "$0")/common.sh"; P=${1:-ce,sm,mix,hold}
has() { [[ ",$P," == *",$1,"* ]]; }
snap start
if has ce; then
  $CE --mode ce --dir uni --side src --streams 1,2,4,8 --sizes 1M,4M,16M,64M,256M,1G,4G --tag ce_uni_src
  $CE --mode ce --dir uni --side dst --streams 1,4 --sizes 1M,16M,256M,1G,4G --tag ce_uni_dst
  $CE --mode ce --dir bidir --side src --streams 1,2,4,8 --sizes 1M,4M,16M,64M,256M,1G,4G --tag ce_bidir_src
fi
if has sm; then
  $CE --mode sm_push --dir uni --ctas 8,16,24,32,64,106,212,424 --sizes 16M,256M,1G --tag sm_push_uni
  $CE --mode sm_push --dir bidir --ctas 16,32,64,212,424 --sizes 16M,256M,1G --tag sm_push_bidir
  $CE --mode sm_pull --dir uni --side dst --ctas 16,32,64,212,424 --sizes 16M,256M,1G --tag sm_pull_uni
  $CE --mode sm_pull --dir bidir --side dst --ctas 32,64,212,424 --sizes 256M,1G --tag sm_pull_bidir
fi
if has mix; then
  for pct in 25 50 75; do $CE --mode ce+sm --dir uni --streams 1,4 --ctas 32,106 --smpct $pct --sizes 256M,1G --tag mix_uni_$pct; done
  $CE --mode ce+sm --dir bidir --streams 4 --ctas 32,106 --smpct 50 --sizes 1G --tag mix_bidir_50
fi
if has hold; then
  for c in "ce uni 4 212" "ce bidir 4 212" "sm_push uni 1 212" "sm_push bidir 1 212" "sm_push uni 1 16"; do set -- $c
    cnt "pre $1_$2_$4"; $CE --mode $1 --dir $2 --streams $3 --ctas $4 --sizes 1G --hold_s 3 --tag hold_$1_$2_$4; cnt "post $1_$2_$4"; done
  cnt "pre idle"; sleep 3; cnt "post idle"
fi
snap end
