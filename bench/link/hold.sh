#!/bin/bash
# Container, CVD=<2 gpus>: NVLink counter brackets around saturating CE / SM / TMA transfers (needs link + micro builds).
. "$(dirname "$0")/common.sh"
h() { local n=$1; shift; cnt "pre $n"; "$@"; cnt "post $n"; }
snap start
h sm_pull_uni_424 $CE --mode sm_pull --dir uni --side dst --ctas 424 --sizes 1G --hold_s 3 --tag hold_sm_pull_uni_424
h sm_pull_bidir_424 $CE --mode sm_pull --dir bidir --side dst --ctas 424 --sizes 1G --hold_s 3 --tag hold_sm_pull_bidir_424
h ce_dst_uni $CE --mode ce --dir uni --side dst --streams 1 --sizes 1G --hold_s 3 --tag hold_ce_dst_uni
h ce_dst_bidir $CE --mode ce --dir bidir --side dst --streams 4 --sizes 1G --hold_s 3 --tag hold_ce_dst_bidir
h mix_uni_106 $CE --mode ce+sm --dir uni --streams 1 --ctas 106 --smpct 50 --sizes 1G --hold_s 3 --tag hold_mix_uni_106
h bw_tma_st_uni $BW --n 2 --method tma_st --pattern uni --chunk 4K --bytes 64M --cpsm 1 --nsm 32 --iters 3000 --tag hold
h bw_tma_st_a2a $BW --n 2 --method tma_st --pattern a2a --chunk 2K --bytes 64M --cpsm 1 --nsm 32 --iters 3000 --tag hold
h bw_tma_ld_uni $BW --n 2 --method tma_ld --pattern uni --chunk 16K --bytes 64M --cpsm 2 --nsm 106 --iters 3000 --tag hold
h bw_push_st_uni $BW --n 2 --method push_st --pattern uni --chunk 8K --bytes 64M --cpsm 1 --nsm 32 --iters 3000 --tag hold
snap end
