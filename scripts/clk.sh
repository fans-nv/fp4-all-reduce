#!/bin/bash
# Login node: HBM clock lock/reset with readback verification. Usage: clk.sh lock|reset|read NODE [why]
# lock: -lmc L,L then every GPU must read L; reset: -rmc then every GPU must read MEMCLK_DEFAULT.
# State per node in $WORK/clock/NODE.state, every action appended to $WORK/clock/CLOCK.log.
. "$(dirname "$0")/env.sh"
OP=$1 N=$2 WHY=${3:-planned}; C=$WORK/clock; mkdir -p $C
[ -z "$MEMCLK_LOCK" ] && [ "$OP" != read ] && { echo "MEMCLK_LOCK unset for CLUSTER=$CLUSTER: no clock control"; exit 0; }
S="srun --jobid=$JOB --overlap -N1 -n1 --nodelist=$N --cpus-per-task=1 --mem=1G"
rb() { $S nvidia-smi --query-gpu=clocks.mem --format=csv,noheader,nounits | tr -d " " | paste -sd,; }
rep() { local v=$1 s=$1; for ((i = 1; i < NGPU; i++)); do s="$s,$v"; done; echo $s; }
case $OP in
 read) echo "$N $(rb)" ;;
 lock)
  { echo "=== $N LOCK $(date '+%F %T %Z'); before: $(rb)"; $S bash -c "sudo -n nvidia-smi -lmc $MEMCLK_LOCK,$MEMCLK_LOCK; echo rc=\$?" 2>&1; } >> $C/CLOCK.log
  sleep 2; R=$(rb); echo "$N readback $(date +%T): $R" >> $C/CLOCK.log
  if [ "$R" = "$(rep $MEMCLK_LOCK)" ]; then echo LOCKED_$MEMCLK_LOCK > $C/$N.state; else echo LOCK_FAILED > $C/$N.state; fi
  echo "STATE[$N]=$(cat $C/$N.state) $R $(date '+%F %T %Z')" | tee -a $C/CLOCK.log
  [ "$(cat $C/$N.state)" = LOCK_FAILED ] && exit 1 ;;
 reset)
  [ "$(cat $C/$N.state 2>/dev/null)" = RESET_VERIFIED ] && exit 0
  { echo "=== $N RESET ($WHY) $(date '+%F %T %Z')"; $S bash -c 'sudo -n nvidia-smi -rmc; echo rc=$?' 2>&1; } >> $C/CLOCK.log
  sleep 3; R=$(rb)
  if [ "$R" = "$(rep $MEMCLK_DEFAULT)" ]; then echo RESET_VERIFIED > $C/$N.state; else echo RESET_UNVERIFIED > $C/$N.state; fi
  echo "STATE[$N]=$(cat $C/$N.state) $R [$WHY] $(date '+%F %T %Z')" | tee -a $C/CLOCK.log
  [ "$(cat $C/$N.state)" = RESET_UNVERIFIED ] && exit 1 ;;
 *) echo "usage: clk.sh lock|reset|read NODE [why]"; exit 2 ;;
esac
