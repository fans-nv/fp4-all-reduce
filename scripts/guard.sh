#!/bin/bash
# Login node backstop: at epoch $1 reset clocks on every listed node not RESET_VERIFIED. Usage: nohup guard.sh EPOCH NODE... &
. "$(dirname "$0")/env.sh"; AT=$1; shift; mkdir -p $WORK/clock
while [ "$(date +%s)" -lt "$AT" ]; do sleep 10; done
echo "guard fired $(date +%T)" >> $WORK/clock/CLOCK.log
for N in "$@"; do bash "$(dirname "$0")/clk.sh" reset $N guard & done; wait
