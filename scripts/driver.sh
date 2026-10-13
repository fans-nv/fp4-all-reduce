#!/bin/bash
# Login node: runs a cell file sequentially, one step per cell. Usage: driver.sh <cellfile> [deadline HH:MM local]
# Cell line: <cell> <node> <gpus> <timeout_s> [VAR=val ...] -- <fp4ar_bench args>   ('#' and blank lines ignored)
# NODE0/NODE1 in the node column are replaced by the env vars of the same name.
. "$(dirname "$0")/env.sh"
F=$1; DL=${2:-}; L=${DRIVER_LOG:-$LOGS}; mkdir -p $L
CUT=0; [ -n "$DL" ] && CUT=$(date -d "today $DL" +%s)
while read -r CELL N G T REST; do
  [ -z "$CELL" ] || [ "${CELL:0:1}" = "#" ] && continue
  N=${N/NODE0/$NODE0}; N=${N/NODE1/$NODE1}
  if [ $CUT -gt 0 ] && [ $(( $(date +%s) + T )) -gt $CUT ]; then echo "SKIP $CELL (deadline)" >> $L/driver.log; continue; fi
  echo "START $CELL $N $G $(date +%T)" >> $L/driver.log
  bash "$(dirname "$0")/step.sh" $N $G $T run_cell.sh $CELL $REST > $L/$CELL.step.log 2>&1; rc=$?
  echo "END $CELL rc=$rc $(date +%T)" >> $L/driver.log
  grep -q CLOCK_ABORT $L/$CELL.step.log && { echo "ABORT clocks" >> $L/driver.log; exit 3; }
done < "$F"
echo "DRIVER DONE $F $(date +%T)" >> $L/driver.log
