#!/bin/bash
# Container wrapper so build_all.sh runs as one step with its log kept. Usage: build_step.sh [tag]
. "$(dirname "$0")/env.sh"; mkdir -p $LOGS
bash "$(dirname "$0")/build_all.sh" "$@" 2>&1 | tee -a $LOGS/build.log
exit ${PIPESTATUS[0]}
