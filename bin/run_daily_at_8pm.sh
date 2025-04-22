#!/bin/bash

SYNC_SCRIPT="/c/Users/gurov/Python_Projects/BA/sync_from_pi.sh"
LAST_RUN_DATE=""

while true; do
  CURRENT_TIME=$(date +%H:%M)
  TODAY=$(date +%F)

  if [[ "$CURRENT_TIME" == "20:00" && "$TODAY" != "$LAST_RUN_DATE" ]]; then
    echo "Running sync at $(date)"
    bash "$SYNC_SCRIPT"
    LAST_RUN_DATE="$TODAY"
  fi

  sleep 900  # Check every 15 minute
done


