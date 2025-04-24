#!/bin/bash

PI_USER="gurov"
PI_HOST="160.85.154.227"
REMOTE_DIR="/home/gurov/Raspi2_Kombucha/data/Experiment_4"
LOCAL_DIR="/c/Users/gurov/Python_Projects/BA/data/Experiment_4/"

# Get list of remote files
REMOTE_FILES=$(ssh ${PI_USER}@${PI_HOST} "ls ${REMOTE_DIR}")

for filename in $REMOTE_FILES; do
  if [ ! -f "${LOCAL_DIR}${filename}" ]; then
    echo "Copying new file: $filename"
    scp ${PI_USER}@${PI_HOST}:${REMOTE_DIR}/${filename} ${LOCAL_DIR}
  else
    echo "Skipping existing file: $filename"
  fi
done

