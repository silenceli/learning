#!/bin/bash

PWD=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

# For container run
LEARNING_RATE=${LEARNING_RATE:-0.002}
EPOCHS=${EPOCHS:-5}
MODEL_DIR=${MODEL_DIR:-"/tmp/model/"}

echo "Current DIR: $PWD"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "MODEL_DIR: $MODEL_DIR"

mkdir -p $MODEL_DIR

# Install requirements.txt
pip install -r requirements.txt

# RUN!
echo python main.py --epochs ${EPOCHS} --model_dir ${MODEL_DIR} --learning_rate ${LEARNING_RATE}
python main.py --epochs ${EPOCHS} --model_dir ${MODEL_DIR} --learning_rate ${LEARNING_RATE}