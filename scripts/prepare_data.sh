#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
set -euo pipefail

CONFIG=datasets.yaml

tasks=$(python - <<EOF
import yaml
cfg = yaml.safe_load(open("$CONFIG"))
print(" ".join(cfg["datasets"].keys()))
EOF
)

for task in $tasks; do
  echo "Processing task: $task"

  data_dir=$(python - <<EOF
import yaml
cfg = yaml.safe_load(open("$CONFIG"))
print(cfg["datasets"]["$task"]["root"])
EOF
)

  python prepare_data.py \
    --task_type "$task" \
    --data_dir "$data_dir" \
    --output_dir "$data_dir" \
    --vision_model_name google/siglip-base-patch16-224 \
    --test_split 0.2

done