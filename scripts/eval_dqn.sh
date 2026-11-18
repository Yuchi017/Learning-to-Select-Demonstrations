#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Path to a checkpoint produced by scripts/train_dqn.sh
CHECKPOINT="train_res/age_prediction/Qwen2.5-VL-7B-Instruct/<run_id>/checkpoints/step_5000.pth"

python eval_dqn.py \
    --task_type "age_prediction" \
    --model_checkpoint "$CHECKPOINT" \
    --inf_model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --eval_num_samples 1000 \
    --eval_k_shots 16 \
    --eval_random \
    --eval_knn \
    --eval_dqn
