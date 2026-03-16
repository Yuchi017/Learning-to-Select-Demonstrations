#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python train_dqn.py \
    --task_type "age_prediction" \
    --model_name "google/gemma-3-4b-it" \
    --training_steps 20000

# CUDA_VISIBLE_DEVICES=3 python train_dqn.py \
#     --task_type "age_prediction" \
#     --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
#     --training_steps 20000

# CUDA_VISIBLE_DEVICES=3 python train_dqn.py \
#     --task_type "age_prediction" \
#     --model_name "OpenGVLab/InternVL2-8B" \
#     --training_steps 20000

# CUDA_VISIBLE_DEVICES=3 python train_dqn.py \
#     --task_type "age_prediction" \
#     --model_name "microsoft/Phi-3.5-vision-instruct" \
#     --training_steps 20000

# CUDA_VISIBLE_DEVICES=3 python train_dqn.py \
#     --task_type "aesthetic_score" \
#     --model_name "google/gemma-3-4b-it" \
#     --training_steps 20000

# CUDA_VISIBLE_DEVICES=3 python train_dqn.py \
#     --task_type "facial_beauty" \
#     --model_name "google/gemma-3-4b-it" \
#     --training_steps 20000

# CUDA_VISIBLE_DEVICES=3 python train_dqn.py \
#     --task_type "wild_image_quality" \
#     --model_name "google/gemma-3-4b-it" \
#     --training_steps 20000

# CUDA_VISIBLE_DEVICES=3 python train_dqn.py \
#     --task_type "modified_image_quality" \
#     --model_name "google/gemma-3-4b-it" \
#     --training_steps 20000
