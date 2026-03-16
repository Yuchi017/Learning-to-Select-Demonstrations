CUDA_VISIBLE_DEVICES=3 python eval_dqn.py \
    --task_type "age_prediction" \
    --model_checkpoint "train_res/age_prediction/Qwen2.5-VL-7B-Instruct/run_1773628262/checkpoints/step_5000.pth" \
    --inf_model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --eval_num_samples 1000 \
    --eval_k_shots 16 \
    --eval_random \
    --eval_knn \
    --eval_dqn \