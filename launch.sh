#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"
export VLLM_ATTENTION_BACKEND=XFORMERS

# python3 examples/data_preprocess/medqa_conf.py --local_dir ~/data/medqa_conf

CONFIG_NAME="medqa_config_3B"
experiment_name="test_think_tags_1.5B"
train_from_scratch=true
max_token_len=8192

# Count the number of GPUs specified in CUDA_VISIBLE_DEVICES
IFS=',' read -ra gpu_array <<< "$CUDA_VISIBLE_DEVICES"
n_gpus=${#gpu_array[@]}

if [ "$train_from_scratch" = true ]; then
    resume_mode="disable"
else
    resume_mode="auto"
fi


HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" \
    +trainer.experiment_name="$experiment_name" \
    +trainer.resume_mode="$resume_mode" \
    +trainer.n_gpus_per_node="$n_gpus" \
    ++actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$(( max_token_len * 1 ))" \
    ++actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$(( max_token_len * 1 ))" \
    ++actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$(( max_token_len * 1 ))" \
    ++critic.forward_max_token_len_per_gpu="$(( max_token_len * 2 ))" \
    ++critic.ppo_max_token_len_per_gpu="$(( max_token_len * 2 ))" \
    2>&1

