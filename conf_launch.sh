#!/bin/bash

export CUDA_VISIBLE_DEVICES="2,3"
export VLLM_ATTENTION_BACKEND=XFORMERS

# python3 examples/data_preprocess/medqa_conf.py --local_dir ~/data/medqa_conf
# python3 examples/data_preprocess/mmlu_baseline.py --local_dir ~/data/mmlu_baseline

CONFIG_NAME="conf_medqa_config"
experiment_name="test_conf_1_5B_sin"
train_from_scratch=true

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
    2>&1

