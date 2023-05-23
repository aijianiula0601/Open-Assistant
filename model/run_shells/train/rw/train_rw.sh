#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit


#--------------------------------------------------------------------
# 切换环境
# conda activate open-assistant
# 环境安装：按照model/model_training的README.md来安装
#--------------------------------------------------------------------

cd ../../../../



save_base_dir="/mnt/cephfs/hjh/train_record/nlp/open-assistant"
dataset_dir="${save_base_dir}/dataset"
train_output_dir="${save_base_dir}/rw_outputs"
log_dir="${train_output_dir}/run_logs"
deepspeed_config="$(pwd)/model/run_shells/deepspeed_config.json"
config_file="$(pwd)/model/model_training/configs/config_rm.yaml"
wandb_entity="rw"
random_port=12343


rm -rf ${train_output_dir}

#----------------
# 单卡
#----------------
python model/model_training/trainer_rm.py \
--configs defaults_rm oasst-rm-1-pythia-6B \
--wandb-entity ${wandb_entity} \
--cache_dir ${dataset_dir} \
--output_dir ${train_output_dir} \
--deepspeed \
--deepspeed_config ${deepspeed_config}

#----------------
# 多卡
#----------------
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --nproc_per_node=4 --master_port=${random_port} model/model_training/trainer_rm.py \
--configs defaults llama-7b webgpt_dataset_only \
--cache_dir ${dataset_dir} \
--output_dir ${train_output_dir} \
--deepspeed \
--deepspeed_config "$(pwd)/model/run_shells/deepspeed_config.json" \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 8 \
--num_train_epochs 3 \
--logging_steps 2 \
--save_total_limit 3 \
--use_flash_attention false \
--log_dir ${log_dir} \
--show_dataset_stats