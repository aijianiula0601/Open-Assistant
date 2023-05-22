#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

#--------------------------------------------------------------------
# 切换环境
# conda activate open-assistant
# 环境安装：按照model/model_training的README.md来安装
# 问题记录：
# 1.No module named "oasst_data"
#     cd oasst-data
#     pip install .
#--------------------------------------------------------------------

cd ../../../../

save_base_dir="/mnt/cephfs/hjh/train_record/nlp/open-assistant"
dataset_dir="${save_base_dir}/dataset"
train_output_dir="${save_base_dir}/stft_outputs"
log_dir="${train_output_dir}/run_logs"
deepspeed_config="$(pwd)/model/run_shells/deepspeed_config.json"
random_port=12343

rm -rf ${train_output_dir}
mkdir -p ${train_output_dir}

# export shared modules
export PYTHONPATH=$PYTHONPATH:../../oasst-shared

#----------------
# 单卡
#----------------
#python model/model_training/trainer_sft.py \
#--configs debug galactica-125m webgpt_dataset_only \
#--cache_dir ${dataset_dir} \
#--output_dir ${train_output_dir} \
#--deepspeed \
#--deepspeed_config ${deepspeed_config}

#----------------
# 多卡
#----------------
#export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
torchrun --nproc_per_node=8 --master_port=${random_port} model/model_training/trainer_sft.py \
  --configs defaults llama-7b webgpt_dataset_only \
  --cache_dir ${dataset_dir} \
  --output_dir ${train_output_dir} \
  --deepspeed \
  --deepspeed_config ${deepspeed_config} \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 8 \
  --num_train_epochs 3 \
  --logging_steps 2 \
  --save_total_limit 3 \
  --use_flash_attention false \
  --log_dir ${log_dir} \
  --save_steps 50 \
  --show_dataset_stats

#export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
#torchrun --nproc_per_node=8 --master_port=${random_port} model/model_training/trainer_sft.py \
#  --configs defaults llama-7b webgpt_dataset_only \
#  --cache_dir ${dataset_dir} \
#  --log_dir ${log_dir} \
#  --output_dir ${train_output_dir} \
#  --num_train_epochs 10 \
#  --per_device_train_batch_size 16 \
#  --per_device_eval_batch_size 8 \
#  --gradient_accumulation_steps 8 \
#  --save_strategy "steps" \
#  --save_steps 50 \
#  --save_total_limit 100 \
#  --learning_rate 2e-5 \
#  --weight_decay 0. \
#  --logging_steps 1 \
#  --gradient_checkpointing True \
#  --deepspeed \
#  --deepspeed_config ${deepspeed_config}
