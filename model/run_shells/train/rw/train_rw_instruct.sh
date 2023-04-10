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
config_file="$(pwd)/model/reward/instructor/configs/electra-base-dis-webgpt.yml"
random_port=12343
wandb_entity="rw_instruct"

rm -rf ${train_output_dir}

#----------------
# 单卡
#----------------
#python model/reward/instructor/trainer.py ${config_file} \
#--wandb-entity ${wandb_entity} \
#--deepspeed \
#--output_dir ${train_output_dir} \
#--deepspeed_config ${deepspeed_config}

#python model/reward/instructor/trainer.py -h

#----------------
# 多卡
#----------------
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
torchrun --nproc_per_node=8 --master_port=${random_port} model/reward/instructor/trainer.py ${config_file} \
--wandb-entity ${wandb_entity} \
--output_dir ${train_output_dir} \
--deepspeed \
--deepspeed_config ${deepspeed_config}
#--per_device_train_batch_size 16 \
#--per_device_eval_batch_size 8 \
#--num_train_epochs 3 \
#--logging_steps 2 \
#--save_total_limit 3 \
#--use_flash_attention false \
#--log_dir ${log_dir} \
#--show_dataset_stats