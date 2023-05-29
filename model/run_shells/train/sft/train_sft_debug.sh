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

echo "pwd:$(pwd)"

save_base_dir="/mnt/cephfs/hjh/train_record/nlp/open-assistant"
dataset_dir="${save_base_dir}/dataset"
train_output_dir="${save_base_dir}/stft_outputs"
log_dir="${train_output_dir}/run_logs"
deepspeed_config="$(pwd)/model/run_shells/deepspeed_config.json"
random_port=12343

#rm -rf ${train_output_dir}
mkdir -p ${train_output_dir}

# export shared modules
export PYTHONPATH=$PYTHONPATH:../../oasst-shared


#----------------
# 多卡
#----------------

#导出一个模型做测试
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
torchrun --nproc_per_node=6 --master_port=${random_port} model/model_training/trainer_sft.py \
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
  --save_steps 5 \
  --show_dataset_stats
