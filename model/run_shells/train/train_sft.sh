#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit


#--------------------------------------------------------------------
# 切换环境
# conda activate colossalai
# transformer安装(报错：cannot import name 'LlamaConfig' from 'transformers')：
#   pip install git+https://github.com/huggingface/transformers
#--------------------------------------------------------------------

cd ../../../


deepspeed_config="/root/workspace/hjh/pycharm_projects/nlp/stanford_alpaca/run_shells/train/deepspeed_config.json"

save_base_dir="/mnt/cephfs/hjh/train_record/nlp/open-assistant"
dataset_dir="${save_base_dir}/dataset"
train_output_dir="${save_base_dir}/train_stft_outputs"
save_out_dir="${train_output_dir}/sft_train_output"

export DATA_PATH=${dataset_dir}
export MODEL_PATH=${save_base_dir}

# export shared modules
export PYTHONPATH=$PYTHONPATH:../../oasst-shared

#python model/model_training/trainer_sft.py --configs debug webgpt_dataset_only per_digit_tokens --cache_dir ${dataset_dir} --output_dir ${save_out_dir}
# if you want to use wandb, add
#--wandb_entity your_username/team_name

#python model/model_training/trainer_sft.py --configs defaults galactica-125m --cache_dir $DATA_PATH --output_dir $MODEL_PATH/sft_model

python model/model_training/trainer_sft.py \
--configs debug galactica-125m webgpt_dataset_only \
--cache_dir $DATA_PATH \
--output_dir $MODEL_PATH/sft_model \
--deepspeed \
--deepspeed_config ${deepspeed_config}
