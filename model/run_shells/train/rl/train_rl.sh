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

#save_base_dir="/mnt/cephfs/hjh/train_record/nlp/open-assistant"

#----------------
# 多卡
#----------------
localhost="202.168.100.178"
export TRITON_HOST_RM=${localhost}:8002/andreaskoepf-oasst-rm-2-pythia-1.4b-10000
export TRITON_HOST_REF=${localhost}:8005/OpenAssistant/stablelm-7b-sft-v7-epoch-3

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  OMP_NUM_THREADS=1 accelerate launch --main_process_port 29501 \
  --config_file model/model_training/configs/accelerate_config.yaml --num_processes 6 \
  model/model_training/trainer_rl.py \
  --configs defaults defaults_rlhf llama_rlhf oasst_export_latin_cyrillic_rlhf
