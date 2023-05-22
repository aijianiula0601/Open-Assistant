#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../

llama_path="/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/pretrain_models/llama"
output_dir="${llama_path}/for_open-assistant_llama_7b"

rm -rf ${output_dir}

python /root/miniconda3/envs/open-assistant/lib/python3.10/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir ${llama_path} \
    --model_size 7B \
    --output_dir ${output_dir}