

#************************************************************
#***
# 把rm,sft转换为triton可用图
#***
#************************************************************



#--------------------------------------------
#修改config_rl中的llama_rlhf
#  triton_host_rm: 202.168.100.178:8002/andreaskoepf-oasst-rm-2-pythia-1.4b-10000
#  triton_host_sft: 202.168.100.178:8005/andreaskoepf-oasst-sft-4-pythia-12b-epoch-3.5
#--------------------------------------------
CUDA_VISIBLE_DEVICES=0 \
python to_triton.py --configs llama_rlhf --triton_mode rm


#--------------------------------------------
#修改config_rl中的llama_rlhf中的sft_config
#     model_name: OpenAssistant/stablelm-7b-sft-v7-epoch-3
#--------------------------------------------
CUDA_VISIBLE_DEVICES=1 \
python to_triton.py --configs llama_rlhf --triton_mode sft


#--------------------------------------------
# 采用trition启动推理服务
#--------------------------------------------

model_base_dir="/workspace/hjh/pycharm_projects/nlp/Open-Assistant/on_master/00263f20217780b54b754d753cbb6821c0b6490a/model/model_training/triton_models"
rm_model_dir="${model_base_dir}/model_store_rm"
CUDA_VISIBLE_DEVICES=0 \
tritonserver --model-repository=${rm_model_dir} --http-port 8001 --grpc-port 8002 --metrics-port 8003


stf_model_dir="${model_base_dir}/model_store_sft"
CUDA_VISIBLE_DEVICES=1 \
tritonserver --model-repository=${stf_model_dir} --http-port 8004 --grpc-port 8005 --metrics-port 8006
