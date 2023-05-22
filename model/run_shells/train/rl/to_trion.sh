

#-----------------------------------------------------------------------
#把rm,sft转换为triton可用图
#-----------------------------------------------------------------------



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