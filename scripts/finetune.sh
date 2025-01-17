#!/bin/bash

###### 每次实验都需要修改的地方 ######
PYTHON=/home/sporeking/miniconda3/envs/py312/bin/python
NUMS=50 #跑多少轮
DEVICE_ID=0 # 用哪张卡
DELETE_OLD_MODELS=0 # 0表示不删除旧模型和配置，1表示删除旧模型和配置
MODEL_NAME="ramdom-ABL-PPO-large-222-5" # 设置模型名称
CONFIGMAP="test_random_big_maps.config" # 设置地图文件:
ENV="MiniGrid-ConfigWorld-v0" # 设置环境名称
# 可选环境：MiniGrid-ConfigWorld-v0, MiniGrid-ConfigWorld-Random
# 对应固定环境和随机环境：固定环境的config地图有3项，分别是课程123的地图；随机环境的config地图有15项，课程123各5种地图
# 设置三个课程的总步数（累加关系）
TOTAL_STEPS=200000
CONTRAST_FUNC="SSIM"
###################################

### 各种超参数
LR=0.00006
DISCOUNT=0.995
ALGO=ppo
EPOCHS=8
BATCH_SIZE=128
FRAMES_PER_PROC=512

# 循环执行 10 次
for FIXED_MAP in $(seq 10 $NUMS); do
  # 生成唯一的模型名
  # if [ "$DELETE_OLD_MODELS" == "1" ]; then
  #   echo "Warning: Deleting old model and config..."
  #   rm -rf $MODEL_CONFIG_FOLDER
  #   rm -rf storage/$MODEL_NAME
  # else
  #   echo "Use old model and config"
  # fi
  
  # 修改任务配置并执行训练
  CUDA_VISIBLE_DEVICES=$DEVICE_ID $PYTHON finetune.py --fixed-map $FIXED_MAP --task-config task3 --algo $ALGO --env $ENV --lr $LR --model $MODEL_NAME --discount $DISCOUNT --epochs $EPOCHS --frames-per-proc $FRAMES_PER_PROC --frames $TOTAL_STEPS --seed $FIXED_MAP --configmap $CONFIGMAP --curriculum 3 --contrast $CONTRAST_FUNC
  if [ $? -gt 4 ]; then
    echo "Error during task 3, stopping the script."
    exit 1
  fi
done

# # 替换配置文件中的地图
# sed -i "${START_LINE},${END_LINE}d" $CONFIGMAP
# printf "%s\n" "$MAP_3" >> $CONFIGMAP