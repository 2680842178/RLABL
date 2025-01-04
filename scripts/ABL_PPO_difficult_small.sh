#!/bin/bash

###### 每次实验都需要修改的地方 ######
DEVICE_ID=1 # 用哪张卡
DELETE_OLD_MODELS=0 # 0表示不删除旧模型和配置，1表示删除旧模型和配置
BASE_MODEL_NAME="20240104-ABL-PPO-difficult-small" # 设置模型名称
CONFIGMAP="difficult_small_maps.config" # 设置地图文件:
ENV="MiniGrid-ConfigWorld-v0" # 设置环境名称
# 可选环境：MiniGrid-ConfigWorld-v0, MiniGrid-ConfigWorld-Random
# 对应固定环境和随机环境：固定环境的config地图有3项，分别是课程123的地图；随机环境的config地图有15项，课程123各5种地图
# 设置三个课程的总步数（累加关系）
CURRICULUM_1_STEPS=200000
CURRICULUM_2_STEPS=400000
CURRICULUM_3_STEPS=600000
DISCOVER_STEPS=200000 # discover过程的最多步数，注意这步数是算在总步数里的，所以最好小于单个课程训练的步数。
###################################

# 初始化任务配置文件：单目标状态，3个节点，2个边，1个agent
START_CONFIG_CONTENT="graph:  
  nodes:
    - 0
    - 1
    - 2
  # 0 == die, 1 == reward. 
  edges:
    - from: 2
      to: 1
      id: 0
      with_agent: 1
    - from: 2
      to: 0
      id: 1
      with_agent: 0
  start_node: 2
agent_num: 1"

### 各种超参数
LR=0.00006
DISCOUNT=0.995
ALGO=ppo
EPOCHS=8
BATCH_SIZE=128
FRAMES_PER_PROC=512

# 循环执行 10 次
for i in $(seq 1 10); do
  # 生成唯一的模型名
  MODEL_NAME="$BASE_MODEL_NAME-${i}"
  MODEL_CONFIG_FOLDER="config/$MODEL_NAME"

  if [ "$DELETE_OLD_MODELS" == "1" ]; then
    echo "Warning: Deleting old model and config..."
    rm -rf $MODEL_CONFIG_FOLDER
    rm -rf storage/$MODEL_NAME
  else
    echo "Use old model and config"
  fi
  
  if [ ! -d $MODEL_CONFIG_FOLDER ]; then
    echo "The folder $MODEL_CONFIG_FOLDER does not exist, creating it..."
    mkdir -p $MODEL_CONFIG_FOLDER/task1
    touch $MODEL_CONFIG_FOLDER/task1/config.yaml
    NEW_TASK_CONFIG=$MODEL_CONFIG_FOLDER/task1/config.yaml
    printf "%s\n" "$START_CONFIG_CONTENT" >> $NEW_TASK_CONFIG
  else
    echo "The folder $MODEL_CONFIG_FOLDER already exists."
  fi

  # 修改任务配置并执行训练
  CUDA_VISIBLE_DEVICES=$DEVICE_ID python discover_anomaly.py --task-config task1 --discover 0 --algo $ALGO --env $ENV --lr $LR  --model $MODEL_NAME --discount $DISCOUNT --epochs $EPOCHS --frames-per-proc $FRAMES_PER_PROC --frames $CURRICULUM_1_STEPS --seed $i --configmap $CONFIGMAP --curriculum 1 --discover-steps $DISCOVER_STEPS
  if [ $? -gt 4 ]; then
    echo "Error during task 1, stopping the script."
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=$DEVICE_ID python discover_anomaly.py --task-config task1 --discover 1 --algo $ALGO --env $ENV --lr $LR  --model $MODEL_NAME --discount $DISCOUNT --epochs $EPOCHS --frames-per-proc $FRAMES_PER_PROC --frames $CURRICULUM_2_STEPS --seed $i --configmap $CONFIGMAP --curriculum 2 --discover-steps $DISCOVER_STEPS
  if [ $? -gt 4 ]; then
    echo "Error during task 2, stopping the script."
    exit 1
  fi

  CUDA_VISIBLE_DEVICES=$DEVICE_ID python discover_anomaly.py --task-config task2 --discover 1 --algo $ALGO --env $ENV --lr $LR --model $MODEL_NAME --discount $DISCOUNT --epochs $EPOCHS --frames-per-proc $FRAMES_PER_PROC --frames $CURRICULUM_3_STEPS --seed $i --configmap $CONFIGMAP --curriculum 3 --discover-steps $DISCOVER_STEPS
  if [ $? -gt 4 ]; then
    echo "Error during task 3, stopping the script."
    exit 1
  fi
done

# # 替换配置文件中的地图
# sed -i "${START_LINE},${END_LINE}d" $CONFIGMAP
# printf "%s\n" "$MAP_3" >> $CONFIGMAP