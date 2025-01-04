#!/bin/sh
# 此处需要参数：0表示不删除旧模型和配置，1表示删除旧模型和配置
if [ "$#" -ne 1 ]; then
  echo "Need a parameter to decide whether to delete old model and config: 0 for no(save), 1 for yes(delete)"
  exit 1
fi 

# 获取系统可用的GPU数量
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
echo "Available GPUs: $NUM_GPUS"

# 初始化任务配置文件：单目标状态，3个节点，2个边，1个agent
START_CONFIG_CONTENT="graph:  
  nodes:
    - 0
    - 1
    - 2
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

# 设置模型名称和配置文件夹
MODEL_NAME=20241230-seed1-multigpu
MODEL_CONFIG_FOLDER=config/$MODEL_NAME

if [ "$1" == "1" ]; then
  echo "Warning: Deleting old model and config..."
  rm -rf $MODEL_CONFIG_FOLDER
  rm -rf storage/$MODEL_NAME
else
  echo "Use old model and config"
fi

if [ ! -d $MODEL_CONFIG_FOLDER ]; then
  echo "The folder $MODEL_CONFIG_FOLDER does not exist, creating it..."
  mkdir $MODEL_CONFIG_FOLDER
  mkdir $MODEL_CONFIG_FOLDER/task1
  touch $MODEL_CONFIG_FOLDER/task1/config.yaml
  NEW_TASK_CONFIG=$MODEL_CONFIG_FOLDER/task1/config.yaml
  printf "%s\n" "$START_CONFIG_CONTENT" >> $NEW_TASK_CONFIG
else
  echo "The folder $MODEL_CONFIG_FOLDER already exists."
fi

CONFIGMAP="configmap.config"
START_LINE=15
END_LINE=22

# 三个任务地图
MAP_1="map_grid = x, x, x, x, x, x, x, x
                  x, x, x, x, x, x, x, x
                  x, -, E, x, E, E, G, x
                  x, -, E, x, -, -, -, x
                  x, E, -, x, -, x, x, x
                  x, E, -, -, -, -, E, x
                  x, E, E, E, -, S, E, x
                  x, x, x, x, x, x, x, x"

MAP_2="map_grid = x, x, x, x, x, x, x, x
                  x, x, x, x, x, x, x, x
                  x, -, E, x, E, E, G, x
                  x, -, E, x, -, -, -, x
                  x, E, S, x, D, x, x, x
                  x, E, -, -, -, -, E, x
                  x, E, E, E, -, -, E, x
                  x, x, x, x, x, x, x, x"

MAP_3="map_grid = x, x, x, x, x, x, x, x
                  x, x, x, x, x, x, x, x
                  x, -, E, x, E, E, G, x
                  x, -, E, x, -, -, -, x
                  x, E, -, x, D, x, x, x
                  x, E, K, -, -, -, E, x
                  x, E, E, E, -, S, E, x
                  x, x, x, x, x, x, x, x"

LR=0.0001
DISCOUNT=0.99
ALGO=dqn
EPOCHS=16
BATCH_SIZE=128
FRAMES_PER_PROC=512

# 修改为直接运行训练脚本
run_training() {
    # 创建GPU设备列表字符串 (例如: "0,1,2")
    GPU_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
    echo "Using GPUs: $GPU_DEVICES"
    
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES python discover_anomaly.py \
        --task-config $1 \
        --discover $2 \
        --algo $ALGO \
        --env $3 \
        --lr $LR \
        --AnomalyNN test_8 \
        --model $MODEL_NAME \
        --discount $DISCOUNT \
        --epochs $EPOCHS \
        --frames-per-proc $FRAMES_PER_PROC \
        --frames $4
}

# 执行训练任务
sed -i "${START_LINE},${END_LINE}d" $CONFIGMAP
printf "%s\n" "$MAP_1" >> $CONFIGMAP
run_training "task1" 0 "MiniGrid-ConfigWorld-v0-havekey" 100000

# add door to the map
sed -i "${START_LINE},${END_LINE}d" $CONFIGMAP
printf "%s\n" "$MAP_2" >> $CONFIGMAP
run_training "task1" 1 "MiniGrid-ConfigWorld-v0-havekey" 200000

# add key to the map
sed -i "${START_LINE},${END_LINE}d" $CONFIGMAP
printf "%s\n" "$MAP_3" >> $CONFIGMAP
run_training "task2" 1 "MiniGrid-ConfigWorld-v0" 300000