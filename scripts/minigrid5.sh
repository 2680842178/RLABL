#!/bin/sh
# 此处需要参数：0表示不删除旧模型和配置，1表示删除旧模型和配置
if [ "$#" -ne 1 ]; then
  echo "Need a parameter to decide whether to delete old model and config: 0 for no(save), 1 for yes(delete)"
  exit 1
fi 
# 设置模型名称和配置文件夹

CONFIGMAP="configmap.config"
START_LINE=15
END_LINE=22

# 三个任务地图

MAP_1="map_grid = x, x, x, x, x, x, x, x
           x, x, x, x, x, x, x, x
           x, -, E, x, E, E, G, x
           x, E, E, x, -, -, -, x
           x, K, -, x, D, x, x, x
           x, -, -, -, -, -, E, x
           x, E, E, E, -, S, E, x
           x, x, x, x, x, x, x, x"

LR=0.0001
DISCOUNT=0.99
ALGO=dqn
EPOCHS=16
BATCH_SIZE=128
FRAMES_PER_PROC=512

for i in $(seq 1 30); do
  MODEL_NAME="20241220-seed1-fullDFA-${i}"
  MODEL_CONFIG_FOLDER="config/$MODEL_NAME"

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

  cp -r ./config/full_DFA_config_minigrid/* ./config/$MODEL_NAME

  sed -i "${START_LINE},${END_LINE}d" $CONFIGMAP
  printf "%s\n" "$MAP_1" >> $CONFIGMAP
  python discover_anomaly.py --task-config task1 --discover 0 --algo $ALGO --env MiniGrid-ConfigWorld-v0 --lr $LR --AnomalyNN test_8 --model $MODEL_NAME --discount $DISCOUNT --epochs $EPOCHS --frames-per-proc $FRAMES_PER_PROC --frames 500000
done