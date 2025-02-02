#!/bin/sh
# 检查参数：0表示不删除旧模型和配置，1表示删除旧模型和配置
if [ "$#" -ne 1 ]; then
  echo "Need a parameter to decide whether to delete old model and config: 0 for no(save), 1 for yes(delete)"
  exit 1
fi 

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

CONFIGMAP="configmap.config"
START_LINE=15
END_LINE=22

# 三个任务地图
MAP_1="map_grid = x, x, x, x, x, x, x, x
           x, x, x, x, x, x, x, x
           x, -, E, x, E, E, G, x
           x, E, E, x, -, -, -, x
           x, -, -, x, -, x, x, x
           x, -, -, -, -, -, E, x
           x, E, E, E, -, S, E, x
           x, x, x, x, x, x, x, x"

MAP_2="map_grid = x, x, x, x, x, x, x, x
           x, x, x, x, x, x, x, x
           x, -, E, x, E, E, G, x
           x, E, E, x, -, -, -, x
           x, S, -, x, D, x, x, x
           x, -, -, -, -, -, E, x
           x, E, E, E, -, -, E, x
           x, x, x, x, x, x, x, x"

MAP_3="map_grid = x, x, x, x, x, x, x, x
           x, x, x, x, x, x, x, x
           x, -, E, x, E, E, G, x
           x, E, E, x, -, -, -, x
           x, K, -, x, D, x, x, x
           x, -, -, -, -, -, E, x
           x, E, E, E, -, S, E, x
           x, x, x, x, x, x, x, x"

LR=0.0001
DISCOUNT=0.995
ALGO=ppo
EPOCHS=8
BATCH_SIZE=128
FRAMES_PER_PROC=512

# 循环执行 20 次
for i in $(seq 25 30); do
  # 生成唯一的模型名
  MODEL_NAME="20241221-discover-ppo-${i}"
  MODEL_CONFIG_FOLDER="config/$MODEL_NAME"
  
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
  sed -i "${START_LINE},${END_LINE}d" $CONFIGMAP
  printf "%s\n" "$MAP_1" >> $CONFIGMAP
  python discover_anomaly.py --task-config task1 --discover 0 --algo $ALGO --env MiniGrid-ConfigWorld-v0-havekey --lr $LR --AnomalyNN test_8 --model $MODEL_NAME --discount $DISCOUNT --epochs $EPOCHS --frames-per-proc $FRAMES_PER_PROC --frames 100000 --seed $i

  sed -i "${START_LINE},${END_LINE}d" $CONFIGMAP
  printf "%s\n" "$MAP_2" >> $CONFIGMAP
  python discover_anomaly.py --task-config task1 --discover 1 --algo $ALGO --env MiniGrid-ConfigWorld-v0-havekey --lr $LR --AnomalyNN test_8 --model $MODEL_NAME --discount $DISCOUNT --epochs $EPOCHS --frames-per-proc $FRAMES_PER_PROC --frames 200000 --seed $i

  sed -i "${START_LINE},${END_LINE}d" $CONFIGMAP
  printf "%s\n" "$MAP_3" >> $CONFIGMAP
  python discover_anomaly.py --task-config task2 --discover 1 --algo $ALGO --env MiniGrid-ConfigWorld-v0 --lr $LR --AnomalyNN test_8 --model $MODEL_NAME --discount $DISCOUNT --epochs $EPOCHS --frames-per-proc $FRAMES_PER_PROC --frames 500000 --seed $i
done

# # 替换配置文件中的地图
# sed -i "${START_LINE},${END_LINE}d" $CONFIGMAP
# printf "%s\n" "$MAP_3" >> $CONFIGMAP