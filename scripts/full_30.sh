
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
for i in $(seq 1 30); do
  # 生成唯一的模型名
  MODEL_NAME="20241202-seed1-discover-${i}"
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
  python discover_anomaly.py --task-config task1 --discover 0 --algo $ALGO --env MiniGrid-ConfigWorld-v0-havekey --lr $LR --AnomalyNN test_8 --model $MODEL_NAME --discount $DISCOUNT --epochs $EPOCHS --frames-per-proc $FRAMES_PER_PROC --frames 100000

  sed -i "${START_LINE},${END_LINE}d" $CONFIGMAP
  printf "%s\n" "$MAP_2" >> $CONFIGMAP
  python discover_anomaly.py --task-config task1 --discover 1 --algo $ALGO --env MiniGrid-ConfigWorld-v0-havekey --lr $LR --AnomalyNN test_8 --model $MODEL_NAME --discount $DISCOUNT --epochs $EPOCHS --frames-per-proc $FRAMES_PER_PROC --frames 200000

  sed -i "${START_LINE},${END_LINE}d" $CONFIGMAP
  printf "%s\n" "$MAP_3" >> $CONFIGMAP
  python discover_anomaly.py --task-config task2 --discover 1 --algo $ALGO --env MiniGrid-ConfigWorld-v0 --lr $LR --AnomalyNN test_8 --model $MODEL_NAME --discount $DISCOUNT --epochs $EPOCHS --frames-per-proc $FRAMES_PER_PROC --frames 500000
done

# # 替换配置文件中的地图
# sed -i "${START_LINE},${END_LINE}d" $CONFIGMAP
# printf "%s\n" "$MAP_3" >> $CONFIGMAP