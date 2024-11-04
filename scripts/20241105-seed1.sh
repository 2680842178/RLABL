#!/bin/sh
CONFIGMAP="configmap.config"
START_LINE=15
END_LINE=22

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
           x, E, -, x, D, x, x, x
           x, E, S, -, -, -, E, x
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

MODEL_NAME=20241105-seed1
LR=0.0002

sed -i "${START_LINE},${END_LINE}d" $CONFIGMAP
printf "%s\n" "$MAP_1" >> $CONFIGMAP
python discover_anomaly.py --task-config task1 --discover 0 --algo ppo --env MiniGrid-ConfigWorld-v0-havekey --lr $LR --AnomalyNN test_8 --model $MODEL_NAME --discount 0.995 --frames 40000
# add door to the map
sed -i "${START_LINE},${END_LINE}d" $CONFIGMAP
printf "%s\n" "$MAP_2" >> $CONFIGMAP
python discover_anomaly.py --task-config task1 --discover 1 --algo ppo --env MiniGrid-ConfigWorld-v0-havekey --lr $LR --AnomalyNN test_8 --model $MODEL_NAME --discount 0.995 --frames 80000
# add key to the map
sed -i "${START_LINE},${END_LINE}d" $CONFIGMAP
printf "%s\n" "$MAP_3" >> $CONFIGMAP
python discover_anomaly.py --task-config task2 --discover 1 --algo ppo --env MiniGrid-ConfigWorld-v0 --lr $LR --AnomalyNN test_8 --model $MODEL_NAME --discount 0.995 --frames 120000

# # 替换配置文件中的地图
# sed -i "${START_LINE},${END_LINE}d" $CONFIGMAP
# printf "%s\n" "$MAP_3" >> $CONFIGMAP