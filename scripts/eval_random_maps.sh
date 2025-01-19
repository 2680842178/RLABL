BASE_MODEL_NAME="ramdom-ABL-PPO-large-real"
ENV="MiniGrid-ConfigWorld-Random"
CONFIGMAP="test_random_big_maps.config"
FIRST=4
LAST=6

for i in $(seq $FIRST $LAST); do
    MODEL_NAME="$BASE_MODEL_NAME-${i}"
    MODEL_CONFIG_FOLDER="config/$MODEL_NAME"
    /home/sporeking/miniconda3/envs/py312/bin/python evaluate_random_maps.py --model $MODEL_NAME --env $ENV --seed $i --configmap $CONFIGMAP
done