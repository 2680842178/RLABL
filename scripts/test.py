import gymnasium as gym
import matplotlib.pyplot as plt
from utils.env import RandomMinigridEnv, RandomMinigridEnvHavekey
import torch

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)
fig_path = "./test_env_figs/"
# 测试无钥匙的环境

for i in range(5):
    env = RandomMinigridEnvHavekey(curriculum=1, fixed_map=i)
    obs, _ = env.reset()
    img = obs['image']
    plt.imsave(fig_path + f"test_fig_1_{i}.bmp", img)

for i in range(5):
    env = RandomMinigridEnvHavekey(curriculum=2, fixed_map=20+i)
    obs, _ = env.reset()
    img = obs['image']
    plt.imsave(fig_path + f"test_fig_2_{i}.bmp", img)

for i in range(5):
    env = RandomMinigridEnv(curriculum=3, fixed_map=40+i)
    obs, _ = env.reset()
    img = obs['image']
    plt.imsave(fig_path + f"test_fig_3_{i}.bmp", img)


MAP_3="map_grid=   x, x, x, x, x, x, x, x, x, x, x
            x, x, x, x, x, x, x, x, x, x, x
            x, E, E, E, E, x, -, -, -, G, x
            x, -, -, -, -, x, -, -, -, -, x
            x, -, -, K, -, x, -, -, -, -, x
            x, -, -, -, -, x, D, x, x, x, x
            x, -, -, -, -, -, -, -, -, -, x
            x, E, -, -, -, -, -, -, -, -, x
            x, E, -, -, -, -, -, -, -, -, x
            x, E, -, -, -, S, -, E, E, E, x
            x, x, x, x, x, x, x, x, x, x, x"