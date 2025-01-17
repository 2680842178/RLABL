import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from utils.env import CustomMinigridEnv
import torch
import cv2

def preprocess(obss):
    obss = cv2.resize(obss, (300, 300), interpolation=cv2.INTER_NEAREST)
    obs_list = [obss]
    obs_list = np.array(obs_list)
    return torch.tensor(obs_list, device=device, dtype=torch.float)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
fig_path = "./test_env_figs/"
# 测试无钥匙的环境

for i in range(5):
    env = CustomMinigridEnv(curriculum=1,random_num=1, config_path="configmap.config")
    obs, _ = env.reset()
    print(i)
    next_obs, _, _, _, _ = env.step(i)


    # obs = preprocess(obs['image'])
    # next_obs = preprocess(next_obs['image'])

    # mutation = next_obs[0] - obs[0]
    mutation = next_obs['image'] - obs['image']
    # mutation = np.squeeze(mutation).cpu().numpy().astype(np.uint8)
    plt.imshow(mutation)
    plt.show()

    # plt.imsave(fig_path + f"test_fig_1_{i}.bmp", img)

# for i in range(5):
#     env = RandomMinigridEnvHavekey(curriculum=2, fixed_map=20+i)
#     obs, _ = env.reset()
#     img = obs['image']
#     plt.imsave(fig_path + f"test_fig_2_{i}.bmp", img)

# for i in range(5):
#     env = RandomMinigridEnv(curriculum=3, fixed_map=40+i)
#     obs, _ = env.reset()
#     img = obs['image']
#     plt.imsave(fig_path + f"test_fig_3_{i}.bmp", img)


# MAP_3="map_grid=   x, x, x, x, x, x, x, x, x, x, x
#             x, x, x, x, x, x, x, x, x, x, x
#             x, E, E, E, E, x, -, -, -, G, x
#             x, -, -, -, -, x, -, -, -, -, x
#             x, -, -, K, -, x, -, -, -, -, x
#             x, -, -, -, -, x, D, x, x, x, x
#             x, -, -, -, -, -, -, -, -, -, x
#             x, E, -, -, -, -, -, -, -, -, x
#             x, E, -, -, -, -, -, -, -, -, x
#             x, E, -, -, -, S, -, E, E, E, x
#             x, x, x, x, x, x, x, x, x, x, x"