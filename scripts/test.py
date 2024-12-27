import gymnasium as gym
import matplotlib.pyplot as plt
from utils.env import RandomMinigridEnv
import torch

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)
# fig_path = "./test_env_figs/"
# # 测试无钥匙的环境
# env = gym.make('MiniGrid-ConfigWorld-v0', render_mode='rgb_array')
# env = RandomEnvWrapper(env)
# for i in range(10):
#     obs, _ = env.reset()
#     img = obs['image']
#     plt.imsave(fig_path + f"test_fig_nokey_{i}.bmp", img)

# env = gym.make('MiniGrid-ConfigWorld-v0', render_mode='rgb_array')
# for i in range(10):
#     obs, _ = env.reset()
#     img = obs['image']
#     plt.imsave(fig_path + f"test_fig_havekey_{i}.bmp", img)


