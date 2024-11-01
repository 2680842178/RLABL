import pygame
import gymnasium as gym
from gymnasium import Env
import math
import random
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# from collections import namedtuple, deque
# from itertools import count
# import torch
import numpy as np
# from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper,RGBImgPartialObsWrapper
# from collections import deque
# from rl_agent import Agent
# from rl_train import pre_process,init_state
# import time
# import queue
import cv2
import torch
import os

# for i in range(5):
#     env = gym.make('MiniGrid-ConfigWorld-v0', reward_type=i, render_mode='rgb_array')
#     obs = env.reset()
#     obs = env.render()
#     cv2.imwrite(f'./env_images/reward_type_{i}.png', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

env = gym.make('MiniGrid-ConfigWorld-v0-havekey', render_mode='rgb_array')
obs, _ = env.reset()

print(obs['image'])   
plt.imshow(obs['image'])
plt.show()

