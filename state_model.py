import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import cv2

from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper


class StateNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(StateNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.conv = nn.Sequential(
            nn.Conv2d(state_size[1], 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU()
            )

        self.fc = nn.Sequential(
            nn.Linear(128*7*7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        conv_out = self.conv(state).view(state.size()[0], -1)
        return self.fc(conv_out)


def pre_process(observation):
    """Process (210, 160, 3) picture into (1, 84, 84)"""
    x_t = cv2.cvtColor(cv2.resize(observation, (84, 84)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(x_t, (1, 84, 84)), x_t


def stack_state(processed_obs):
    """Four frames as a state"""
    return np.stack((processed_obs, processed_obs, processed_obs, processed_obs), axis=0)


if __name__ == '__main__':
    env: MiniGridEnv = gym.make(
        "MiniGrid-ConfigWorld-v0",
        tile_size=32,
        screen_size="640",
    )
    env = RGBImgObsWrapper(env)

    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)

    obs_dict, _ = env.reset()
    obs = obs_dict['image']
    x_t, img = pre_process(obs)
    state = stack_state(img)
    print(np.shape(state[0]))

    # plt.imshow(img, cmap='gray')
    # 用cv2模块显示
    # cv2.imshow('Breakout', img)
    # cv2.waitKey(0)

    state = torch.randn(32, 4, 84, 84)  # (batch_size, color_channel, img_height,img_width)
    state_size = state.size()

    cnn_model = QNetwork(state_size, action_size=4, seed=1)
    outputs = cnn_model(state)
    print(outputs)
