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


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.conv = nn.Sequential( \
            nn.Conv2d(state_size[1], 32, kernel_size=12, stride=6),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=9, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU()
            )

        self.fc = nn.Sequential(
            nn.Linear(256*3*3, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        conv_out = self.conv(state).view(state.size()[0], -1)
        return self.fc(conv_out)




if __name__ == '__main__':

    state = torch.randn(16, 3, 300, 300)  # (batch_size, color_channel, img_height,img_width)
    print(state)
    state_size = state.size()
    print(state_size)
    cnn_model = QNetwork(state_size, action_size=4, seed=1)
    outputs = cnn_model(state)
    print(outputs)
