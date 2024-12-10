import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw
from gymnasium import spaces

class ImageObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 修改观测空间为图片空间 
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(600, 600, 3), dtype=np.uint8
        )

    def observation(self, obs):
        # 调用环境的渲染功能，返回图片观测值
        img = Image.new("RGB", (160, 210), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"State: {obs}", fill=(0, 0, 0))
        return np.array(img)

    def step(self, action):
        # 获取环境的下一步数据
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 修改 obs 为图片观测
        img_obs = self.observation(obs)
        return img_obs, reward, terminated, truncated, info

    def gen_obs(self):
        return self.reset()[0]

    def reset(self, **kwargs):
        # 重置环境，并获取初始状态
        obs, info = self.env.reset(**kwargs)
        # 修改 obs 为图片观测
        img_obs = self.observation(obs)
        return img_obs, info
