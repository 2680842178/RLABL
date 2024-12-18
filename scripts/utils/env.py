import gymnasium as gym
import copy
from minigrid.core.grid import Grid
from minigrid.minigrid_env import MiniGridEnv
from gymnasium import spaces
from gymnasium.core import ObservationWrapper, Wrapper

class TestWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(5)
        self.action_mapping = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 5
        }
        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, 64, 64),  # number of cells
            dtype="uint8",
        )
        new_spaces = self.observation_space.spaces.copy()
        new_spaces["image"] = new_image_space

    def step(self, action):
        if hasattr(action, 'item'):
            action = action.item()
        mapped_action = self.action_mapping[action]
        return self.env.step(mapped_action)

    def observation(self, obs):
        return {**obs, "image": obs['image']}

class RandomEnvWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def _gen_grid(self, width, height):
        pass

    def reset(self):
        return super().reset()

def make_env(env_key, seed=None, render_mode="rgb_array"):
    env = gym.make(env_key, render_mode=render_mode)
    # env = TestWrapper(env)
    env.reset(seed=seed)
    return env

def copy_env(copied_env, env_key, seed=None, render_mode="rgb_array"):
    env_code = copied_env.grid.encode()
    new_env = gym.make(env_key, render_mode=render_mode)
    new_env.grid, _ = new_env.grid.decode(env_code)
    new_env.agent_pos = copy.deepcopy(copied_env.agent_pos)
    new_env.agent_dir = copy.deepcopy(copied_env.agent_dir)
    new_env.reset(seed=seed)
    return new_env