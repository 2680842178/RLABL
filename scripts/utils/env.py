import gymnasium as gym
import copy
from minigrid.core.grid import Grid
from minigrid.minigrid_env import MiniGridEnv

def make_env(env_key, seed=None, render_mode="rgb_array"):
    env = gym.make(env_key, render_mode=render_mode)
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