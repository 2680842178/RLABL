import gymnasium as gym
import copy
import random
from minigrid.core.grid import Grid
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Box, Lava
from gymnasium import spaces
from gymnasium.core import ObservationWrapper, Wrapper
from configparser import ConfigParser

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

def read_maps_from_config(config_file='random_maps.config'):
    """
    读取配置文件中的所有地图，并返回一个地图列表。
    
    参数:
    - config_file: 配置文件路径
    
    返回:
    - maps: 地图列表，每张地图是一个二维列表
    """
    config = ConfigParser()
    config.read(config_file, encoding='UTF-8')
    
    maps = []
    if 'random_maps' not in config.sections():
        raise ValueError("配置文件中缺少 [random_maps] 部分。")
    
    for key in sorted(config['random_maps'].keys()):
        map_str = config['random_maps'][key]
        map_lines = map_str.strip().split('\n')
        map_grid = []
        for line in map_lines:
            # 移除前后的空白字符，并按逗号分隔
            row = [cell.strip() for cell in line.strip().split(',')]
            map_grid.append(row)
        maps.append(map_grid)
    
    return maps

class RandomMinigridEnv(MiniGridEnv):
    """
    ## Registered Configurations

    - `MiniGrid-RandomEnvWrapper-v0`
    """

    def __init__(self, size=8,config_path='random_maps.config', max_steps: int | None = None, **kwargs):
        self.size=size
        # 设置最大步数
        if max_steps is None:
            max_steps = 20 * self.size

        # 定义任务空间
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
        )
        self.config_path = config_path
        self.maps = read_maps_from_config(config_path)
        self.curriculum = kwargs.get('curriculum', 1)
        # 初始化父类，传入地图的宽度和高度
        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=max_steps,
            **kwargs,
        )
    @staticmethod
    def _gen_mission():
        return (
            "get the key from the room, "
            "unlock the door and "
            "go to the goal"
        )

    def _gen_grid(self, width, height):
        #print(f"选中的地图: {maps}")  # 调试信息
        if self.curriculum == 1:
            selected_map = random.choice(self.maps[:20])
        elif self.curriculum == 2:
            selected_map = random.choice(self.maps[20:40])
        else:
            selected_map = random.choice(self.maps[40:60])
        # print(f"选中的地图: {selected_map}")  # 调试信息
        self.selected_map = selected_map
        map_grid = self.selected_map
        self.size=len(selected_map)
        width  = len(selected_map)
        height = len(selected_map)
        self.grid = Grid(width, height)
        for i in range(1,len(selected_map)):
            for j in range(0,len(selected_map)):
                if(selected_map[i][j]== 'x'):
                    self.grid.set(j, i, Wall())

                if(selected_map[i][j]== 'S'):
                    self.agent_pos = self.place_agent(
                        top=(j, i), size=(1, 1)
                    )

                if(selected_map[i][j]== 'G'):
                    self.grid.set(j,i, Goal())

                if(selected_map[i][j]== 'D'):
                    # colors = set(COLOR_NAMES)
                    # color = self._rand_elem(sorted(colors))
                    self.grid.set(j,i, Door("blue", is_locked=True))

                if(selected_map[i][j]== 'K'):
                    self.grid.set(j,i, Key("blue"))

                if(selected_map[i][j]== 'E'):
                    self.grid.vert_wall(j,i,1, Lava)

                if(selected_map[i][j]== 'B'):
                    self.grid.set(j,i, Box("blue",Key("blue")))

        # 生成任务字符串
        self.mission = self._gen_mission()
    
    def observation(self, obs):
        return {**obs, "image": obs['image']}
   
class RandomMinigridEnvHavekey(MiniGridEnv):
    """
    ## Registered Configurations

    - `MiniGrid-RandomEnvWrapper-v0`
    """

    def __init__(self, size=8,config_path='random_maps.config', max_steps: int | None = None, **kwargs):
        self.size=size
        # 设置最大步数
        if max_steps is None:
            max_steps = 20 * self.size

        # 定义任务空间
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
        )
        self.config_path = config_path
        self.maps = read_maps_from_config(config_path)
        self.curriculum = kwargs.get('curriculum', 1)

        # 初始化父类，传入地图的宽度和高度
        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=max_steps,
            **kwargs,
        )
    @staticmethod
    def _gen_mission():
        return (
            "get the key from the room, "
            "unlock the door and "
            "go to the goal"
        )

    def _gen_grid(self, width, height):
        #print(f"选中的地图: {maps}")  # 调试信息
        if self.curriculum == 1:
            selected_map = random.choice(self.maps[:20])
        elif self.curriculum == 2:
            selected_map = random.choice(self.maps[20:40])
        else:
            selected_map = random.choice(self.maps[40:60])
        # print(f"选中的地图: {selected_map}")  # 调试信息
        self.selected_map = selected_map
        map_grid = self.selected_map
        self.size=len(map)
        width  = len(map)
        height = len(map)
        # Create the grid
        self.grid = Grid(width, height)

        self.grid.set(0,0, Key("blue"))
        self.carrying = Key("blue")

        KEYCOLOR=None
        for i in range(1,len(map)):
            for j in range(0,len(map)):
                if(map[i][j]== 'x'):
                    self.grid.set(j, i, Wall())

                if(map[i][j]== 'S'):
                    self.agent_pos = self.place_agent(
                        top=(j, i), size=(1, 1)
                    )

                if(map[i][j]== 'G'):
                    self.grid.set(j,i, Goal())

                if(map[i][j]== 'D'):
                    # colors = set(COLOR_NAMES)
                    # color = self._rand_elem(sorted(colors))
                    self.grid.set(j,i, Door("blue", is_locked=True))

                # if(map[i][j]== 'K'):
                #     self.grid.set(0,0, Key("blue"))
                if(map[i][j]== 'E'):
                    self.grid.vert_wall(j,i,1, Lava)

                if(map[i][j]== 'B'):
                    self.grid.set(j,i, Box("blue",Key("blue")))

        # 生成任务字符串
        self.mission = self._gen_mission()
    
    def observation(self, obs):
        return {**obs, "image": obs['image']}


def make_env(env_key, seed=None, render_mode="rgb_array", **kwargs):
    if env_key == "MiniGrid-ConfigWorld-Random":
        env = RandomMinigridEnv(kwargs=kwargs)
    elif env_key == "MiniGrid-ConfigWorld-Random-havekey":
        env = RandomMinigridEnvHavekey(kwargs=kwargs)
    else:
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