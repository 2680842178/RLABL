from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Goal, Lava

class ConfigWorldEnv(MiniGridEnv):

    """
    ## Registered Configurations

    - `MiniGrid-ConfigWorld-v0`

    """

    def __init__(self, size=19, max_steps: int | None = None,**kwargs):

        self.size = size
        if max_steps is None:
            max_steps = 10 * size
        mission_space = MissionSpace(
            mission_func=self._gen_mission,

        )
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

        from configparser import ConfigParser

        config = ConfigParser()
        config.read('configmap.config', encoding='UTF-8')
        string = config['map']['map_grid']
        lines = string.split("\n")
        map = []
        for i in lines:
            i = i.replace(" ", "")
            map.append(i.split(","))


        self.size=len(map)
        width  = len(map)
        height = len(map)
        # Create the grid
        self.grid = Grid(width, height)
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

                if(map[i][j]== 'K'):
                    self.grid.set(j,i, Key("blue"))

                if(map[i][j]== 'E'):
                    self.grid.vert_wall(j,i,1, Lava)


        # Generate the mission string
        self.mission = (
            "get the key from the room, "
            "unlock the door and "
            "go to the goal"
        )
