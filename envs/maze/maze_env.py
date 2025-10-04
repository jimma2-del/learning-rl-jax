import os

import jax.numpy as jnp
from jax import random
import jax

import numpy as np

from gymnasium import Env
from gymnasium.spaces import MultiDiscrete, Discrete

class MazeEnv(Env):
    MAP_FILE = os.path.join(os.path.dirname(__file__), "maps", "basic_goal.txt")
    MAP_RAW = None

    ROWS_DELIMITER = "\n\n"
    COLS_DELIMITER = "   "
    WALL_TILE = "WWW"
    END_TILE = "END"

    MAP_SHAPE = None
    TILE_REWARDS = None # reward given upon moving to the tile
    TILE_IS_PASSABLE = None # can move through, ie. not a wall
    TILE_IS_END = None # whether to end the episode upon moving to the tile
    SPAWNPOINTS = None # valid starting positions for reset()

    # highest non-negative END tile becomes the goal
    GOAL_POS = None
    GOAL_REWARD = 0

    ACTIONS = jnp.asarray(( (-1,0), (1,0), (0,-1), (0,1) ), dtype="int32")

    MIN_COORDS = jnp.asarray((0, 0), dtype="int32")
    MAX_COORDS = None

    MAX_STEPS = 50 #100

    instances = 0

    @staticmethod
    def initialize_class():
        MazeEnv.load_map()

    @staticmethod
    def load_map():
        with open(MazeEnv.MAP_FILE, "r") as f:
            MazeEnv.MAP_RAW = f.read()

        rows = MazeEnv.MAP_RAW.split(MazeEnv.ROWS_DELIMITER)

        num_cols = len(rows[0].split("\n")[0].split(MazeEnv.COLS_DELIMITER))
        MazeEnv.MAP_SHAPE = (len(rows), num_cols)
        MazeEnv.MAX_COORDS = jnp.asarray(MazeEnv.MAP_SHAPE, dtype="int32") \
            - jnp.asarray((1, 1), dtype="int32") # subtract 1 because 0-indexed

        tile_rewards = np.zeros(MazeEnv.MAP_SHAPE, dtype="int32")
        tile_is_passable = np.zeros(MazeEnv.MAP_SHAPE, dtype="int8")
        tile_is_end = np.zeros(MazeEnv.MAP_SHAPE, dtype="int8")
        spawnpoints = []

        MazeEnv.GOAL_REWARD = 0

        for y, row in enumerate(rows):
            rewards, types = map(lambda a: a.split(MazeEnv.COLS_DELIMITER), row.split("\n"))

            for x, (reward, type_str) in enumerate(zip(rewards, types)):
                tile_rewards[y, x] = int(reward)
                tile_is_passable[y, x] = type_str != MazeEnv.WALL_TILE
                tile_is_end[y, x] = type_str == MazeEnv.END_TILE

                if not (type_str == MazeEnv.WALL_TILE or type_str == MazeEnv.END_TILE):
                    spawnpoints.append(jnp.asarray((y, x), dtype="int32"))

                if type_str == "END" and tile_rewards[y, x] > MazeEnv.GOAL_REWARD:
                    MazeEnv.GOAL_POS = jnp.asarray((y, x), dtype="int32")
                    MazeEnv.GOAL_REWARD = tile_rewards[y, x]

        MazeEnv.TILE_REWARDS = jnp.asarray(tile_rewards)
        MazeEnv.TILE_IS_PASSABLE = jnp.asarray(tile_is_passable)
        MazeEnv.TILE_IS_END = jnp.asarray(tile_is_end)
        MazeEnv.SPAWNPOINTS = jnp.asarray(spawnpoints)

    def __init__(self, key): 
        MazeEnv.instances += 1

        self.action_space = Discrete(4) # up, down, left, right
        self.observation_space = MultiDiscrete(MazeEnv.MAP_SHAPE) # y, x

        self.key = key

        self.pos = None # (y, x)
        self.steps = 0

    def render(self):
        array_map = list(map(lambda row: list(row), MazeEnv.MAP_RAW.split("\n")))

        r = 3*self.pos[0]
        c = 6*self.pos[1] + 3

        array_map[r][c] = "*"

        return "\n".join(map(lambda row: "".join(row),array_map))

    def reset(self):
        self.steps = 0

        self.key, subkey = random.split(self.key)
        self.pos = random.choice(subkey, MazeEnv.SPAWNPOINTS)

        # self.pos = jnp.asarray((0, 0), dtype="int32")

        return self.pos, {}

    def step(self, action):
        self.steps += 1

        self.pos = MazeEnv.update_pos(self.pos, action)

        return self.pos, MazeEnv.TILE_REWARDS[tuple(self.pos)], \
            MazeEnv.TILE_IS_END[tuple(self.pos)], self.steps >= MazeEnv.MAX_STEPS, {}

    @staticmethod
    @jax.jit
    def update_pos(pos, action):
        n_pos = pos + MazeEnv.ACTIONS[action] # move according to action

        # prevent going out of bounds
        n_pos_in_bounds = jnp.minimum(MazeEnv.MAX_COORDS, jnp.maximum(MazeEnv.MIN_COORDS, n_pos))
        move = n_pos_in_bounds - pos

        # cancel movement if new tile is a wall
        n_pos = pos + move*MazeEnv.TILE_IS_PASSABLE[tuple(n_pos)]

        return n_pos

MazeEnv.initialize_class()