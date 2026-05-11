from chex import dataclass
from jax.typing import ArrayLike
from jax import Array
from typing import Any

from os import path

import jax.numpy as jnp
import jax

import numpy as np

from core.envs.base import Environment, Space

MAP_DIR = path.join(path.dirname(__file__), "maps")
get_map_path = lambda x: path.join(MAP_DIR, x)

@dataclass(frozen=True)
class State:
    pos: jax.Array
    steps: ArrayLike

class GridworldEnv(Environment[State, Array, ArrayLike, str]):
    ROWS_DELIMITER = "\n\n"
    COLS_DELIMITER = "   "
    WALL_TILE = "WWW"
    END_TILE = "END"

    ACTIONS = jnp.asarray(( (-1,0), (1,0), (0,-1), (0,1) ), dtype=jnp.int32)

    @classmethod
    def default_map(cls, map_name: str, max_steps: int = 50):
        try:
            with open(get_map_path(map_name + ".txt"), "r") as f:
                map_data = f.read()
        except FileNotFoundError:
            raise Exception(f"Map '{map_name}' not found.")
        
        return cls(map_data, max_steps)

    def __init__(self, map_data: str, max_steps: int = 50) -> None:
        self.max_steps = max_steps

        self.map_data = map_data

        rows = map_data.split(GridworldEnv.ROWS_DELIMITER)

        num_cols = len(rows[0].split("\n")[0].split(GridworldEnv.COLS_DELIMITER))
        self.map_shape = (len(rows), num_cols)

        self.min_coords = jnp.asarray((0, 0), dtype=jnp.int32)
        self.max_coords = jnp.asarray(self.map_shape, dtype=jnp.int32) \
            - jnp.asarray((1, 1), dtype=jnp.int32) # subtract 1 because 0-indexed

        tile_rewards = np.zeros(self.map_shape, dtype="int32")
        tile_is_passable = np.zeros(self.map_shape, dtype="int8")
        tile_is_end = np.zeros(self.map_shape, dtype="int8")
        spawnpoints = []

        # highest non-negative END tile becomes the goal
        self.goal_pos = None
        self.goal_reward = 0

        for y, row in enumerate(rows):
            rewards, types = map(lambda a: a.split(GridworldEnv.COLS_DELIMITER), row.split("\n"))

            for x, (reward, type_str) in enumerate(zip(rewards, types)):
                tile_rewards[y, x] = int(reward)
                tile_is_passable[y, x] = type_str != GridworldEnv.WALL_TILE
                tile_is_end[y, x] = type_str == GridworldEnv.END_TILE

                if not (type_str == GridworldEnv.WALL_TILE or type_str == GridworldEnv.END_TILE):
                    spawnpoints.append(jnp.asarray((y, x), dtype=jnp.int32))

                if type_str == "END" and tile_rewards[y, x] > self.goal_reward:
                    self.goal_pos = jnp.asarray((y, x), dtype=jnp.int32)
                    self.goal_reward = tile_rewards[y, x]

        self.tile_rewards = jnp.asarray(tile_rewards) # reward given upon moving to the tile
        self.tile_is_passable = jnp.asarray(tile_is_passable) # can move through, ie. not a wall
        self.tile_is_end = jnp.asarray(tile_is_end) # whether to end the episode upon moving to the tile
        self.spawnpoints = jnp.asarray(spawnpoints) # valid starting positions for reset()

    def reset(self, key: Array) -> tuple[State, dict[Any, Any]]:
        pos = jax.random.choice(key, self.spawnpoints)
        # self.pos = jnp.asarray((0, 0), dtype="int32")

        return State(pos=pos, steps=0), { "goal": self.goal_pos }

    def step(self, key: Array, state: State, action: ArrayLike) \
        -> tuple[State, Array, Array, Array, dict[Any, Any]]:
        n_pos = self.update_pos(state.pos, action)
        n_steps = state.steps + 1

        return State(pos=n_pos, steps=n_steps), self.tile_rewards[tuple(n_pos)], \
            self.tile_is_end[tuple(n_pos)], n_steps >= self.max_steps, { "goal": self.goal_pos }

    def update_pos(self, pos: Array, action: ArrayLike) -> Array:
        n_pos = pos + GridworldEnv.ACTIONS[action] # move according to action

        # prevent going out of bounds
        n_pos_in_bounds = jnp.clip(n_pos, self.min_coords, self.max_coords)
        move = n_pos_in_bounds - pos

        # cancel movement if new tile is a wall
        n_pos = pos + move*self.tile_is_passable[tuple(n_pos)]

        return n_pos

    def get_obs(self, key: jax.Array, state: State) -> Array:
        return state.pos

    def render(self, state: State, Action: ArrayLike) -> Array:
        array_map = list(map(lambda row: list(row), self.map_data.split("\n")))

        r = 3*state.pos[0]
        c = 6*state.pos[1] + 3

        array_map[r][c] = "*"

        return "\n".join(map(lambda row: "".join(row), array_map))

    @property
    def observation_space(self) -> Space[Array]:
        return Space(low=self.min_coords, high=self.max_coords)

    @property
    def action_space(self) -> Space[ArrayLike]:
        return Space(low=jnp.array(0), high=jnp.array(3))


    def visualize_q_table(self, q_vals) -> str:
        result = ""

        for y, row in enumerate(q_vals):

            for x, tile in enumerate(row):
                if self.tile_is_end[y,x] or not self.tile_is_passable[y,x]:
                    result += "@"
                else:
                    best_action = np.argmax(tile)
                    result += ( "^", "v", "<", ">" )[best_action]

                result += " "

            result += "\n"

        return result