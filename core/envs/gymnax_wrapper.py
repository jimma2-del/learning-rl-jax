from typing import Any, Generic
from typing_extensions import TypeVar
from abc import ABC, abstractmethod

import functools

import jax
from jax.typing import ArrayLike

import jax.numpy as jnp

from gymnax.environments.environment import Environment as GymnaxEnv
import gymnax.environments.spaces as GymnaxSpaces

from core.envs.base import Environment, Space

def space_from_gymnax_space(gymnax_space: GymnaxSpaces.Space) -> Space:
    if isinstance(gymnax_space, GymnaxSpaces.Discrete):
        return Space(low=jnp.array(0, dtype=jnp.int32), high=jnp.array(gymnax_space.n - 1, dtype=jnp.int32))

    if isinstance(gymnax_space, GymnaxSpaces.Box):
        return Space(
            low=jnp.broadcast_to(gymnax_space.low, gymnax_space.shape).astype(jnp.float32), 
            high=jnp.broadcast_to(gymnax_space.high, gymnax_space.shape).astype(dtype=jnp.float32)
        )

    if isinstance(gymnax_space, GymnaxSpaces.Dict):
        sub_spaces = { key: space_from_gymnax_space(space) for key, space in gymnax_space.spaces.items() }
        low = { key: space.low for key, space in sub_spaces.items() }
        high = { key: space.high for key, space in sub_spaces.items() }
        return Space(low=low, high=high)

    if isinstance(gymnax_space, GymnaxSpaces.Tuple):
        sub_spaces = [ space_from_gymnax_space(space) for space in gymnax_space.spaces ]
        low = [ space.low for space in sub_spaces ]
        high = [ space.high for space in sub_spaces ]
        return Space(low=low, high=high)

    raise ValueError("Unknown Gymnax Space type.")

TEnvState = TypeVar("TEnvState")
TEnvParams = TypeVar("TEnvParams")

class GymnaxWrapper(Environment[TEnvState, ArrayLike, ArrayLike], Generic[TEnvState, TEnvParams]):
    """Wrapper for Gymnax environments."""

    def __init__(self, gymnax_env: GymnaxEnv[TEnvState, TEnvParams], gymnax_params: TEnvParams | None = None):
        self.gymnax_env = gymnax_env
        self.gymnax_params = gymnax_params

        self._observation_space = space_from_gymnax_space(self.gymnax_env.observation_space(gymnax_params))
        self._action_space = space_from_gymnax_space(self.gymnax_env.action_space(gymnax_params))

    def reset(self, key: jax.Array) -> tuple[TEnvState, dict[Any, Any]]:
        obs, state = self.gymnax_env.reset_env(key, self.gymnax_params)
        return state, {}

    def step(self, key: jax.Array, state: TEnvState, action: ArrayLike) \
        -> tuple[TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        obs, state, reward, terminated, info = self.gymnax_env.step_env(key, state, action, self.gymnax_params)
        return state, reward, terminated, False, info

    def get_obs(self, key: jax.Array, state: TEnvState) -> ArrayLike:
        return self.gymnax_env.get_obs(state=state, params=self.gymnax_params, key=key)

    @property
    def observation_space(self) -> Space[ArrayLike]:
        return self._observation_space

    @property
    def action_space(self) -> Space[ArrayLike]:
        return self._action_space

if __name__ == "__main__":
    space = space_from_gymnax_space(GymnaxSpaces.Dict({
        "foo": GymnaxSpaces.Box(-10, 12, (), jnp.float32),
        "foo2": GymnaxSpaces.Box(-10, 12, (2,3), jnp.float32),
        "bar": GymnaxSpaces.Discrete(5),
        "foobar": GymnaxSpaces.Tuple((
            GymnaxSpaces.Box(-10, 12, (), jnp.float32),
            GymnaxSpaces.Discrete(5)
        ))
    }))

    print("low", space.low)
    print("high", space.high)

    key = jax.random.key(0)
    print("sample", space.sample(key))