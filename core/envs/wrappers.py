from typing import Any, Generic
from typing_extensions import TypeVar
from abc import ABC, abstractmethod

import functools

import jax
from jax.typing import ArrayLike

import jax.numpy as jnp

from .base import Environment, Space

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")
TEnvAction = TypeVar("TEnvAction")
TRenderFrame = TypeVar("TRenderFrame", default=None)

class Wrapper(
    Generic[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
    Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame]
):

    def __init__(self, env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame]):
        self.env = env

    @property
    def unwrapped(self) -> Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame]:
        """Get the underlying Environment, without any wrappers."""

        if isinstance(self.env, Wrapper):
            return self.env.unwrapped

        return self.env

    # fowards all Environment methods/properties to the internal env by default

    def reset(self, key: jax.Array) -> tuple[TEnvState, dict[Any, Any]]:
        return self.env.reset(key)

    def step(self, key: jax.Array, state: TEnvState, action: TEnvAction) \
        -> tuple[TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        return self.env.step(key, state, action)

    def get_obs(self, key: jax.Array, state: TEnvState) -> TEnvObs:
        return self.env.get_obs(key, state)

    def render(self, state: TEnvState, action: ArrayLike) -> TRenderFrame:
        return self.env.render(state, action)

    @property
    def observation_space(self) -> Space[TEnvObs]:
        return self.env.observation_space

    @property
    def action_space(self) -> Space[TEnvAction]:
        return self.env.action_space

    @property
    def name(self) -> str:
        """Environment name."""
        return self.env.name