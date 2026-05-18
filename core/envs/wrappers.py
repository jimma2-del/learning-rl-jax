from typing import Any, Generic
from typing_extensions import TypeVar
from abc import ABC, abstractmethod

import chex

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
    def unwrapped(self) -> Environment: # there is no way to type this properly
        """Get the underlying Environment, without any wrappers."""

        if isinstance(self.env, Wrapper):
            return self.env.unwrapped

        return self.env

    # forwards all Environment methods/properties to the internal env by default

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

class ObsRangeNormalizeWrapper(
    Generic[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
    Wrapper[TEnvState, TEnvObs, TEnvAction, TRenderFrame]
):
    """Normalizes the observations using the range (low, high) of the observation space.

    Centers observations around the midpoint of the observation space.
    Approximates standard deviation using half-range.

    IMPORTANT: Does not properly handle unbounded leaves (infinite low/high).

    """

    def __init__(self, 
        env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
        ranges: Space[TEnvObs] | None = None
    ):
        super().__init__(env)

        self.ranges: Space[TEnvObs] = ranges if ranges is not None else super().observation_space
        chex.assert_trees_all_equal_structs(env.observation_space.low, self.ranges.low,
            "`ranges` treedef does not match with `env.observation space`.")

    def get_obs(self, key: jax.Array, state: TEnvState) -> jax.Array:
        obs = super().get_obs(key, state)

        mean = jax.tree.map(lambda high, low: (high+low) / 2, self.ranges.high, self.ranges.low)
        std = jax.tree.map(lambda high, low: (high-low) / 2, self.ranges.high, self.ranges.low)
            # simple half-range approximation

        return jax.tree.map(lambda obs, mean, std: (obs-mean) / std, obs, mean, std)

    @property
    def observation_space(self) -> Space[TEnvObs]:
        return Space(
            low = jax.tree.map(lambda leaf: jnp.full_like(leaf, -2), self.ranges.low), 
            high = jax.tree.map(lambda leaf: jnp.full_like(leaf, 2), self.ranges.high)
        )