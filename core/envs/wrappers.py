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

    def reset(self, key: chex.PRNGKey) -> tuple[TEnvState, dict[Any, Any]]:
        return self.env.reset(key)

    def step(self, key: chex.PRNGKey, state: TEnvState, action: TEnvAction) \
        -> tuple[TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        return self.env.step(key, state, action)

    def get_obs(self, key: chex.PRNGKey, state: TEnvState) -> TEnvObs:
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
    """Normalizes observations to [-1, 1) using the range (low, high) of the observation space.
    Ignores unbounded (infinite low/high) leaves, keeping the values unaltered.
    """

    def __init__(self, 
        env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
        normalize_obs_space: Space[TEnvObs] | None = None
    ):
        super().__init__(env)

        self.normalize_obs_space: Space[TEnvObs] = normalize_obs_space \
            if normalize_obs_space is not None else super().observation_space

        chex.assert_trees_all_equal_structs(env.observation_space.low, self.normalize_obs_space.low,
            "`ranges` treedef does not match with `env.observation space`.")

        self.obs_ranges = jax.tree.map(lambda high, low: high - low, 
            self.normalize_obs_space.high, self.normalize_obs_space.low)
        self.obs_centers = jax.tree.map(lambda high, low: (high+low) / 2, 
            self.normalize_obs_space.high, self.normalize_obs_space.low)

    def get_obs(self, key: chex.PRNGKey, state: TEnvState) -> jax.Array:
        obs = super().get_obs(key, state)

        normalized = jax.tree.map(lambda obs, center, range: (obs-center) / range * 2, 
            obs, self.obs_centers, self.obs_ranges)

        return jax.tree.map(lambda normed, orig: jnp.where(jnp.isfinite(normed), normed, orig),
            normalized, obs)

    @property
    def observation_space(self) -> Space[TEnvObs]:
        return Space(
            low = jax.tree.map(lambda leaf, range: jnp.where(jnp.isfinite(range), -jnp.ones_like(leaf), leaf),
                self.normalize_obs_space.low, self.obs_ranges),
            high = jax.tree.map(lambda leaf, range: jnp.where(jnp.isfinite(range), jnp.ones_like(leaf), leaf),
                self.normalize_obs_space.high, self.obs_ranges)
        )