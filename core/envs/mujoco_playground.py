"""
Utility for converting a MuJoCo Playground (MjxEnv) environment into an Environment following the API of this repository.
NOTE: these "wrappers" are unrelated to the Wrapper type in wrappers.py, which take in an Environment from this repository.
"""

from typing import Any

import numpy as np

import jax
from jax.typing import ArrayLike
import chex

import jax.numpy as jnp

from mujoco_playground import MjxEnv, State as MjxState

from core.envs.base import Environment, Space

class MuJoCoPlaygroundWrapper(Environment[MjxState, jax.Array, jax.Array, np.ndarray]):
    """Wrapper for MuJoCo Playground (MjxEnv) environments."""

    @staticmethod
    def combine_metrics_info(metrics: dict[Any, Any], info: dict[Any, Any]) -> dict[Any, Any]:
        comb = info.copy()
        
        for key, val in metrics.items():
            comb['metrics/' + str(key)] = val

        return comb

    def __init__(self, mjx_env: MjxEnv, render_settings: dict[Any, Any] = {}) -> None:
        """
        `mjx_env`: MjxEnv instance to convert.
            IMPORTANT: For most use cases, ensure `mjx_env` is not parallelized and does not auto reset.
                Eg. do NOT apply `mujoco_playground.wrapper.wrap_for_brax_training()`.
                However, this wrapper can handle the Brax EpisodeWrapper.
        """

        self.mjx_env = mjx_env
        self.render_settings = render_settings

        obs_max = jax.tree.map(lambda leaf: jnp.full(leaf, jnp.inf), mjx_env.observation_size)
        obs_min = jax.tree.map(lambda leaf: -leaf, obs_max)
        self._observation_space = Space(obs_min, obs_max)

        ctrl_range = jnp.array(mjx_env.mj_model.actuator_ctrlrange, dtype=jnp.float32)    
        self._action_space = Space(ctrl_range[:, 0], ctrl_range[:, 1])

    def reset(self, key: chex.PRNGKey) -> tuple[MjxState, dict[Any, Any]]:
        state = self.mjx_env.reset(key)
        return state, self.combine_metrics_info(state.metrics, state.info)

    def step(self, key: chex.PRNGKey, state: MjxState, action: jax.Array) \
        -> tuple[MjxState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        state = self.mjx_env.step(state, action)

        truncated = jnp.array(False)
        if 'truncation' in state.info:
            truncated = state.info['truncation'] 

        terminated = jnp.logical_and(state.done, jnp.logical_not(truncated))

        return state, state.reward, terminated, truncated, self.combine_metrics_info(state.metrics, state.info)

    def get_obs(self, key: chex.PRNGKey, state: MjxState) -> jax.Array:
        return state.obs

    def render(self, state: MjxState, action: ArrayLike) -> np.ndarray:
        return self.mjx_env.render([ state ], **self.render_settings)[0]

    @property
    def observation_space(self) -> Space[jax.Array]:
        return self._observation_space

    @property
    def action_space(self) -> Space[jax.Array]:
        return self._action_space

if __name__ == "__main__":
    ...