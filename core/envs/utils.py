from typing import Any, Generic, Callable
from typing_extensions import TypeVar

import chex

import jax
from jax.typing import ArrayLike

import jax.numpy as jnp

from flax import nnx

from core.envs.base import Environment
from core.envs.wrappers import VmapAutoResetWrapper

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")
TEnvAction = TypeVar("TEnvAction")
TRenderFrame = TypeVar("TRenderFrame", default=None)

def evaluate_episodes(rngs: nnx.Rngs, 
    env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame], 
    policy: Callable[[nnx.Rngs, TEnvObs], TEnvAction], 
    episodes: int, n_envs = 8
) -> tuple[jax.Array, jax.Array]:
    """Runs the policy in `n_envs` environments in parallel until completing `episodes` episodes, 
    returning an array of trajectory returns and and an array of trajectory lengths.
    """

    env = VmapAutoResetWrapper(env)

    def iter(carry):
        num_eps_done, rngs, env_states, cur_returns, eps_returns, cur_lens, eps_lens = carry

        obs = env.get_obs(rngs.env(), env_states)
        actions = nnx.vmap(policy)(rngs.fork(split=n_envs), obs)
        env_states, rewards, terminated, truncated, infos = env.step(rngs.env(), env_states, actions)

        dones = jnp.logical_or(terminated, truncated)
        cur_returns = cur_returns + rewards
        cur_lens = cur_lens + 1

        # add metrics to aggregate array of eps metrics if done

        def env_aggregate(carry, x):
            num_eps_done, eps_returns, eps_lens = carry
            done, cur_return, cur_len = x

            eps_returns = eps_returns.at[num_eps_done].set(cur_return)
            eps_lens = eps_lens.at[num_eps_done].set(cur_len)

            num_eps_done = jnp.where(done, num_eps_done + 1, num_eps_done)

            return (num_eps_done, eps_returns, eps_lens), None

        (num_eps_done, eps_returns, eps_lens), _ = jax.lax.scan(env_aggregate, 
            (num_eps_done, eps_returns, eps_lens), (dones, cur_returns, cur_lens))

        # reset cur metrics for each env if done
        not_dones = jnp.logical_not(dones)
        cur_returns = cur_returns * not_dones
        cur_lens = cur_lens * not_dones

        return num_eps_done, rngs, env_states, cur_returns, eps_returns, cur_lens, eps_lens

    env_states, info = env.reset(rngs.env(), num=n_envs)

    cur_returns = jnp.empty(n_envs)
    cur_lens = jnp.empty(n_envs)

    eps_returns = jnp.empty(episodes)
    eps_lens = jnp.empty(episodes)

    num_eps_done, rngs, env_states, cur_returns, eps_returns, cur_lens, eps_lens = nnx.while_loop(
        lambda input: input[0] < episodes, 
        iter, 
        (0, rngs, env_states, cur_returns, eps_returns, cur_lens, eps_lens)
    )

    return eps_returns, eps_lens