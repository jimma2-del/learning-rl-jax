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

@chex.dataclass(frozen=True)
class Transition(Generic[TEnvState, TEnvObs, TEnvAction]):
    state: TEnvState
    obs: TEnvObs
    action: TEnvAction
    reward: ArrayLike
    next_state: TEnvState
    terminated: ArrayLike
    truncated: ArrayLike
    info: dict[Any, Any]

def rollout_auto_reset(rngs: nnx.Rngs, 
    env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame], 
    policy: Callable[[nnx.Rngs, TEnvObs], TEnvAction], 
    iter: int, n_envs: int,
    initial_env_states: TEnvState | None = None
) -> Transition[TEnvState, TEnvObs, TEnvAction]:
    """Collect a rollout of `Transition` samples.

    Runs `n_envs` environments in parallel for `iter` iterations,
        for a total of `iter * n_envs` transitions.
    Samples actions according to `policy`.
    Initializes initial environment states if none given.

    Returns: transitions, final environment states
    """

    env = VmapAutoResetWrapper(env)

    def batched_env_step(states: TEnvState, rngs: nnx.Rngs) -> tuple[TEnvState, Transition]:
        obs = env.get_obs(rngs.env(), states)
        actions = nnx.vmap(policy)(rngs.fork(split=n_envs))(rngs, obs)
        new_states, rewards, terminated, truncated, infos = env.step(rngs.env(), states, actions)

        return (
            new_states,
            Transition(
                state=states, obs=obs, action=actions, reward=rewards, 
                next_state=infos.pop(env.NEXT_STATE_INFO_KEY),
                terminated=terminated, truncated=truncated, info=infos
            )
        )

    if initial_env_states is None:
        initial_env_states, info = env.reset(rngs.env(), num=n_envs)

    env_states, transitions = nnx.scan(batched_env_step)(initial_env_states, rngs.fork(split=iter))
    transitions = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), transitions) # flatten to remove axis 0

    return transitions, env_states

def evaluate_episodes(rngs: nnx.Rngs, 
    env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame], 
    policy: Callable[[nnx.Rngs, TEnvObs], TEnvAction], 
    episodes: int, n_envs = 8
) -> tuple[jax.Array, jax.Array]:
    """Collect a rollout of `Transition` samples.

    Runs `n_envs` environments in parallel for `iter` iterations,
        for a total of `iter * n_envs` transitions.
    Samples actions according to `policy`.
    Initializes initial environment states if none given.

    Returns: transitions, final environment states
    """

    env = VmapAutoResetWrapper(env)

    def iter(carry):
        num_eps_done, rngs, env_states, cur_returns, eps_returns, cur_lens, eps_lens = carry

        obs = env.get_obs(rngs.env(), env_states)
        actions = nnx.vmap(policy)(rngs.fork(split=n_envs))(rngs, obs)
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