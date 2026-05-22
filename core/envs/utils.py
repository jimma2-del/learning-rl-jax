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