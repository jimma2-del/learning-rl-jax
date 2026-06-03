from abc import ABC, abstractmethod

import jax.numpy as jnp
import jax

from jax.typing import ArrayLike
from chex import dataclass
from typing import TypeVar, Generic, Protocol, TypeAlias

from flax import nnx

from core.envs.utils import Policy, Critic

TScheduleValue = TypeVar('TScheduleValue')

class Schedule(Generic[TScheduleValue], Protocol):
    def __call__(self, steps: int) -> TScheduleValue:
        ...

Scheduleable: TypeAlias = TScheduleValue | Schedule[TScheduleValue]

class PolicyNetwork(Policy, nnx.Module): 
    ...

class CriticNetwork(Critic, nnx.Module): 
    ...

"""UNOFFICIAL Algo spec; not currently enforced, subject to change

class Algo(Generic[TTrainingState, TPolicy, TEnvState, TEnvObs, TEnvAction]):
    
    attributes:
        env
        hyperparameters? should this be standardized?

    methods:
        __init__(env, *nonstandard params (eg. hyperparameters))
        get_action(rngs, policy, obs, deterministic flag?, optional params) -> TEnvAction
        init_training_state(rngs, optional params (eg. network, replay buffer state, prefill steps)) -> TTrainingState
        train_epoch(rngs, training_state, epoch_steps) -> TTrainingState, metrics (probably dict? dataclass overkill?)

        create_default_policy(rngs)? can be used as dummy for loading

        rollout(rngs, policy, iter, initial_env_states (optional), optional params) -> Transition?
            - should this be standardized/public? or private implementation detail


@chex.dataclass
class TrainingState(ABC, Generic[TEnvState, TPolicy]): should this be standardized?
    attributes: steps, env_states, policy
"""