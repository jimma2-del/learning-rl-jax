from abc import ABC, abstractmethod

import jax.numpy as jnp
import jax

from jax.typing import ArrayLike
from chex import dataclass
from typing import TypeVar, Generic, Protocol, TypeAlias

TScheduleValue = TypeVar('TScheduleValue')

class Schedule(Generic[TScheduleValue], Protocol):
    def __call__(self, steps: int) -> TScheduleValue:
        ...

Scheduleable: TypeAlias = TScheduleValue | Schedule[TScheduleValue]

def resolve_scheduleable(scheduleable: Scheduleable[TScheduleValue], steps: int) -> TScheduleValue:
    if callable(scheduleable): return scheduleable(steps)
    return scheduleable

"""UNOFFICIAL Algo spec; not currently enforced, subject to change

class Algo(Generic[TTrainingState, TPolicy, TEnvState, TEnvObs, TEnvAction]):
    
    attributes:
        env
        hyperparameters? should this be standardized?

    methods:
        __init__(env, *nonstandard params (eg. hyperparameters))
        get_action(rngs, policy, obs, optional params) -> TEnvAction
        init_training_state(rngs, optional params (eg. network, replay buffer state, prefill steps)) -> TTrainingState
        train_epoch(rngs, training_state, epoch_steps) -> TTrainingState, metrics (probably dict? dataclass overkill?)

        rollout(rngs, policy, iter, initial_env_states (optional), optional params) -> Transition?
            - should this be standardized/public? or private implementation detail


@chex.dataclass
class TrainingState(ABC, Generic[TEnvState, TPolicy]): should this be standardized?
    attributes: steps, env_states, policy
"""