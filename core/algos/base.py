from abc import ABC, abstractmethod

import jax.numpy as jnp
import jax

from jax.typing import ArrayLike
from chex import dataclass
from typing import TypeVar, Generic, Protocol, TypeAlias

from flax import nnx

from core.envs.utils import PolicyWithRngs, PolicyWithoutRngs, CriticWithRngs, CriticWithoutRngs

TScheduleValue = TypeVar('TScheduleValue')

class Schedule(Generic[TScheduleValue], Protocol):
    def __call__(self, steps: int) -> TScheduleValue:
        ...

Scheduleable: TypeAlias = TScheduleValue | Schedule[TScheduleValue]

TEnvObs = TypeVar("TEnvObs")
TEnvAction = TypeVar("TEnvAction")

# class PolicyNetworkWithRngs(Generic[TEnvObs, TEnvAction], PolicyWithRngs[TEnvObs, TEnvAction], nnx.Module): ...
# class PolicyNetworkWithoutRngs(Generic[TEnvObs, TEnvAction], PolicyWithoutRngs[TEnvObs, TEnvAction], nnx.Module): ...
# PolicyNetwork: TypeAlias = PolicyNetworkWithRngs[TEnvObs, TEnvAction] | PolicyNetworkWithoutRngs[TEnvObs, TEnvAction]

# class CriticNetworkWithRngs(Generic[TEnvObs], CriticWithRngs[TEnvObs], nnx.Module): ...
# class CriticNetworkWithoutRngs(Generic[TEnvObs], CriticWithoutRngs[TEnvObs], nnx.Module): ...
# CriticNetwork: TypeAlias = CriticNetworkWithRngs[TEnvObs] | CriticNetworkWithoutRngs[TEnvObs]

# Python's type checker is not advanced enough for stuff like above yet
class PolicyNetworkWithRngs(Generic[TEnvObs, TEnvAction], nnx.Module):
    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs) -> TEnvAction: ...
class PolicyNetworkWithoutRngs(Generic[TEnvObs, TEnvAction], nnx.Module):
    def __call__(self, obs: TEnvObs) -> TEnvAction: ...
PolicyNetwork: TypeAlias = PolicyNetworkWithRngs[TEnvObs, TEnvAction] | PolicyNetworkWithoutRngs[TEnvObs, TEnvAction]

class CriticNetworkWithRngs(Generic[TEnvObs], nnx.Module):
    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs) -> ArrayLike: ...
class CriticNetworkWithoutRngs(Generic[TEnvObs], nnx.Module):
    def __call__(self, obs: TEnvObs) -> ArrayLike: ...
CriticNetwork: TypeAlias = CriticNetworkWithRngs[TEnvObs] | CriticNetworkWithoutRngs[TEnvObs]

"""UNOFFICIAL Algo spec; not currently enforced, subject to change

class Algo(Generic[TTrainingState, TPolicy, TEnvState, TEnvObs, TEnvAction]):
    
    attributes:
        env
        hyperparameters? should this be standardized?

    methods:
        __init__(env, *nonstandard params (eg. hyperparameters))
        get_action(rngs, policy, obs, deterministic flag?, optional params) -> TEnvAction
        init_training_state(rngs, optional params (eg. network, replay buffer state, prefill steps)) -> TTrainingState
        train(rngs, training_state, steps) -> TTrainingState, metrics dict

        create_default_policy(rngs)? can be used as dummy for loading

        rollout(rngs, policy, iter, initial_env_states (optional), optional params) -> Transition?
            - should this be standardized/public? or private implementation detail


@chex.dataclass
class TrainingState(ABC, Generic[TEnvState, TPolicy]): should this be standardized?
    attributes: steps, env_states, policy
"""