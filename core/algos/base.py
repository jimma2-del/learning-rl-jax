from abc import ABC, abstractmethod

import jax.numpy as jnp
import jax

from jax.typing import ArrayLike
from chex import dataclass
from typing import TypeVar, Generic, Protocol, TypeAlias, Sequence

from flax import nnx

from core.utils.func_utils import optionally_pass
from core.envs.base import Space

TScheduleValue = TypeVar('TScheduleValue')

class Schedule(Generic[TScheduleValue], Protocol):
    def __call__(self, steps: int) -> TScheduleValue:
        ...

Scheduleable: TypeAlias = TScheduleValue | Schedule[TScheduleValue]

TEnvObs = TypeVar("TEnvObs")
TEnvAction = TypeVar("TEnvAction")

class PolicyWithRngs(Generic[TEnvObs, TEnvAction], Protocol):
    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs) -> TEnvAction: ...
class PolicyWithoutRngs(Generic[TEnvObs, TEnvAction], Protocol):
    def __call__(self, obs: TEnvObs) -> TEnvAction: ...
Policy: TypeAlias = PolicyWithoutRngs[TEnvObs, TEnvAction] | PolicyWithRngs[TEnvObs, TEnvAction]

class ActionDistributionActor(Generic[TEnvObs, TEnvAction], nnx.Module):
    """Actor which chooses actions by sampling from a distribution outputted by the policy."""

    def __init__(self, policy: Policy[TEnvObs, TEnvAction], action_space: Space[TEnvAction],
            deterministic: bool = False, squash_continuous: bool = True) -> None:
        self.policy = policy
        self.action_space = action_space

        self.deterministic = deterministic
        self.squash_continuous = squash_continuous

    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs | None = None,
            deterministic: bool | None = None, squash_continuous: bool | None = None) -> TEnvAction:
        """Computes an action distribution for the given observation, and samples an action from it.

        `squash_continuous`: If False, does not squash continuous values, leaving them unbounded.
            Useful, eg. for sampling raw outputs, before softplus or tanh.

        `deterministic`: If True, takes the mode of the action distribution instead of 
            sampling a random action. See `Space.sample_distribution()`.
        """

        if deterministic is None: deterministic = self.deterministic
        if squash_continuous is None: squash_continuous = self.squash_continuous

        action_dist = self.action_distribution(obs, rngs)
        return self.action_space.sample_distribution(rngs.actions(), action_dist, 
            squash_continuous=squash_continuous, deterministic=deterministic)

    def action_distribution(self, obs, rngs: nnx.Rngs | None = None) -> TEnvAction:
        """Applies the policy on the observation to compute an action distribution.
        See `Space.sample_distribution()` for details on the structure of the returned distribution."""
        return optionally_pass(self.policy, rngs=rngs)(obs)

class DiscreteQFuncWithRngs(Generic[TEnvObs], Protocol):
    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs) -> jax.Array: ...
class DiscreteQFuncWithoutRngs(Generic[TEnvObs], Protocol):
    def __call__(self, obs: TEnvObs) -> jax.Array: ...
DiscreteQFunc: TypeAlias = DiscreteQFuncWithRngs[TEnvObs] | DiscreteQFuncWithoutRngs[TEnvObs]

class GreedyQActor(Generic[TEnvObs], nnx.Module):
    """Actor which chooses discrete (scalars in the range [0, num_actions-1]) actions
        taking the action with the highest Q value out of the Q values outputted by the Q function.
    
    Supports epsilon-greedy actions, returning a random action instead with probability epsilon.
    """

    def __init__(self, q_func: DiscreteQFunc[TEnvObs], num_actions: int, 
            deterministic: bool = False, epsilon: ArrayLike = jnp.array(0.0)) -> None:
        self.q_func = q_func
        self.num_actions = int(num_actions)

        self.deterministic = deterministic
        self.epsilon = jnp.array(epsilon, dtype=jnp.float32)

    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs | None = None,
            deterministic: bool | None = None, epsilon: ArrayLike | None = None) -> ArrayLike:
        """Returns a random action with probablity `epsilon`, otherwise 
            computes a Q value for every possible action and returns the action with the highest Q value.

        `deterministic`: If True, overrides `epsilon` and always returns a greedy action.
        """

        if deterministic is None: deterministic = self.deterministic
        if epsilon is None: epsilon = self.epsilon

        greedy_action = self.greedy_action(obs, rngs=rngs)
        if deterministic: return greedy_action

        random_action = self.random_action(rngs, shape=greedy_action.shape)
        take_random_action = jax.random.uniform(rngs.actions(), shape=greedy_action.shape) < epsilon

        return jnp.where(take_random_action, random_action, greedy_action)

    def q_values(self, obs, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Applies the Q function on the observation to compute a Q value for every possible action."""
        return optionally_pass(self.q_func, rngs=rngs)(obs)

    def greedy_action(self, obs: TEnvObs, rngs: nnx.Rngs | None = None) -> ArrayLike:
        """Computes a Q value for every possible action and returns the action with the highest Q value."""
        return jnp.argmax(self.q_values(obs, rngs=rngs), axis=-1)

    def random_action(self, rngs: nnx.Rngs, shape: Sequence[int] = ()) -> ArrayLike:
        return jax.random.randint(rngs.actions(), shape=shape, minval=0, maxval=self.num_actions)

"""UNOFFICIAL Algo spec; not currently enforced, subject to change

class Algo(Generic[TTrainingState, TActor (bound=core.env.utils.Actor), TEnvState, TEnvObs, TEnvAction]):
    
    attributes:
        env
        hyperparameters? should this be standardized?

    methods:
        __init__(env, *nonstandard params (eg. hyperparameters))
        get_action(rngs, policy, obs, deterministic flag?, optional params) -> TEnvAction
        init_training_state(rngs, optional params (eg. network, replay buffer state, prefill steps)) -> TTrainingState
        train(rngs, training_state, steps) -> TTrainingState, metrics dict

        make_actor(rngs, optional params?)? 
            can be used as dummy for loading orbax checkpoints if only the actor was saved

@chex.dataclass
class TrainingState(ABC, Generic[TEnvState, TActor]): should this be standardized?
    attributes: steps: ArrayLike (jnp int scalar), env_states: TEnvState, actor: TActor

    not enforced as not all algorithms have it: `networks`, `optimizer`
        - `(optimized_?)networks` is an nnx.Module which holds all networks that are trained by `optimizer`;
            use this name for consistency
"""