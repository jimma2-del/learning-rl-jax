from typing import TypeVar, Generic, Any, Sequence

import math

import numpy as np

import jax.numpy as jnp
import jax
from jax import flatten_util
from jax.typing import ArrayLike

from chex import dataclass
import chex

from flax import nnx

from core.utils import ReplayBuffer, ReplayBufferState, LinearlyInterpolatedTable
from core.utils.func_utils import try_call
from core.utils.batch_utils import flatten_batched_tree

from core.envs.base import Environment, Space
from core.algos.base import Scheduleable, GreedyQActor

from core.envs.wrappers import AutoResetWrapper
from core.envs.utils import rollout, Actor

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")

class TabularQFunc(Generic[TEnvObs], nnx.Module):
    """Get Q values by looking up in a discrete Q table, rounding indices if they fall between gridpoints.
    Implements the `core.algos.base.DiscreteQFunc` protocol."""

    def __init__(self, 
        num_actions: int,
        observation_space: Space[TEnvObs],
        obs_resolution: TEnvObs | None = None,
        q_table_values: jax.Array = None
    ) -> None:
        """`obs_resolution`: PyTree with same treedef/shape as env.observation_space, defaults to 1."""
        self.num_actions = num_actions
        
        self.obs_shapes_dtypes = observation_space.shapes_dtypes

        if obs_resolution is None:
            obs_resolution = jax.tree.map(np.ones_like, observation_space.low)

        self.obs_resolution = jax.tree.map(lambda x: np.asarray(x), obs_resolution)
        chex.assert_trees_all_equal_shapes(self.obs_resolution, observation_space.low)

        self.obs_low_flattened = np.asarray(flatten_util.ravel_pytree(observation_space.low)[0])
        self.obs_high_flattened = np.asarray(flatten_util.ravel_pytree(observation_space.high)[0])
        self.obs_resolution_flattened = np.asarray(flatten_util.ravel_pytree(obs_resolution)[0])

        dim_lens = (self.obs_high_flattened - self.obs_low_flattened) / self.obs_resolution_flattened
        q_table_shape = (num_actions, *(1 + np.round(dim_lens)).astype(int))

        if q_table_values is None:
            q_table_values = jnp.zeros(q_table_shape)

        assert q_table_values.shape == q_table_shape, \
            f"Expected shape {q_table_shape} for `q_table_values`, got {q_table_values.shape}."

        self.q_table_values = nnx.Param(q_table_values)

    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Returns a Q value for every possible action, given `obs`."""
        return jax.vmap(self.get_table_value, in_axes=(None, 0), out_axes=-1)(obs, jnp.arange(self.num_actions))

    def get_table_value(self, obs: TEnvObs, action: ArrayLike) -> ArrayLike:
        """Returns the Q value for the given `obs` and `action`."""
        indices = (action, *self.obs_table_indices(obs))
        return self.q_table_values.value[indices]

    def adjust_table_value(self, obs: TEnvObs, action: ArrayLike, adjust: ArrayLike) -> None:
        """Adjust the Q table's value for the given `obs` and `action` by `adjust`."""
        indices = (action, *self.obs_table_indices(obs))
        self.q_table_values.value = self.q_table_values.value.at[indices].add(adjust)

    def obs_table_indices(self, obs: TEnvObs) -> Sequence[ArrayLike]:
        flattened_obs = flatten_batched_tree(self.obs_shapes_dtypes, obs)

        array_is = jnp.clip(
            jnp.round((flattened_obs - self.obs_low_flattened) / self.obs_resolution_flattened), 
            0, jnp.array(self.q_table_values.value.shape[1:]) - 1
        ).astype(int)

        return tuple(jnp.moveaxis(array_is, -1, 0))

class LinInterpTabularQFunc(Generic[TEnvObs], TabularQFunc[TEnvObs]):
    """Get Q values by linearly interpolating nearby table corners.
    Implements the `core.algos.base.DiscreteQFunc` protocol."""

    def __init__(self, 
        num_actions: int,
        observation_space: Space[TEnvObs],
        obs_resolution: TEnvObs,
        q_table_values: jax.Array = None
    ) -> None:
        """`obs_resolution`: PyTree with same treedef/shape as env.observation_space, defaults to 1."""

        # `LinearlyInterpolatedTable` ensures the space is fully covered by the table corners, 
            # while `TabularQFunc` rounds, potentially rounding down; correct for this
        observation_space = Space(low=observation_space.low,
            high=jax.tree.map(lambda lo, hi, res: np.ceil((hi-lo) / res)*res + lo, 
                observation_space.low, observation_space.high, obs_resolution))

        super().__init__(num_actions, observation_space, obs_resolution, q_table_values)

        self.lin_interp_table = LinearlyInterpolatedTable(
            min=self.obs_low_flattened,
            max=self.obs_high_flattened,
            step=self.obs_resolution_flattened
        )

    def get_table_value(self, obs: TEnvObs, action: ArrayLike) -> ArrayLike:
        """Returns the Q value for the given `obs` and `action`."""
        return self.lin_interp_table.get(self.q_table_values.value[action], self.obs_table_indices(obs))

    def adjust_table_value(self, obs: TEnvObs, action: ArrayLike, adjust: ArrayLike) -> None:
        """Adjust the Q table's value for the given `obs` and `action` by `adjust`."""

        corner_indices, adjust_amounts = self.lin_interp_table.get_corner_adjustments(
            self.q_table_values.value[action], self.obs_table_indices(obs), adjust)

        adjust_indices_tuple = (action[:, None], ) + tuple(jnp.moveaxis(corner_indices, -1, 0))
        self.q_table_values.value = self.q_table_values.value.at[adjust_indices_tuple].add(adjust_amounts)

    def obs_table_indices(self, obs: TEnvObs) -> Sequence[ArrayLike]:
        return flatten_batched_tree(self.obs_shapes_dtypes, obs)

@dataclass(frozen=True)
class Hyperparameters:
    n_envs: int = 32

    discount_rate: Scheduleable[float] = 0.95
    learning_rate: Scheduleable[float] = 0.1

    batch_size: int = 32

    epsilon: Scheduleable[float] = 0.05
        # it is recommended to use a schedule: decay from 1 to ~0.05 over ~10% of training steps
        # eg. optax.schedules.linear_schedule(1, 0.05, 0.1*steps)

    replay_buffer_size: int = 100_000

    train_freq: int = 4 # does around 1 gradient step per train_freq env steps
        # NOTE: if n_envs > train_freq, we take 1 step in each env, followed by multiple gradient steps
        # NOTE: will round up or down if not divisible evenly

    target_update_interval: int = 1000

@dataclass(frozen=True)
class Transition(Generic[TEnvObs]):
    obs: TEnvObs
    action: ArrayLike
    reward: ArrayLike
    next_obs: TEnvObs
    terminated: ArrayLike

@dataclass(frozen=True)
class TrainingState(Generic[TEnvState, TEnvObs]):
    steps: ArrayLike
    env_states: TEnvState
    actor: GreedyQActor

    replay_buffer_state: ReplayBufferState[Transition[TEnvObs]]
    policy_q_func: TabularQFunc
    target_q_func: TabularQFunc

class TabularQLearning(Generic[TEnvState, TEnvObs]):
    """Implementation of Tabular Q-Learning."""

    def __init__(self, 
        env: Environment[TEnvState, TEnvObs, ArrayLike],
        hyperparameters: Hyperparameters = Hyperparameters()
    ) -> None:
        """IMPORTANT: `env` must already be batched; eg. wrap with `VmapWrapper` BEFORE passing in."""

        assert (
            jnp.isscalar(env.action_space.low) 
            and jnp.issubdtype(env.action_space.shapes_dtypes.dtype, jnp.integer)
            and env.action_space.low == 0
        ), "Action space for Q-Learning must be discrete (jnp integer scalar, min=0)."

        self.env = env
        self.num_actions = int(env.action_space.high + 1)

        self.hyperparameters = hyperparameters

        # make replay buffer
        self.transition_shapes_dtypes = Transition(
            obs = self.env.observation_space.shapes_dtypes,
            action = self.env.action_space.shapes_dtypes,
            reward = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
            next_obs = self.env.observation_space.shapes_dtypes,
            terminated = jax.ShapeDtypeStruct(shape=(), dtype=jnp.bool)
        )

        self.replay_buffer = ReplayBuffer[Transition[TEnvObs]](
            self.transition_shapes_dtypes, self.hyperparameters.replay_buffer_size)

    def make_default_q_func(self, init_val: ArrayLike | None = None, rngs=None) -> TabularQFunc:
        q_func = TabularQFunc(self.num_actions, self.env.observation_space)

        if init_val is not None:
            q_func.q_table_values.value = jnp.full_like(q_func.q_table_values.value, init_val)

        return q_func

    def make_actor(self, q_func: TabularQFunc | None = None, epsilon: ArrayLike | None = None, **kwargs) -> GreedyQActor:
        if q_func is None: q_func = self.make_default_q_func(**kwargs)
        if epsilon is None: epsilon = try_call(self.hyperparameters.epsilon, 0)
        return GreedyQActor(q_func, self.num_actions, epsilon=epsilon)

    def rollout_transitions(self,
        rngs: nnx.Rngs, 
        actor: Actor, 
        iter: int,
        initial_env_states: TEnvState | None = None,
    ) -> tuple[Transition[TEnvObs], TEnvState]:
        """Collect a rollout of `Transition`s.

        Runs `n_envs` environments in parallel for `iter` steps each,
            for a total of `iter * n_envs` transitions.

        Initializes initial environment states if none given.

        Returns: transitions, final environment states
        """

        (unreset_obs, timesteps), env_states, final_infos = rollout(
            rngs, self.env, actor,
            iter, self.hyperparameters.n_envs,
            initial_env_states,

            take_func = lambda timesteps, rngs: (
                self.env.get_obs(
                    jax.random.split(rngs.env(), self.hyperparameters.n_envs), 
                    timesteps.info[AutoResetWrapper.UNRESET_STATE_INFO_KEY]
                ), 
                timesteps.replace(state=None, info=None) # remove unnecessary fields to save memory
            )
        )

        final_obs = self.env.get_obs(
            jax.random.split(rngs.env(), self.hyperparameters.n_envs), 
            final_infos[AutoResetWrapper.UNRESET_STATE_INFO_KEY]
        )
        
        next_obs = jax.tree.map(lambda middles, finals: 
                jnp.concatenate((middles[1:], finals[None, ...]), axis=0),
            unreset_obs, final_obs)

        timesteps = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), timesteps) # flatten to remove axis 0
        next_obs = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), next_obs) # flatten to remove axis 0

        transitions = Transition(obs=timesteps.obs, action=timesteps.action, 
            reward=timesteps.reward, next_obs=next_obs, terminated=timesteps.terminated)

        return transitions, env_states

    def init_training_state(self,
        rngs: nnx.Rngs,
        policy_q_func: TabularQFunc | None = None,
        target_q_func: TabularQFunc | None = None,
        replay_buffer_state: ReplayBufferState[Transition[TEnvObs]] | None = None,
        prefill_steps: int = 10_000
    ) -> TrainingState[TEnvState, TEnvObs]:
        if policy_q_func is None: policy_q_func = self.make_default_q_func()
        if target_q_func is None: target_q_func = nnx.clone(policy_q_func)

        actor = self.make_actor(policy_q_func)

        if replay_buffer_state is None:
            replay_buffer_state = self.replay_buffer.init()

        # prefill replay buffer
        transitions, env_states = nnx.jit(self.rollout_transitions, static_argnames=('iter', 'actor'))(rngs,
            lambda obs, rngs: (actor.random_action(rngs, (self.hyperparameters.n_envs,)), {}),
            math.ceil(prefill_steps / self.hyperparameters.n_envs),
        )

        replay_buffer_state = self.replay_buffer.insert(replay_buffer_state, transitions)

        return TrainingState(
            steps = jnp.array(0, dtype=jnp.int32),
            env_states = env_states,    
            actor = actor,

            replay_buffer_state = replay_buffer_state,
            policy_q_func = policy_q_func,
            target_q_func = target_q_func
        )

    def train(self, rngs: nnx.Rngs, training_state: TrainingState[TEnvState, TEnvObs], steps: int) \
            -> tuple[TrainingState[TEnvState, TEnvObs], dict[Any, Any]]:
        """Train from the given `training_state`, returning an updated `training_state` and metrics."""
        steps_per_env_per_iter = math.ceil(self.hyperparameters.train_freq / self.hyperparameters.n_envs)
        total_steps_per_iter = steps_per_env_per_iter * self.hyperparameters.n_envs
        learn_steps_per_iter = math.ceil(self.hyperparameters.n_envs / self.hyperparameters.train_freq)

        def train_iteration(training_state: TrainingState[TEnvState, TEnvObs], rngs: nnx.Rngs) \
                -> tuple[TrainingState[TEnvState, TEnvObs], dict[Any, Any]]:
            env_states = training_state.env_states
            steps = training_state.steps
            actor = training_state.actor

            replay_buffer_state = training_state.replay_buffer_state
            policy_q_func = training_state.policy_q_func
            target_q_func = training_state.target_q_func
            
            ## sample transitions from environment ##
            actor.epsilon.value = try_call(self.hyperparameters.epsilon, steps)

            transitions, env_states = self.rollout_transitions(rngs, 
                actor, steps_per_env_per_iter, env_states)

            replay_buffer_state = self.replay_buffer.insert(replay_buffer_state, transitions)

            steps += total_steps_per_iter

            ## update q functions ##
            def learn_step(policy_q_func: TabularQFunc, rngs: nnx.Rngs) \
                    -> tuple[TabularQFunc, dict[Any, Any]]:

                sampled_transitions = self.replay_buffer.sample(rngs.transitions(), 
                    replay_buffer_state, self.hyperparameters.batch_size)

                next_qs = target_q_func(sampled_transitions.next_obs)
                max_next_qs = jnp.max(next_qs, axis=-1)
                # zero out q_val if terminated
                max_next_qs = max_next_qs * jnp.logical_not(sampled_transitions.terminated)

                target_qs = sampled_transitions.reward \
                    + try_call(self.hyperparameters.discount_rate, steps)*max_next_qs
                pred_qs = policy_q_func.get_table_value(sampled_transitions.obs, sampled_transitions.action)

                # only used as a metric
                loss = jnp.mean(jnp.power(target_qs - pred_qs, 2))

                adjusts = try_call(self.hyperparameters.learning_rate, steps) * (target_qs - pred_qs) \
                    / self.hyperparameters.batch_size # make "learning rate" independent of batch size
                policy_q_func.adjust_table_value(sampled_transitions.obs, sampled_transitions.action, adjusts)

                return policy_q_func, { 'q_loss': loss }

            policy_q_func, metrics = nnx.scan(learn_step)(policy_q_func, rngs.fork(split=learn_steps_per_iter))

            # update target if enough steps have passed
            update_target = steps % self.hyperparameters.target_update_interval < total_steps_per_iter
            nnx.update(target_q_func, nnx.cond(update_target, 
                lambda policy_q_func, target_q_func: nnx.state(policy_q_func), 
                lambda policy_q_func, target_q_func: nnx.state(target_q_func),
                policy_q_func, target_q_func
            ))

            return TrainingState(
                steps=steps,
                env_states=env_states,
                actor=actor,

                replay_buffer_state=replay_buffer_state,
                policy_q_func=policy_q_func,
                target_q_func=target_q_func,
            ), jax.tree.map(lambda x: jnp.mean(x), metrics)

        iterations = math.ceil(steps / total_steps_per_iter)
        training_state, metrics = nnx.scan(train_iteration)(training_state, rngs.fork(split=iterations))

        # set into eval mode for the user
        training_state.actor.epsilon.value = jnp.array(0, dtype=jnp.float32)

        return training_state, jax.tree.map(lambda x: jnp.mean(x), metrics)