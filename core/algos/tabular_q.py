import math

from flax import nnx

import jax.numpy as jnp
import jax
from jax import flatten_util

from jax.typing import ArrayLike
from chex import dataclass
import chex
from typing import TypeVar, Generic, Any

import functools

from core.algos.base import Scheduleable, resolve_scheduleable

from core.envs.base import Environment, Space
from core.envs.wrappers import VmapAutoResetWrapper, VmapWrapper, AutoResetWrapper
from core.utils import ReplayBuffer, ReplayBufferState

@dataclass(frozen=True)
class Hyperparameters:
    n_envs: int = 32

    discount_rate: Scheduleable[float] = 0.95
    learning_rate: Scheduleable[float] = 0.1

    batch_size: int = 32

    epsilon: Scheduleable[float] = 0.1
        # it is recommended to use a schedule: decay from 1 to ~0.05 over ~10% of training steps
        # eg. optax.schedules.linear_schedule(1, 0.05, 0.1*steps)

    replay_buffer_size: int = 1000

    train_freq: int = 4 # does around 1 gradient step per train_freq env steps
        # NOTE: if n_envs > train_freq, we take 1 step in each env, followed by multiple gradient steps
        # NOTE: will round up or down if not divisible evenly

    target_update_interval: int = 1000

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")

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
    replay_buffer_state: ReplayBufferState[Transition[TEnvObs]]

    policy: jax.Array
        # perhaps named confusingly; policy q-values, used for getting actions, though not directly an actor
        # named like this so api matches with other algos
    target: jax.Array # target q-values

class TabularQ(Generic[TEnvState, TEnvObs]):
    """Implementation of Tabular Q-Learning."""

    def __init__(self, 
        env: Environment[TEnvState, TEnvObs, ArrayLike],
        obs_resolution: ArrayLike = None, # pytree with same shape as env.observation_space, defaults to 1
        observation_space: Space[TEnvObs] = None, # override environment observation space
        hyperparameters: Hyperparameters = Hyperparameters()
    ) -> None:
        assert (
            jnp.isscalar(env.action_space.low) 
            and jnp.issubdtype(env.action_space.shapes_dtypes.dtype, jnp.integer)
            and env.action_space.low == 0
        ), "Action space for Q-Learning must be discrete (jnp integer scalar, min=0)."

        self.env = env
        self.num_actions = int(env.action_space.high + 1)

        self.hyperparameters = hyperparameters

        self.observation_space = observation_space if observation_space is not None else env.observation_space

        if obs_resolution is None:
            obs_resolution = jax.tree.map(jnp.ones_like, self.observation_space.low)

        self.obs_resolution = obs_resolution

        self.obs_low_flattened, self.obs_unflatten_func = flatten_util.ravel_pytree(self.observation_space.low)
        self.obs_high_flattened = flatten_util.ravel_pytree(self.observation_space.high)[0]
        self.obs_resolution_flattened = flatten_util.ravel_pytree(obs_resolution)[0]

        self.q_table_shape = (1 + jnp.round(
            (self.obs_high_flattened - self.obs_low_flattened) / self.obs_resolution_flattened)) \
            .astype(int).tolist()

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

    def get_q_table_index(self, obs: TEnvObs) -> jax.Array:
        flattened_obs, _ = flatten_util.ravel_pytree(obs)

        return jnp.clip(
            jnp.round((flattened_obs - self.obs_low_flattened) / self.obs_resolution_flattened), 
            0, jnp.array(self.q_table_shape) - 1
        ).astype(int)

    def get_greedy_action(self, rngs: nnx.Rngs, policy: jax.Array, obs: TEnvObs) -> ArrayLike:
        q_vals = jax.vmap(lambda qs, obs: qs[tuple(self.get_q_table_index(obs))], 
            in_axes=[0, None])(policy, obs)
        return jnp.argmax(q_vals)

    def get_random_action(self, rngs: nnx.Rngs) -> ArrayLike:
        return jax.random.randint(rngs.actions(), shape=(), minval=0, maxval=self.num_actions)

    def get_action(self, rngs: nnx.Rngs, policy: jax.Array, obs: TEnvObs, epsilon: ArrayLike = 0) -> ArrayLike:
        random_action = self.get_random_action(rngs)
        greedy_action = self.get_greedy_action(rngs, policy, obs)

        return jnp.where(jax.random.uniform(rngs.actions()) < epsilon, 
            random_action, greedy_action)

    def create_default_policy(self, rngs: nnx.Rngs, init_val: ArrayLike = jnp.array(0, dtype=jnp.float32)) -> jax.Array:
        return jnp.repeat(jnp.full(self.q_table_shape, init_val, dtype=jnp.float32)[None, ...], 
            self.num_actions, axis=0)

    def rollout(self,
        rngs: nnx.Rngs, 
        policy: jax.Array, 
        iter: int,
        initial_env_states: TEnvState | None = None,
        epsilon: ArrayLike = 0
    ) -> Transition[TEnvObs]:
        """Collect a rollout of `Transition`s.

        Runs `n_envs` environments in parallel for `iter` iterations,
            for a total of `iter * n_envs` transitions.

        Initializes initial environment states if none given.

        Returns: transitions, final environment states
        """

        #env = VmapWrapper(AutoResetWrapper(self.env))
        env = VmapAutoResetWrapper(self.env)

        def batched_env_step(states: TEnvState, rngs: nnx.Rngs) -> tuple[TEnvState, Transition[TEnvObs]]:
            obs = env.get_obs(rngs.env(), states)

            actions = nnx.vmap(lambda rngs, obs: self.get_action(rngs, policy, obs, epsilon))(
                rngs.fork(split=self.hyperparameters.n_envs), obs)

            new_states, rewards, terminated, truncated, infos = env.step(rngs.env(), states, actions)
            next_obs = env.get_obs(rngs.env(), infos.pop(env.NEXT_STATE_INFO_KEY))

            return (
                new_states,
                Transition(obs=obs, action=actions, reward=rewards, next_obs=next_obs,terminated=terminated)
            )

        if initial_env_states is None:
            initial_env_states, info = env.reset(rngs.env(), num=self.hyperparameters.n_envs)

        env_states, transitions = nnx.scan(batched_env_step)(initial_env_states, rngs.fork(split=iter))
        transitions = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), transitions) # flatten to remove axis 0

        return transitions, env_states

    def init_training_state(self,
        rngs: nnx.Rngs,
        policy: jax.Array | None = None,
        replay_buffer_state: ReplayBufferState[Transition[TEnvObs]] | None = None,
        prefill_steps: int = 10_000
    ) -> TrainingState[TEnvState, TEnvObs]:
        if policy is None:
            policy = self.create_default_policy(rngs)

        if replay_buffer_state is None:
            replay_buffer_state = self.replay_buffer.init()

        # prefill replay buffer
        transitions, env_states = nnx.jit(self.rollout, static_argnames=('iter'))(rngs,
            policy,
            math.ceil(prefill_steps / self.hyperparameters.n_envs),
            epsilon=1
        )

        replay_buffer_state = self.replay_buffer.insert(replay_buffer_state, transitions)

        return TrainingState(
            steps = jnp.array(0, dtype=jnp.int32),
            env_states = env_states,    
            replay_buffer_state = replay_buffer_state,

            policy = policy,
            target = policy
        )

    @functools.partial(nnx.jit, static_argnames=('self', 'epoch_steps'))
    def train_epoch(self, 
        rngs: nnx.Rngs,
        training_state: TrainingState[TEnvState, TEnvObs],
        epoch_steps: int,
    ) -> tuple[TrainingState[TEnvState, TEnvObs], dict[Any, Any]]:
        """Train for one 'epoch' -- one fully JIT compiled segment."""

        steps_per_env_per_iter = math.ceil(self.hyperparameters.train_freq / self.hyperparameters.n_envs)
        total_steps_per_iter = steps_per_env_per_iter * self.hyperparameters.n_envs
        learn_steps_per_iter = math.ceil(self.hyperparameters.n_envs / self.hyperparameters.train_freq)

        def train_iteration(training_state: TrainingState[TEnvState, TEnvObs], rngs: nnx.Rngs) \
                -> tuple[TrainingState[TEnvState, TEnvObs], dict[Any, Any]]:
            env_states = training_state.env_states
            steps = training_state.steps
            replay_buffer_state = training_state.replay_buffer_state

            policy = training_state.policy
            target = training_state.target
            
            ## sample transitions from environment ##

            transitions, env_states = self.rollout(rngs, 
                policy,
                steps_per_env_per_iter,
                env_states,
                epsilon=resolve_scheduleable(self.hyperparameters.epsilon, steps)
            )

            replay_buffer_state = self.replay_buffer.insert(replay_buffer_state, transitions)

            steps += total_steps_per_iter

            ## update policy ##

            def learn_step(policy: jax.Array, rngs: nnx.Rngs) \
                    -> tuple[tuple[nnx.Module, nnx.Optimizer], dict[Any, Any]]:

                sampled_transitions = self.replay_buffer.sample(rngs.transitions(), 
                    replay_buffer_state, self.hyperparameters.batch_size)

                q_is = jax.vmap(self.get_q_table_index)(sampled_transitions.next_obs)
                next_qs = target[(slice(None),) + tuple(q_is.T)]
                max_next_qs = jnp.max(next_qs, axis=0)
                # zero out q_val if terminated
                max_next_qs = max_next_qs * jnp.logical_not(sampled_transitions.terminated)

                target_qs = sampled_transitions.reward \
                    + resolve_scheduleable(self.hyperparameters.discount_rate, steps)*max_next_qs

                adjust_is = (sampled_transitions.action, ) \
                    + tuple(jax.vmap(self.get_q_table_index)(sampled_transitions.obs).T)
                pred_qs = policy[adjust_is]

                # only used as a metric
                loss = jnp.mean(jnp.power(target_qs - pred_qs, 2))

                adjusts = resolve_scheduleable(self.hyperparameters.learning_rate, steps) * (target_qs - pred_qs) \
                    / self.hyperparameters.batch_size # make "learning rate" independent of batch size
                policy = policy.at[adjust_is].add(adjusts)

                return policy, { 'critic_loss': loss }

            policy, metrics = nnx.scan(learn_step)(policy, rngs.fork(split=learn_steps_per_iter))

            # update target if enough steps have passed
            update_target = steps % self.hyperparameters.target_update_interval < total_steps_per_iter
            target = jax.lax.cond(update_target, lambda: policy, lambda: target)

            return TrainingState(
                steps=steps,
                env_states=env_states,
                replay_buffer_state=replay_buffer_state,
                policy=policy,
                target=target,
            ), jax.tree.map(lambda x: jnp.mean(x), metrics)

        iterations = math.ceil(epoch_steps / total_steps_per_iter)
        training_state, metrics = nnx.scan(train_iteration)(training_state, rngs.fork(split=iterations))

        return training_state, jax.tree.map(lambda x: jnp.mean(x), metrics)