import math

import jax.numpy as jnp
import jax
from jax import flatten_util

from jax.typing import ArrayLike
from chex import dataclass
from typing import TypeVar, Generic

import functools

from core.envs.base import Environment, Space
from core.utils import ReplayBuffer, ReplayBufferState

@dataclass(frozen=True)
class TabularQHyperparameters:
    n_envs: int = 32

    discount_rate: float = 0.95
    learning_rate: float = 0.01

    batch_size: int = 32
        # NOTE: we don't average the update (divide by batch size), so higher batch size -> higher learning rate

    #epsilon: float = 0.05

    epsilon_initial: float = 1
    epsilon_final: float = 0.05
    epsilon_anneal_fraction: float = 0.1

    replay_buffer_size: int = 1000

    train_freq: int = 1
    target_update_interval: int = 1000

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")

class TabularQ(Generic[TEnvState, TEnvObs]):
    """Implementation of Tabular Q-Learning."""

    @dataclass(frozen=True)
    class Transition:
        cur_obs: TEnvObs
        action: ArrayLike
        reward: ArrayLike
        new_obs: TEnvObs
        terminated: ArrayLike

    @dataclass(frozen=True)
    class TrainingState:
        steps: ArrayLike
        env_states: TEnvState
        policy_q_vals: jax.Array
        target_q_vals: jax.Array
        epsilon: ArrayLike

    def __init__(self, 
        env: Environment[TEnvState, TEnvObs, ArrayLike],
        obs_resolution: ArrayLike = None, # pytree with same shape as env.observation_space, defaults to 1
        observation_space: Space[TEnvObs] = None, # override environment observation space
        hyperparameters: TabularQHyperparameters = TabularQHyperparameters()
    ) -> None:
        assert (
            jnp.isscalar(env.action_space.low) 
            and jnp.issubdtype(env.action_space.shapes_dtypes.dtype, jnp.integer)
            and env.action_space.low == 0
        ), "Action space for Q-Learning must be discrete (jnp integer scalar, min=0)."

        self.env = env
        self.num_actions = env.action_space.high + 1

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
        self.transition_shapes_dtypes = TabularQ.Transition(
            cur_obs = self.env.observation_space.shapes_dtypes,
            action = self.env.action_space.shapes_dtypes,
            reward = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
            new_obs = self.env.observation_space.shapes_dtypes,
            terminated = jax.ShapeDtypeStruct(shape=(), dtype=jnp.bool)
        )

        self.replay_buffer = ReplayBuffer[TabularQ.Transition](
            self.transition_shapes_dtypes, self.hyperparameters.replay_buffer_size)

    def get_q_table_index(self, obs: TEnvObs) -> jax.Array:
        flattened_obs = flatten_util.ravel_pytree(obs)[0]

        return jnp.clip(
            jnp.round((flattened_obs - self.obs_low_flattened) / self.obs_resolution_flattened), 
            0, jnp.array(self.q_table_shape) - 1
        ).astype(int)

    def get_greedy_action(self, q_table_vals: jax.Array, obs: TEnvObs) -> ArrayLike:
        q_vals = jax.vmap(lambda qs, obs: qs[tuple(self.get_q_table_index(obs))], 
            in_axes=[0, None])(q_table_vals, obs)
        return jnp.argmax(q_vals)

    def get_action(self, key: jax.Array, q_table_vals: jax.Array, 
        epsilon: ArrayLike, obs: TEnvObs) -> ArrayLike:

        do_greedy_key, random_action_key = jax.random.split(key)

        random_action = jax.random.randint(random_action_key, shape=(), minval=0, maxval=self.num_actions)
        greedy_action = self.get_greedy_action(q_table_vals, obs)

        return jnp.where(jax.random.uniform(do_greedy_key) < epsilon, 
            random_action, greedy_action)

    def init_q_table_vals(self, init_val: ArrayLike = jnp.array(0, dtype=jnp.float32)) -> jax.Array:
        return jnp.repeat(jnp.full(self.q_table_shape, init_val, dtype=jnp.float32)[None, ...], 
            self.num_actions, axis=0)

    def train(self,
        key: jax.Array,
        steps: int,

        init_q_vals: ArrayLike | None = None,
        log_interval_steps: int = 100_000,
    ) -> jax.Array:
        """Train the q-table. Returns q-table with updated values."""

        ## initialize ##
        replay_buffer_state = self.replay_buffer.init()

        key, reset_key = jax.random.split(key, 2)
        reset_keys = jax.random.split(reset_key, self.hyperparameters.n_envs)

        env_states, info = jax.vmap(self.env.reset)(reset_keys)

        if init_q_vals == None:
            init_q_vals = self.init_q_table_vals()

        training_state = self.TrainingState(
            steps = jnp.array(0, dtype=jnp.int32),
            env_states = env_states,
            policy_q_vals = init_q_vals,
            target_q_vals = init_q_vals,
            epsilon = jnp.array(self.hyperparameters.epsilon_initial, dtype=jnp.float32)
        )

        epsilon_anneal_amount = self.hyperparameters.epsilon_initial - self.hyperparameters.epsilon_final
        epsilon_anneal_steps = steps * self.hyperparameters.epsilon_anneal_fraction
        epsilon_anneal_rate = epsilon_anneal_amount / epsilon_anneal_steps

        while training_state.steps < steps:
            key, train_key = jax.random.split(key, 2)

            training_state, replay_buffer_state = self.train_epoch(train_key, 
                log_interval_steps, epsilon_anneal_rate, training_state, replay_buffer_state)

            print(f"Completed steps={training_state.steps}")

        return training_state.policy_q_vals

    @functools.partial(jax.jit, static_argnames=('self', 'steps'))
    def train_epoch(
        self, key: jax.Array, steps: int, epsilon_anneal_rate: float,
        training_state: TrainingState, replay_buffer_state: ReplayBufferState
    ) -> tuple[TrainingState, ReplayBufferState]:
        """Train for one 'epoch' -- one fully JIT compiled segment."""

        steps_per_update = self.hyperparameters.train_freq * self.hyperparameters.n_envs

        def train_iteration(carry: tuple[jax.Array, TabularQ.TrainingState, ReplayBufferState], _):
            key, training_state, replay_buffer_state = carry

            env_states = training_state.env_states
            steps = training_state.steps
            policy_q_vals = training_state.policy_q_vals
            target_q_vals = training_state.policy_q_vals
            epsilon = training_state.epsilon
            
            ## sample transitions from environment ##

            key, step_key = jax.random.split(key, 2)
            step_keys = jax.random.split(step_key, self.hyperparameters.n_envs)

            def env_step(carry: tuple[jax.Array, TEnvState], _):
                key, env_state = carry

                key, action_key, step_key, reset_key, cur_obs_key, new_obs_key = jax.random.split(key, 6)

                cur_obs = self.env.get_obs(cur_obs_key, env_state)
                action = self.get_action(action_key, policy_q_vals, epsilon, cur_obs)
                new_state, reward, terminated, truncated, info = self.env.step(step_key, env_state, action)
                new_obs = self.env.get_obs(new_obs_key, new_state)

                # reset env if terminated/truncated, don't otherwise
                next_state = jax.lax.cond(jnp.logical_or(terminated, truncated), 
                    lambda: self.env.reset(reset_key)[0], lambda: new_state)

                return (
                    (key, next_state),
                    self.Transition(cur_obs=cur_obs, action=action, reward=reward, new_obs=new_obs, terminated=terminated)
                )

            carry, transitions = jax.lax.scan(jax.vmap(env_step), 
                (step_keys, env_states), None, length=self.hyperparameters.train_freq)
            _, env_states = carry

            steps += steps_per_update

            epsilon = jnp.maximum(epsilon - epsilon_anneal_rate*steps_per_update, self.hyperparameters.epsilon_final)

            transitions: TabularQ.Transition = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), 
                transitions) # flatten to remove axis 0
                
            replay_buffer_state = self.replay_buffer.insert(replay_buffer_state, transitions)

            ## update policy q-table ##

            key, sample_key = jax.random.split(key, 2)
            sampled_transitions = self.replay_buffer.sample(sample_key, 
                replay_buffer_state, self.hyperparameters.batch_size)

            def get_q_adjustments(transition: TabularQ.Transition):
                next_q_vals = jax.vmap(lambda qs, obs: qs[tuple(self.get_q_table_index(obs))], 
                    in_axes=[0, None])(target_q_vals, transition.new_obs)
                
                # zero out q_val if terminated
                next_q_vals = next_q_vals * jnp.logical_not(transition.terminated)

                new_q = transition.reward + self.hyperparameters.discount_rate*jnp.max(next_q_vals)

                adjust_i = self.get_q_table_index(transition.cur_obs)
                old_q = policy_q_vals[transition.action][tuple(adjust_i)]
                adjust = self.hyperparameters.learning_rate * (new_q - old_q)

                return transition.action, adjust_i, adjust

            actions, adjust_is, adjusts = jax.vmap(get_q_adjustments)(sampled_transitions)

            policy_q_vals = policy_q_vals.at[(actions, ) + tuple(adjust_is.T)].add(adjusts)

            # update target_q_vals if enough steps have passed
            update_target_qs = steps % self.hyperparameters.target_update_interval < steps_per_update
            #target_q_vals = jnp.where(update_target_qs, policy_q_vals, target_q_vals)
            target_q_vals = jax.lax.cond(update_target_qs, lambda: policy_q_vals, lambda: target_q_vals)

            return (key, TabularQ.TrainingState(
                steps=steps,
                env_states=env_states,
                policy_q_vals=policy_q_vals,
                target_q_vals=target_q_vals,
                epsilon=epsilon
            ), replay_buffer_state), None

        iterations = math.ceil(steps / (self.hyperparameters.train_freq * self.hyperparameters.n_envs))
        carry, _ = jax.lax.scan(train_iteration, (key, training_state, replay_buffer_state), length=iterations)
        key, training_state, replay_buffer_state = carry

        return training_state, replay_buffer_state