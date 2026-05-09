import math

import jax.numpy as jnp
import jax

from jax.typing import ArrayLike
from chex import dataclass
from typing import TypeVar, Generic

import functools

from core.algos.base import Hyperparameters

from core.envs.base import Environment
from core.utils import ReplayBuffer, ReplayBufferState, LinearlyInterpolatedTable

@dataclass(frozen=True)
class TabularQHyperparameters(Hyperparameters):
    discount_rate: float = 0.95
    learning_rate: float = 0.01

    batch_size: int = 32
        # NOTE: we don't average the update (divide by batch size), so higher batch size -> higher learning rate

    epsilon: float = 0.05

    replay_buffer_size: int = 1000

    train_freq: int = 1
    target_update_interval: int = 1000

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")

class LinearlyInterpolatedTabularQ(Generic[TEnvState, TEnvObs]):
    """Implementation of Tabular Q-Learning, with intermediate q-values calculated through linear interpolation."""

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
        prev_target_qs_update_steps: ArrayLike
        env_states: TEnvState
        policy_q_vals: jax.Array
        target_q_vals: jax.Array

    def __init__(self, 
        env: Environment[TEnvState, TEnvObs, ArrayLike],
        q_table: LinearlyInterpolatedTable,
        hyperparameters: TabularQHyperparameters = TabularQHyperparameters()
    ) -> None:
        assert (
            jnp.isscalar(env.action_space.low) 
            and jnp.issubdtype(env.action_space.leaf_dtypes[0], jnp.integer)
            and env.action_space.low == 0
        ), "Action space for Q-Learning must be discrete (jnp integer scalar, min=0)."

        self.env = env
        self.num_actions = env.action_space.high + 1

        self.q_table = q_table
        self.hyperparameters = hyperparameters

        # make replay buffer
        key = jax.random.key(0) # any random key will work since values in dummy data do not matter
        key, reset_key, action_key, obs_key = jax.random.split(key, 4)

        dummy_state, _ = self.env.reset(reset_key)
        dummy_obs = self.env.get_obs(obs_key, dummy_state)

        self.dummy_transition = LinearlyInterpolatedTabularQ.Transition(
            cur_obs = dummy_obs,
            action = self.env.action_space.sample(action_key),
            reward = jnp.array(0),
            new_obs = dummy_obs,
            terminated = False
        )

        self.replay_buffer = ReplayBuffer(self.dummy_transition, self.hyperparameters.replay_buffer_size)

    def get_greedy_action(self, q_table_vals: jax.Array, obs: TEnvObs) -> ArrayLike:
        q_vals = jax.vmap(self.q_table.get, in_axes=[0, None])(q_table_vals, obs)
        return jnp.argmax(q_vals)

    def get_action(self, key: jax.Array, q_table_vals: jax.Array, obs: TEnvObs) -> ArrayLike:
        do_greedy_key, random_action_key = jax.random.split(key)

        random_action = jax.random.randint(random_action_key, shape=(), minval=0, maxval=self.num_actions)
        greedy_action = self.get_greedy_action(q_table_vals, obs)

        return jnp.where(jax.random.uniform(do_greedy_key) < self.hyperparameters.epsilon, 
            random_action, greedy_action)

    def init_q_table_vals(self, init_val: ArrayLike = jnp.array(0)) -> jax.Array:
        return jnp.repeat(self.q_table.init(init_val)[None, ...], self.num_actions, axis=0)

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
            steps = jnp.array(0),
            prev_target_qs_update_steps = jnp.array(0),
            env_states = env_states,
            policy_q_vals = init_q_vals,
            target_q_vals = init_q_vals,
        )

        while training_state.steps < steps:
            key, train_key = jax.random.split(key, 2)

            training_state, replay_buffer_state = self.train_epoch(
                train_key, log_interval_steps, training_state, replay_buffer_state)

            print(f"Completed steps={training_state.steps}")

        return training_state.policy_q_vals

    @functools.partial(jax.jit, static_argnames=('self', 'steps'))
    def train_epoch(
        self, key: jax.Array, steps: int, 
        training_state: TrainingState, replay_buffer_state: ReplayBufferState
    ) -> tuple[TrainingState, ReplayBufferState]:
        """Train for one 'epoch' -- one fully JIT compiled segment."""

        def train_iteration(carry: tuple[jax.Array, LinearlyInterpolatedTabularQ.TrainingState, ReplayBufferState], _):
            key, training_state, replay_buffer_state = carry

            env_states = training_state.env_states
            steps = training_state.steps
            prev_target_qs_update_steps = training_state.prev_target_qs_update_steps
            policy_q_vals = training_state.policy_q_vals
            target_q_vals = training_state.policy_q_vals
            
            ## sample transitions from environment ##

            key, step_key = jax.random.split(key, 2)
            step_keys = jax.random.split(step_key, self.hyperparameters.n_envs)

            def env_step(carry: tuple[jax.Array, TEnvState], _):
                key, env_state = carry

                key, action_key, step_key, reset_key, cur_obs_key, new_obs_key = jax.random.split(key, 6)

                cur_obs = self.env.get_obs(cur_obs_key, env_state)
                action = self.get_action(action_key, policy_q_vals, cur_obs)
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

            steps += self.hyperparameters.train_freq * self.hyperparameters.n_envs

            transitions = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), transitions)
                # flatten to remove axis 0
            replay_buffer_state = self.replay_buffer.insert(replay_buffer_state, transitions)

            ## update policy q-table ##

            key, sample_key = jax.random.split(key, 2)
            sampled_transitions = self.replay_buffer.sample(sample_key, 
                replay_buffer_state, self.hyperparameters.batch_size)

            def update_q_get_corner_adjustments(transition: LinearlyInterpolatedTabularQ.Transition):
                next_q_vals = jax.vmap(self.q_table.get, in_axes=[0, None])(target_q_vals, transition.new_obs)
                
                # zero out q_val if terminated
                next_q_vals = next_q_vals * jnp.logical_not(transition.terminated)

                new_q = transition.reward + self.hyperparameters.discount_rate*jnp.max(next_q_vals)

                old_q = self.q_table.get(policy_q_vals[transition.action], transition.cur_obs)
                adjust = self.hyperparameters.learning_rate * (new_q - old_q)

                adjust_is, adjusts = self.q_table.adjust_get_corner_adjustments(
                    policy_q_vals[transition.action], transition.cur_obs, adjust)

                return transition.action, adjust_is, adjusts

            actions, adjust_is, adjusts = jax.vmap(update_q_get_corner_adjustments)(sampled_transitions)

            # flatten to remove axis 0; duplicate values in actions to match
            actions = jnp.repeat(actions, adjust_is.shape[1])
            adjust_is = adjust_is.reshape(-1, *adjust_is.shape[2:])
            adjusts = adjusts.reshape(-1, *adjusts.shape[2:])

            policy_q_vals = policy_q_vals.at[(actions, ) + tuple(adjust_is.T)].add(adjusts)

            # update target_q_vals if enough steps have passed
            update_target_qs = steps - prev_target_qs_update_steps >= self.hyperparameters.target_update_interval
            target_q_vals = jnp.where(update_target_qs, policy_q_vals, target_q_vals)
            prev_target_qs_update_steps = jnp.where(update_target_qs, steps, prev_target_qs_update_steps)

            return (key, LinearlyInterpolatedTabularQ.TrainingState(
                steps=steps,
                prev_target_qs_update_steps=prev_target_qs_update_steps,
                env_states=env_states,
                policy_q_vals=policy_q_vals,
                target_q_vals=target_q_vals,
            ), replay_buffer_state), None

        iterations = math.ceil(steps / (self.hyperparameters.train_freq * self.hyperparameters.n_envs))
        carry, _ = jax.lax.scan(train_iteration, (key, training_state, replay_buffer_state), length=iterations)
        key, training_state, replay_buffer_state = carry

        return training_state, replay_buffer_state