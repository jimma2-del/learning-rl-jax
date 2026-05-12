import math

import jax.numpy as jnp
import jax

from jax.typing import ArrayLike
from chex import dataclass
from typing import TypeVar, Generic

import functools

from flax import nnx
import optax

from core.algos.base import Hyperparameters

from core.envs.base import Environment
from core.utils import ReplayBuffer, ReplayBufferState

from core.sample_networks import MLP, MLPFeatureExtractor

@dataclass(frozen=True)
class DQNHyperparameters(Hyperparameters):
    epsilon_initial: float = 1
    epsilon_final: float = 0.05
    epsilon_anneal_fraction: float = 0.1

    replay_buffer_size: int = 1_000_000

    train_freq: int = 4 # does an average of 1 gradient step per train_freq env steps
        # NOTE: if n_envs > train_freq, we take 1 step in each env, followed by multiple gradient steps
    
    # TODO: train_freq is not implemented properly yet; need to add multiple gradient steps

    target_update_interval: int = 1000

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")

class DQN(Generic[TEnvState, TEnvObs]):
    """Implementation of DQN."""

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
        epsilon: ArrayLike

        policy_q_net: nnx.Module
        target_q_net: nnx.Module
        optimizer: nnx.Optimizer

    def __init__(self, 
        env: Environment[TEnvState, TEnvObs, ArrayLike],
        hyperparameters: DQNHyperparameters = DQNHyperparameters()
    ) -> None:
        assert (
            jnp.isscalar(env.action_space.low) 
            and jnp.issubdtype(env.action_space.shapes_dtypes.dtype, jnp.integer)
            and env.action_space.low == 0
        ), "Action space for Q-Learning must be discrete (jnp integer scalar, min=0)."

        self.env = env
        self.num_actions = env.action_space.high + 1

        self.hyperparameters = hyperparameters

        # make replay buffer
        self.transition_shapes_dtypes = DQN.Transition(
            cur_obs = self.env.observation_space.shapes_dtypes,
            action = self.env.action_space.shapes_dtypes,
            reward = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
            new_obs = self.env.observation_space.shapes_dtypes,
            terminated = jax.ShapeDtypeStruct(shape=(), dtype=jnp.bool)
        )

        self.replay_buffer = ReplayBuffer[DQN.Transition](
            self.transition_shapes_dtypes, self.hyperparameters.replay_buffer_size)

    def get_greedy_action(self, rngs: nnx.Rngs, q_net: nnx.module, obs: TEnvObs) -> ArrayLike:
        q_vals = q_net(obs, rngs=rngs)
        return jnp.argmax(q_vals)

    def get_action(self, rngs: nnx.Rngs, q_net: nnx.module, 
        epsilon: ArrayLike, obs: TEnvObs) -> ArrayLike:

        random_action = jax.random.randint(rngs.actions(), shape=(), minval=0, maxval=self.num_actions)
        greedy_action = self.get_greedy_action(rngs, q_net, obs)

        return jnp.where(jax.random.uniform(rngs.actions()) < epsilon, 
            random_action, greedy_action)

    def create_default_q_net(self, rngs: nnx.Rngs) -> nnx.Module:
        FEATURE_EXTRACTOR_OUTPUT_DIM = 256

        return nnx.Sequential(
            MLPFeatureExtractor[TEnvObs](rngs, self.env.observation_space.shapes_dtypes, 
                output_dim=FEATURE_EXTRACTOR_OUTPUT_DIM),
            MLP(rngs, input_dim=FEATURE_EXTRACTOR_OUTPUT_DIM, output_dim=self.num_actions)
        )

    def train(self,
        rngs: nnx.Rngs,
        steps: int,
        q_net: nnx.Module = None,
        log_interval_steps: int = 100_000,
    ) -> jax.Array:
        """Train the q-network. Returns the trained q-network."""

        # create default network if none given
        if q_net is None:
            q_net = self.create_default_q_net(rngs)

        optimizer = nnx.Optimizer(q_net, optax.adamw(learning_rate=Hyperparameters.learning_rate))

        ## initialize ##
        replay_buffer_state = self.replay_buffer.init()

        reset_keys = jax.random.split(rngs.env(), self.hyperparameters.n_envs)
        env_states, info = jax.vmap(self.env.reset)(reset_keys)

        training_state = DQN.TrainingState(
            steps = jnp.array(0, dtype=jnp.int32),
            env_states = env_states,
            epsilon = jnp.array(self.hyperparameters.epsilon_initial, dtype=jnp.float32),

            policy_q_net = q_net,
            target_q_net = nnx.clone(q_net),
            optimizer=optimizer
        )

        epsilon_anneal_amount = self.hyperparameters.epsilon_initial - self.hyperparameters.epsilon_final
        epsilon_anneal_steps = steps * self.hyperparameters.epsilon_anneal_fraction
        epsilon_anneal_rate = epsilon_anneal_amount / epsilon_anneal_steps

        while training_state.steps < steps:
            training_state, replay_buffer_state = self.train_epoch(rngs, 
                log_interval_steps, epsilon_anneal_rate, training_state, replay_buffer_state)

            print(f"Completed steps={training_state.steps}")

        return training_state.policy_q_net

    @functools.partial(nnx.jit, static_argnames=('self', 'steps'))
    def train_epoch(
        self, rngs: nnx.Rngs, steps: int, epsilon_anneal_rate: float,
        training_state: TrainingState, replay_buffer_state: ReplayBufferState
    ) -> tuple[TrainingState, ReplayBufferState]:
        """Train for one 'epoch' -- one fully JIT compiled segment."""

        steps_per_update = self.hyperparameters.train_freq * self.hyperparameters.n_envs

        def train_iteration(carry: tuple[DQN.TrainingState, ReplayBufferState], rngs: nnx.Rngs):
            training_state, replay_buffer_state = carry

            env_states = training_state.env_states
            steps = training_state.steps
            epsilon = training_state.epsilon

            policy_q_net = training_state.policy_q_net
            target_q_net = training_state.target_q_net
            optimizer = training_state.optimizer
            
            ## sample transitions from environment ##

            def env_step(env_state: TEnvState, rngs: nnx.Rngs):
                cur_obs = self.env.get_obs(rngs.env(), env_state)
                action = self.get_action(rngs, policy_q_net, epsilon, cur_obs)
                new_state, reward, terminated, truncated, info = self.env.step(rngs.env(), env_state, action)
                new_obs = self.env.get_obs(rngs.env(), new_state)

                # reset env if terminated/truncated, don't otherwise
                next_state = nnx.cond(jnp.logical_or(terminated, truncated), 
                    lambda rngs: self.env.reset(rngs.env())[0], lambda rngs: new_state, rngs)

                return (
                    next_state,
                    self.Transition(cur_obs=cur_obs, action=action, reward=reward, new_obs=new_obs, terminated=terminated)
                )

            env_steps_rngs = rngs.fork(split=self.hyperparameters.train_freq)

            def batched_env_step(env_states: TEnvState, rngs: nnx.Rngs):
                cur_env_step_rngs = rngs.fork(split=self.hyperparameters.n_envs)
                return nnx.vmap(env_step)(env_states, cur_env_step_rngs)

            env_states, transitions = nnx.scan(batched_env_step, length=self.hyperparameters.train_freq)(
                env_states, env_steps_rngs)

            steps += steps_per_update

            epsilon = jnp.maximum(epsilon - epsilon_anneal_rate*steps_per_update, self.hyperparameters.epsilon_final)

            transitions: DQN.Transition = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), 
                transitions) # flatten to remove axis 0
            replay_buffer_state = self.replay_buffer.insert(replay_buffer_state, transitions)

            ## update policy q-table ##

            sampled_transitions = self.replay_buffer.sample(rngs.transitions(), 
                replay_buffer_state, self.hyperparameters.batch_size)

            max_next_qs = jnp.max(target_q_net(sampled_transitions.new_obs, rngs=rngs.actions()), axis=1)
            # zero out q_val if terminated
            max_next_qs = max_next_qs * jnp.logical_not(sampled_transitions.terminated)

            target_qs = sampled_transitions.reward + self.hyperparameters.discount_rate*max_next_qs

            def loss_func(policy_q_net: nnx.Module, rngs: nnx.Rngs):
                pred_qs_all_actions = policy_q_net(sampled_transitions.cur_obs, rngs=rngs.actions())
                pred_qs = jnp.take_along_axis(pred_qs_all_actions, sampled_transitions.action[None, :], axis=1)

                # simple MSE loss
                return jnp.mean(jnp.pow(target_qs - pred_qs, 2))

            loss_grad_func = nnx.value_and_grad(loss_func)
            loss, grads = loss_grad_func(policy_q_net, rngs)
            optimizer.update(grads) 

            # update target_q_vals if enough steps have passed
            update_target_net = steps % self.hyperparameters.target_update_interval < steps_per_update

            nnx.update(target_q_net, nnx.cond(update_target_net, 
                lambda policy_q_net, target_q_net: nnx.state(policy_q_net), 
                lambda policy_q_net, target_q_net: nnx.state(target_q_net),
                policy_q_net, target_q_net
            ))

            return (DQN.TrainingState(
                steps=steps,
                env_states=env_states,
                epsilon=epsilon,

                policy_q_net=policy_q_net,
                target_q_net=target_q_net,
                optimizer=optimizer
            ), replay_buffer_state), None

        iterations = math.ceil(steps / (self.hyperparameters.train_freq * self.hyperparameters.n_envs))

        train_iter_rngs = rngs.fork(split=iterations)
        carry, _ = nnx.scan(train_iteration, length=iterations)((training_state, replay_buffer_state), train_iter_rngs)

        training_state, replay_buffer_state = carry

        return training_state, replay_buffer_state