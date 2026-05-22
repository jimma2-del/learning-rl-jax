import math

import jax.numpy as jnp
import jax

from jax.typing import ArrayLike
from chex import dataclass
from typing import TypeVar, Generic

import functools

from flax import nnx
import optax

from core.algos.base import Scheduleable, resolve_scheduleable

from core.envs.base import Environment
from core.envs.utils import rollout_auto_reset
from core.utils import ReplayBuffer, ReplayBufferState

from core.sample_networks import MLP, MLPFeatureExtractor

@dataclass(frozen=True)
class DQNHyperparameters:
    n_envs: int = 32

    discount_rate: Scheduleable[float] = 0.99
    learning_rate: Scheduleable[float] = 2.5e-4

    batch_size: int = 32

    epsilon: Scheduleable[float] = 0.1
        # it is recommended to use a schedule: decay from 1 to ~0.05 over ~10% of training steps
        # eg. optax.schedules.linear_schedule(1, 0.05, 0.1*steps)

    replay_buffer_size: int = 1_000_000

    train_freq: int = 4 # does around 1 gradient step per train_freq env steps
        # NOTE: if n_envs > train_freq, we take 1 step in each env, followed by multiple gradient steps
        # NOTE: will round up or down if not divisible evenly

    target_update_interval: int = 1000

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")

class DQN(Generic[TEnvState, TEnvObs]):
    """Implementation of DQN."""

    @dataclass(frozen=True)
    class Transition:
        obs: TEnvObs
        action: ArrayLike
        reward: ArrayLike
        next_obs: TEnvObs
        terminated: ArrayLike

    @dataclass(frozen=True)
    class TrainingState:
        steps: ArrayLike
        env_states: TEnvState

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
            obs = self.env.observation_space.shapes_dtypes,
            action = self.env.action_space.shapes_dtypes,
            reward = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
            next_obs = self.env.observation_space.shapes_dtypes,
            terminated = jax.ShapeDtypeStruct(shape=(), dtype=jnp.bool)
        )

        self.replay_buffer = ReplayBuffer[DQN.Transition](
            self.transition_shapes_dtypes, self.hyperparameters.replay_buffer_size)

    def get_greedy_action(self, rngs: nnx.Rngs, q_net: nnx.module, obs: TEnvObs) -> ArrayLike:
        q_vals = q_net(obs, rngs=rngs)
        return jnp.argmax(q_vals)

    def get_random_action(self, rngs: nnx.Rngs) -> ArrayLike:
        return jax.random.randint(rngs.actions(), shape=(), minval=0, maxval=self.num_actions)

    def get_action(self, rngs: nnx.Rngs, q_net: nnx.module, 
        epsilon: ArrayLike, obs: TEnvObs) -> ArrayLike:

        random_action = self.get_random_action(rngs)
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
        prefill_steps: int = 10_000
    ) -> jax.Array:
        """Train the q-network. Returns the trained q-network."""

        # create default network if none given
        if q_net is None:
            q_net = self.create_default_q_net(rngs)

        optimizer = nnx.Optimizer(q_net, optax.inject_hyperparams(optax.adamw)(
            learning_rate=resolve_scheduleable(self.hyperparameters.learning_rate, 0)))
        #optimizer = nnx.Optimizer(q_net, optax.adamw(learning_rate=2.5e-4))

        ## initialize ##
        replay_buffer_state = self.prefill_replay_buffer(rngs, prefill_steps)

        reset_keys = jax.random.split(rngs.env(), self.hyperparameters.n_envs)
        env_states, info = jax.vmap(self.env.reset)(reset_keys)

        training_state = DQN.TrainingState(
            steps = jnp.array(0, dtype=jnp.int32),
            env_states = env_states,

            policy_q_net = q_net,
            target_q_net = nnx.clone(q_net),
            optimizer = optimizer
        )

        while training_state.steps < steps:
            training_state, replay_buffer_state = self.train_epoch(rngs, 
                log_interval_steps, training_state, replay_buffer_state)

            print(f"Completed steps={training_state.steps}")

        return training_state.policy_q_net
    
    @functools.partial(nnx.jit, static_argnames=('self', 'prefill_steps'))
    def prefill_replay_buffer(self, rngs: nnx.Rngs, prefill_steps: int) -> ReplayBufferState:
        transitions, env_states = rollout_auto_reset(rngs, 
            self.env, 
            lambda rngs, obs: self.get_random_action(rngs), 
            math.ceil(prefill_steps / self.hyperparameters.n_envs), self.hyperparameters.n_envs
        )

        transitions = jax.vmap(lambda x: DQN.Transition())

        replay_buffer_state = self.replay_buffer.init()

        return self.replay_buffer.insert(replay_buffer_state, transitions)

    @functools.partial(nnx.jit, static_argnames=('self', 'steps'))
    def train_epoch(
        self, rngs: nnx.Rngs, steps: int,
        training_state: TrainingState, replay_buffer_state: ReplayBufferState
    ) -> tuple[TrainingState, ReplayBufferState]:
        """Train for one 'epoch' -- one fully JIT compiled segment."""

        steps_per_env_per_iter = math.ceil(self.hyperparameters.train_freq / self.hyperparameters.n_envs)
        total_steps_per_iter = steps_per_env_per_iter * self.hyperparameters.n_envs
        grad_steps_per_iter = math.ceil(self.hyperparameters.n_envs / self.hyperparameters.train_freq)

        def train_iteration(carry: tuple[DQN.TrainingState, ReplayBufferState], rngs: nnx.Rngs):
            training_state, replay_buffer_state = carry

            env_states = training_state.env_states
            steps = training_state.steps

            policy_q_net = training_state.policy_q_net
            target_q_net = training_state.target_q_net
            optimizer = training_state.optimizer    
            
            ## sample transitions from environment ##

            def env_step(env_state: TEnvState, rngs: nnx.Rngs) -> tuple[TEnvState, DQN.Transition]:
                cur_obs = self.env.get_obs(rngs.env(), env_state)

                action = self.get_action(rngs, policy_q_net, 
                    resolve_scheduleable(self.hyperparameters.epsilon, steps), cur_obs)

                next_state, reward, terminated, truncated, info = self.env.step(rngs.env(), env_state, action)
                next_obs = self.env.get_obs(rngs.env(), info[self.env.NEXT_STATE_INFO_KEY])

                return (
                    next_state,
                    self.Transition(cur_obs=cur_obs, action=action, reward=reward, next_obs=next_obs, terminated=terminated)
                )

            def batched_env_step(env_states: TEnvState, rngs: nnx.Rngs) -> tuple[TEnvState, DQN.Transition]:
                return nnx.vmap(env_step)(env_states, rngs.fork(split=self.hyperparameters.n_envs))
            
            env_states, transitions = nnx.scan(batched_env_step)(env_states, 
                rngs.fork(split=steps_per_env_per_iter))

            steps += total_steps_per_iter

            # update optimizer schedules using env steps (rather than default grad steps)
            optimizer.opt_state.hyperparams['learning_rate'].value \
                = resolve_scheduleable(self.hyperparameters.learning_rate, steps)

            transitions: DQN.Transition = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), 
                transitions) # flatten to remove axis 0
            replay_buffer_state = self.replay_buffer.insert(replay_buffer_state, transitions)

            ## update policy q-table ##

            def grad_update(carry: tuple[nnx.Module, nnx.Optimizer], rngs: nnx.Rngs) \
                -> tuple[tuple[nnx.Module, nnx.Optimizer], ArrayLike]:
                policy_q_net, optimizer = carry

                sampled_transitions = self.replay_buffer.sample(rngs.transitions(), 
                    replay_buffer_state, self.hyperparameters.batch_size)

                max_next_qs = jnp.max(target_q_net(sampled_transitions.next_obs, rngs=rngs), axis=1)
                # zero out q_val if terminated
                max_next_qs = max_next_qs * jnp.logical_not(sampled_transitions.terminated)

                target_qs = sampled_transitions.reward \
                    + resolve_scheduleable(self.hyperparameters.discount_rate, steps)*max_next_qs

                def loss_func(policy_q_net: nnx.Module, rngs: nnx.Rngs):
                    pred_qs_all_actions = policy_q_net(sampled_transitions.obs, rngs=rngs)
                        # q-net returns a q-value for every action
                    pred_qs = pred_qs_all_actions[jnp.arange(self.hyperparameters.batch_size), sampled_transitions.action]
                        # take only the q-value corresponding to the chosen action

                    # simple MSE loss
                    return jnp.mean(jnp.power(target_qs - pred_qs, 2))

                loss_grad_func = nnx.value_and_grad(loss_func)
                loss, grads = loss_grad_func(policy_q_net, rngs)
                optimizer.update(grads) 

                return (policy_q_net, optimizer), loss

            (policy_q_net, optimizer), losses = nnx.scan(grad_update)((policy_q_net, optimizer), 
                rngs.fork(split=grad_steps_per_iter))

            # update target_q_vals if enough steps have passed
            update_target_net = steps % self.hyperparameters.target_update_interval < total_steps_per_iter

            nnx.update(target_q_net, nnx.cond(update_target_net, 
                lambda policy_q_net, target_q_net: nnx.state(policy_q_net), 
                lambda policy_q_net, target_q_net: nnx.state(target_q_net),
                policy_q_net, target_q_net
            ))

            return (DQN.TrainingState(
                steps=steps,
                env_states=env_states,

                policy_q_net=policy_q_net,
                target_q_net=target_q_net,
                optimizer=optimizer
            ), replay_buffer_state), None

        iterations = math.ceil(steps / total_steps_per_iter)

        (training_state, replay_buffer_state), _ = nnx.scan(train_iteration)((training_state, replay_buffer_state), 
            rngs.fork(split=iterations))

        return training_state, replay_buffer_state