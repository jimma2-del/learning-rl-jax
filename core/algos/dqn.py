import math

import jax.numpy as jnp
import jax

from jax.typing import ArrayLike
from chex import dataclass
from typing import TypeVar, Generic, Sequence, Callable

import functools

from flax import nnx
import optax

from core.algos.base import Scheduleable, resolve_scheduleable

from core.envs.base import Environment
from core.envs.wrappers import VmapAutoResetWrapper, VmapWrapper, AutoResetWrapper
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

    def rollout(self,
        rngs: nnx.Rngs, 
        q_net: nnx.module, epsilon: ArrayLike, 
        iter: int,
        initial_env_states: TEnvState | None = None
    ) -> Transition:
        """Collect a rollout of `Transition`s.

        Runs `n_envs` environments in parallel for `iter` iterations,
            for a total of `iter * n_envs` transitions.

        Initializes initial environment states if none given.

        Returns: transitions, final environment states
        """

        #env = VmapWrapper(AutoResetWrapper(self.env))
        env = VmapAutoResetWrapper(self.env)

        def batched_env_step(states: TEnvState, rngs: nnx.Rngs) -> tuple[TEnvState, DQN.Transition]:
            obs = env.get_obs(rngs.env(), states)

            actions = nnx.vmap(lambda rngs, obs: self.get_action(rngs, q_net, epsilon, obs))(
                rngs.fork(split=self.hyperparameters.n_envs), obs)

            new_states, rewards, terminated, truncated, infos = env.step(rngs.env(), states, actions)
            next_obs = env.get_obs(rngs.env(), infos.pop(env.NEXT_STATE_INFO_KEY))

            return (
                new_states,
                DQN.Transition(obs=obs, action=actions, reward=rewards, next_obs=next_obs,terminated=terminated)
            )

        if initial_env_states is None:
            initial_env_states, info = env.reset(rngs.env(), num=self.hyperparameters.n_envs)

        env_states, transitions = nnx.scan(batched_env_step)(initial_env_states, rngs.fork(split=iter))
        transitions = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), transitions) # flatten to remove axis 0

        return transitions, env_states

    def train(self,
        rngs: nnx.Rngs,
        steps: int,
        q_net: nnx.Module = None,
        log_interval_steps: int = 100_000,
        prefill_steps: int = 10_000,
        callbacks: Sequence[Callable[[TrainingState], None]] = []
    ) -> jax.Array:
        """Train the q-network. Returns the trained q-network."""

        # create default network if none given
        if q_net is None:
            q_net = self.create_default_q_net(rngs)

        optimizer = nnx.Optimizer(q_net, optax.inject_hyperparams(optax.adamw)(
            learning_rate=resolve_scheduleable(self.hyperparameters.learning_rate, 0)))
        #optimizer = nnx.Optimizer(q_net, optax.adamw(learning_rate=2.5e-4))

        ## initialize ##

        # prefill replay buffer
        transitions, env_states = nnx.jit(self.rollout, static_argnames=('iter'))(rngs,
            q_net, 1,
            math.ceil(prefill_steps / self.hyperparameters.n_envs)
        )

        replay_buffer_state = self.replay_buffer.init()
        replay_buffer_state = self.replay_buffer.insert(replay_buffer_state, transitions)

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

            for callback in callbacks:
                callback(training_state)

        return training_state.policy_q_net
    
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

            transitions, env_states = self.rollout(rngs, 
                policy_q_net, resolve_scheduleable(self.hyperparameters.epsilon, steps),
                steps_per_env_per_iter,
                env_states
            )

            replay_buffer_state = self.replay_buffer.insert(replay_buffer_state, transitions)

            steps += total_steps_per_iter

            # update optimizer schedules using env steps (rather than default grad steps)
            optimizer.opt_state.hyperparams['learning_rate'].value \
                = resolve_scheduleable(self.hyperparameters.learning_rate, steps)

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