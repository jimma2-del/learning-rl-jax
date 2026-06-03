import math

import jax.numpy as jnp
import jax

from jax.typing import ArrayLike
from chex import dataclass
from typing import TypeVar, Generic, Any

import functools

from flax import nnx
import optax

from core.algos.base import Scheduleable
from core.utils.func_utils import try_call, optionally_pass

from core.envs.base import Environment
from core.envs.wrappers import AutoResetWrapper
from core.envs.utils import parallel_rollout
from core.utils import ReplayBuffer, ReplayBufferState

from core.sample_networks import MLP, MLPFeatureExtractor

@dataclass(frozen=True)
class Hyperparameters:
    n_envs: int = 32

    discount_rate: Scheduleable[float] = 0.99
    learning_rate: Scheduleable[float] = 2.5e-4

    batch_size: int = 32

    epsilon: Scheduleable[float] = 0.05
        # it is recommended to use a schedule: decay from 1 to ~0.05 over ~10% of training steps
        # eg. optax.schedules.linear_schedule(1, 0.05, 0.1*steps)

    replay_buffer_size: int = 1_000_000

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

    policy: nnx.Module 
        # perhaps named confusingly; policy q-network, used for getting actions, though not directly an actor
        # named like this so api matches with other algos
    target: nnx.Module
    optimizer: nnx.Optimizer

class DQN(Generic[TEnvState, TEnvObs]):
    """Implementation of DQN."""

    def __init__(self, 
        env: Environment[TEnvState, TEnvObs, ArrayLike],
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

    def get_greedy_action(self, rngs: nnx.Rngs, policy: nnx.module, obs: TEnvObs) -> ArrayLike:
        q_vals = optionally_pass(policy, rngs=rngs)(obs)
        return jnp.argmax(q_vals)

    def get_random_action(self, rngs: nnx.Rngs) -> ArrayLike:
        return jax.random.randint(rngs.actions(), shape=(), minval=0, maxval=self.num_actions)

    def get_action(self, rngs: nnx.Rngs, policy: nnx.module, obs: TEnvObs, epsilon: ArrayLike = 0) -> ArrayLike:
        random_action = self.get_random_action(rngs)
        greedy_action = self.get_greedy_action(rngs, policy, obs)

        return jnp.where(jax.random.uniform(rngs.actions()) < epsilon, 
            random_action, greedy_action)

    def create_default_policy(self, rngs: nnx.Rngs) -> nnx.Module:
        FEATURE_EXTRACTOR_OUTPUT_DIM = 256

        return nnx.Sequential(
            MLPFeatureExtractor[TEnvObs](rngs, self.env.observation_space.shapes_dtypes, 
                output_dim=FEATURE_EXTRACTOR_OUTPUT_DIM),
            MLP(rngs, input_dim=FEATURE_EXTRACTOR_OUTPUT_DIM, output_dim=self.num_actions)
        )

    def rollout_transitions(self,
        rngs: nnx.Rngs, 
        policy: nnx.module, 
        iter: int,
        initial_env_states: TEnvState | None = None,
        epsilon: ArrayLike = 0
    ) -> tuple[Transition[TEnvObs], TEnvState]:
        """Collect a rollout of `Transition`s.

        Runs `n_envs` environments in parallel for `iter` steps each,
            for a total of `iter * n_envs` transitions.

        Initializes initial environment states if none given.

        Returns: transitions, final environment states
        """

        timesteps, env_states, final_infos = parallel_rollout(
            rngs, self.env,
            nnx.vmap(lambda obs, rngs: self.get_action(rngs, policy, obs, epsilon)),
            iter, self.hyperparameters.n_envs,
            initial_env_states,
        )

        next_obs = jax.vmap(jax.vmap(self.env.get_obs))(
            jnp.reshape(
                jax.random.split(rngs.env(), iter * self.hyperparameters.n_envs), 
                (iter, self.hyperparameters.n_envs)
            ), 
            jax.tree.map(lambda middles, finals: jnp.append(middles[1:], finals[None, ...], axis=0),
                timesteps.info[AutoResetWrapper.UNRESET_STATE_INFO_KEY],
                final_infos[AutoResetWrapper.UNRESET_STATE_INFO_KEY]
            )
        )

        timesteps = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), timesteps) # flatten to remove axis 0
        next_obs = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), next_obs) # flatten to remove axis 0

        transitions = Transition(obs=timesteps.obs, action=timesteps.action, 
            reward=timesteps.reward, next_obs=next_obs, terminated=timesteps.terminated)

        return transitions, env_states

    def init_training_state(self,
        rngs: nnx.Rngs,
        policy: nnx.Module | None = None,
        replay_buffer_state: ReplayBufferState[Transition[TEnvObs]] | None = None,
        prefill_steps: int = 10_000
    ) -> TrainingState[TEnvState, TEnvObs]:
        # create default network if none given
        if policy is None:
            policy = self.create_default_policy(rngs)

        optimizer = nnx.Optimizer(policy, optax.inject_hyperparams(optax.adamw)(
            learning_rate=try_call(self.hyperparameters.learning_rate, 0)))
        #optimizer = nnx.Optimizer(q_net, optax.adamw(learning_rate=2.5e-4))

        if replay_buffer_state is None:
            replay_buffer_state = self.replay_buffer.init()

        # prefill replay buffer
        transitions, env_states = nnx.jit(self.rollout_transitions, static_argnames=('iter'))(rngs,
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
            target = nnx.clone(policy),
            optimizer = optimizer
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
            optimizer = training_state.optimizer    
            
            ## sample transitions from environment ##

            transitions, env_states = self.rollout_transitions(rngs, 
                policy,
                steps_per_env_per_iter,
                env_states,
                epsilon=try_call(self.hyperparameters.epsilon, steps)
            )

            replay_buffer_state = self.replay_buffer.insert(replay_buffer_state, transitions)

            steps += total_steps_per_iter

            # update optimizer schedules using env steps (rather than default grad steps)
            optimizer.opt_state.hyperparams['learning_rate'].value \
                = try_call(self.hyperparameters.learning_rate, steps)

            ## update policy ##

            def learn_step(carry: tuple[nnx.Module, nnx.Optimizer], rngs: nnx.Rngs) \
                    -> tuple[tuple[nnx.Module, nnx.Optimizer], dict[Any, Any]]:
                policy, optimizer = carry

                sampled_transitions = self.replay_buffer.sample(rngs.transitions(), 
                    replay_buffer_state, self.hyperparameters.batch_size)

                next_qs = optionally_pass(target, rngs=rngs)(sampled_transitions.next_obs)
                max_next_qs = jnp.max(next_qs, axis=1)
                # zero out q_val if terminated
                max_next_qs = max_next_qs * jnp.logical_not(sampled_transitions.terminated)

                target_qs = sampled_transitions.reward \
                    + try_call(self.hyperparameters.discount_rate, steps)*max_next_qs

                def loss_func(policy: nnx.Module, rngs: nnx.Rngs):
                    pred_qs_all_actions = optionally_pass(policy, rngs=rngs)(sampled_transitions.obs)
                        # q-net returns a q-value for every action
                    pred_qs = pred_qs_all_actions[jnp.arange(self.hyperparameters.batch_size), sampled_transitions.action]
                        # take only the q-value corresponding to the chosen action

                    # simple MSE loss
                    return jnp.mean(jnp.power(target_qs - pred_qs, 2))

                loss_grad_func = nnx.value_and_grad(loss_func)
                loss, grads = loss_grad_func(policy, rngs)
                optimizer.update(grads) 

                return (policy, optimizer), { 'critic_loss': loss }

            (policy, optimizer), metrics = nnx.scan(learn_step)((policy, optimizer), 
                rngs.fork(split=learn_steps_per_iter))

            # update target if enough steps have passed
            update_target = steps % self.hyperparameters.target_update_interval < total_steps_per_iter

            nnx.update(target, nnx.cond(update_target, 
                lambda policy, target: nnx.state(policy), 
                lambda policy, target: nnx.state(target),
                policy, target
            ))

            return TrainingState(
                steps=steps,
                env_states=env_states,
                replay_buffer_state=replay_buffer_state,

                policy=policy,
                target=target,
                optimizer=optimizer
            ), jax.tree.map(lambda x: jnp.mean(x), metrics)

        iterations = math.ceil(epoch_steps / total_steps_per_iter)
        training_state, metrics = nnx.scan(train_iteration)(training_state, rngs.fork(split=iterations))

        return training_state, jax.tree.map(lambda x: jnp.mean(x), metrics)