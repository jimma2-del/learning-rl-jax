from typing import TypeVar, Generic, Any

import math

import numpy as np

import jax.numpy as jnp
import jax
from jax.typing import ArrayLike
from chex import dataclass

from flax import nnx
import optax

from core.utils import ReplayBuffer, ReplayBufferState
from core.utils.func_utils import try_call, optionally_pass
from core.utils.networks import MLP, FlattenAndProject

from core.algos.base import Scheduleable, GreedyQActor, DiscreteQFunc

from core.envs.base import Environment
from core.envs.wrappers import AutoResetWrapper
from core.envs.utils import rollout, Actor

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
    actor: GreedyQActor

    replay_buffer_state: ReplayBufferState[Transition[TEnvObs]]
    policy_q_func: DiscreteQFunc
    target_q_func: DiscreteQFunc
    optimizer: nnx.Optimizer

class DQN(Generic[TEnvState, TEnvObs]):
    """Implementation of DQN."""

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

    def make_default_q_func(self, rngs: nnx.Rngs, 
        hidden_dim: int = 256, num_hidden_layers: int = 2, do_layer_norm: bool = True, activation_func=nnx.relu
    ) -> nnx.Module:
        assert num_hidden_layers >= 1, "`num_hidden_layers` must be at least 1."
        layers = [ FlattenAndProject[TEnvObs](rngs, self.env.observation_space.shapes_dtypes, output_dim=hidden_dim) ]

        if do_layer_norm: layers.append(nnx.LayerNorm(hidden_dim, rngs=rngs))
        layers.append(activation_func)

        layers.append(MLP(rngs, 
            input_dim=hidden_dim, output_dim=self.num_actions,
            hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers-1, 
            do_layer_norm=do_layer_norm, activation_func=activation_func
        ))

        return nnx.Sequential(*layers)

    def make_actor(self, q_func: DiscreteQFunc | None = None, epsilon: ArrayLike | None = None, **kwargs) -> GreedyQActor:
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
        policy_q_func: DiscreteQFunc | None = None,
        target_q_func: DiscreteQFunc | None = None,
        replay_buffer_state: ReplayBufferState[Transition[TEnvObs]] | None = None,
        prefill_steps: int = 10_000
    ) -> TrainingState[TEnvState, TEnvObs]:
        if policy_q_func is None: policy_q_func = self.make_default_q_func(rngs)
        if target_q_func is None: target_q_func = nnx.clone(policy_q_func)

        optimizer = nnx.Optimizer(policy_q_func, optax.inject_hyperparams(optax.adamw)(
            learning_rate=try_call(self.hyperparameters.learning_rate, 0)))

        actor = self.make_actor(policy_q_func)

        if replay_buffer_state is None:
            replay_buffer_state = self.replay_buffer.init()

        # prefill replay buffer
        transitions, env_states = nnx.jit(self.rollout_transitions, static_argnames=('iter', 'actor'))(rngs,
            lambda obs, rngs: actor.random_action(rngs, (self.hyperparameters.n_envs,)),
            math.ceil(prefill_steps / self.hyperparameters.n_envs),
        )

        replay_buffer_state = self.replay_buffer.insert(replay_buffer_state, transitions)

        return TrainingState(
            steps = jnp.array(0, dtype=jnp.int32),
            env_states = env_states,
            actor = actor,

            replay_buffer_state = replay_buffer_state,
            policy_q_func = policy_q_func,
            target_q_func = target_q_func,
            optimizer = optimizer
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
            optimizer = training_state.optimizer    
            
            ## sample transitions from environment ##
            actor.epsilon.value = try_call(self.hyperparameters.epsilon, steps)

            actor.eval()
            actor.deterministic = False

            transitions, env_states = self.rollout_transitions(rngs, 
                actor, steps_per_env_per_iter, env_states)

            replay_buffer_state = self.replay_buffer.insert(replay_buffer_state, transitions)

            steps += total_steps_per_iter

            # update optimizer schedules using env steps (rather than default grad steps)
            optimizer.opt_state.hyperparams['learning_rate'].value \
                = try_call(self.hyperparameters.learning_rate, steps)

            ## update q functions ##
            policy_q_func.train()

            def learn_step(carry: tuple[DiscreteQFunc, nnx.Optimizer], rngs: nnx.Rngs) \
                    -> tuple[tuple[DiscreteQFunc, nnx.Optimizer], dict[Any, Any]]:
                policy_q_func, optimizer = carry

                sampled_transitions = self.replay_buffer.sample(rngs.transitions(), 
                    replay_buffer_state, self.hyperparameters.batch_size)

                next_qs = optionally_pass(target_q_func, rngs=rngs)(sampled_transitions.next_obs)
                max_next_qs = jnp.max(next_qs, axis=-1)
                # zero out q_val if terminated
                max_next_qs = max_next_qs * jnp.logical_not(sampled_transitions.terminated)

                target_qs = sampled_transitions.reward \
                    + try_call(self.hyperparameters.discount_rate, steps)*max_next_qs

                def loss_func(policy_q_func: nnx.Module, rngs: nnx.Rngs):
                    pred_qs_all_actions = optionally_pass(policy_q_func, rngs=rngs)(sampled_transitions.obs)
                        # q-net returns a q-value for every action
                    pred_qs = pred_qs_all_actions[jnp.arange(self.hyperparameters.batch_size), sampled_transitions.action]
                        # take only the q-value corresponding to the chosen action

                    # simple MSE loss
                    return jnp.mean(jnp.power(target_qs - pred_qs, 2))

                loss_grad_func = nnx.value_and_grad(loss_func)
                loss, grads = loss_grad_func(policy_q_func, rngs)
                optimizer.update(grads) 

                return (policy_q_func, optimizer), { 'q_loss': loss }

            (policy_q_func, optimizer), metrics = nnx.scan(learn_step)((policy_q_func, optimizer), 
                rngs.fork(split=learn_steps_per_iter))

            # update target if enough steps have passed
            update_target = steps % self.hyperparameters.target_update_interval < total_steps_per_iter
            nnx.update(target_q_func, nnx.cond(update_target, 
                lambda policy_q_func, target_q_func: nnx.state(policy_q_func), 
                lambda policy_q_func, target_q_func: nnx.state(target_q_func),
                policy_q_func, target_q_func
            ))
            target_q_func.eval()

            return TrainingState(
                steps=steps,
                env_states=env_states,
                actor=actor,

                replay_buffer_state=replay_buffer_state,
                policy_q_func=policy_q_func,
                target_q_func=target_q_func,
                optimizer=optimizer
            ), jax.tree.map(lambda x: jnp.mean(x), metrics)

        training_state.actor.eval()
        training_state.actor.deterministic = False

        training_state.policy_q_func.train()
        training_state.target_q_func.eval()

        iterations = math.ceil(steps / total_steps_per_iter)
        training_state, metrics = nnx.scan(train_iteration)(training_state, rngs.fork(split=iterations))

        return training_state, jax.tree.map(lambda x: jnp.mean(x), metrics)