from typing import TypeVar, Generic, Any, Sequence, Self, Callable

import math

import jax.numpy as jnp
import jax
from jax.typing import ArrayLike
from chex import dataclass

from flax import nnx
import optax

from core.utils import ReplayBuffer, RunningMeanVar
from core.utils.func_utils import try_call, optionally_pass
from core.utils.nnx_modules import MLP, RunningMeanVarNorm

from core.algos.base import Scheduleable, GreedyQActor, AlgoPhase, set_algo_phase

from core.envs.base import Environment, Space
from core.envs.wrappers import AutoResetWrapper
from core.envs.utils import rollout, Actor, RandomActor

@dataclass(frozen=True)
class Hyperparameters:
    n_envs: int = 256

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

    target_update_interval: int = 10_000 # environment steps

    polyak_tau: Scheduleable[float] | None = None # if not None, OVERRIDES `target_update_interval``
        # target is updated after EVERY gradient step if not None

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")
TTrunkOut = TypeVar("TTrunkOut")

class Networks(nnx.Module, Generic[TEnvObs, TTrunkOut]):
    def __init__(self, obs_trunk: Callable[[TEnvObs], TTrunkOut], qs_head: Callable[[TTrunkOut], jax.Array]) -> None:
        self.obs_trunk = obs_trunk
        self.qs_head = qs_head

    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs | None = None) -> jax.Array:
        trunk_out = optionally_pass(self.obs_trunk, rngs=rngs)(obs)
        return optionally_pass(self.qs_head, rngs=rngs)(trunk_out)

    @classmethod
    def make_default(cls, rngs: nnx.Rngs, observation_space: Space[TEnvObs], action_space: Space[ArrayLike]) -> Self:
        return cls(
            cls.make_default_obs_trunk(observation_space),
            cls.make_default_qs_head(rngs, observation_space.flattened_dim, action_space)
        )

    @staticmethod
    def make_default_obs_trunk(
        observation_space: Space[TEnvObs],
        normalize_observations: bool = True, 
        obs_running_mean_var: RunningMeanVar[TEnvObs] | None = None, 
        obs_clip_threshold: float | None = None
    ) -> Callable[[TEnvObs], TTrunkOut]:
        layers = []

        if normalize_observations:
            inp = observation_space.shapes_dtypes if obs_running_mean_var is None else obs_running_mean_var
            layers.append(RunningMeanVarNorm(inp, clip_threshold=obs_clip_threshold))

        layers.append(observation_space.flatten)

        return nnx.Sequential(*layers)

    @staticmethod
    def make_default_qs_head(
        rngs: nnx.Rngs, input_dim: int, action_space: Space[ArrayLike],
        hidden_dims: Sequence[int] = (128, 128), do_layer_norm: bool = True, activation_func=nnx.relu
    ) -> Callable[[TTrunkOut], jax.Array]:
        return MLP(
            rngs, (input_dim, *hidden_dims, int(action_space.high + 1)), 
            do_layer_norm=do_layer_norm, activation_func=activation_func
        )

@dataclass(frozen=True)
class Transition(Generic[TEnvObs]):
    obs: TEnvObs
    action: ArrayLike
    reward: ArrayLike
    next_obs: TEnvObs
    terminated: ArrayLike

@dataclass
class TrainingState(Generic[TEnvState, TEnvObs, TTrunkOut]):
    steps: ArrayLike
    env_states: TEnvState

    networks: Networks[TEnvObs, TTrunkOut]
    optimizer: nnx.Optimizer

    target_networks: Networks[TEnvObs, TTrunkOut]
    replay_buffer: ReplayBuffer[Transition[TEnvObs]]

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
        self.hyperparameters = hyperparameters

        # make replay buffer data shape
        self.transition_shapes_dtypes = Transition(
            obs = self.env.observation_space.shapes_dtypes,
            action = self.env.action_space.shapes_dtypes,
            reward = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
            next_obs = self.env.observation_space.shapes_dtypes,
            terminated = jax.ShapeDtypeStruct(shape=(), dtype=jnp.bool)
        )

    def make_actor(self, 
        networks: Networks[TEnvObs, TTrunkOut] | None = None, 
        epsilon: ArrayLike = jnp.array(0.0), 
        rngs: nnx.Rngs | None = None
    ) -> GreedyQActor[TEnvObs]:
        """`rngs` is only necessary if `networks` is not provided."""

        if networks is None: 
            networks = Networks.make_default(rngs, self.env.observation_space, self.env.action_space)

        return GreedyQActor(
            nnx.Sequential(networks.obs_trunk, networks.qs_head), 
            int(self.env.action_space.high + 1), 
            epsilon=epsilon
        )

    def rollout_transitions(self,
        rngs: nnx.Rngs, 
        actor: Actor[TEnvObs, ArrayLike], 
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
        networks: Networks[TEnvObs, TTrunkOut] | None = None,
        replay_buffer: ReplayBuffer[Transition[TEnvObs]] | None = None,
        prefill_steps: int = 10_000,
    ) -> TrainingState[TEnvState, TEnvObs, TTrunkOut]:
        if networks is None: 
            networks = Networks.make_default(rngs, self.env.observation_space, self.env.action_space)

        optimizer = nnx.Optimizer(networks, optax.inject_hyperparams(optax.adamw)(
            learning_rate=try_call(self.hyperparameters.learning_rate, 0)))

        target_networks = nnx.clone(networks)
        set_algo_phase(target_networks, AlgoPhase.EVAL)

        if self.hyperparameters.polyak_tau is not None: # convert datatypes to floats
            nnx.update(target_networks, optax.incremental_update(
                nnx.state(target_networks), nnx.state(target_networks), 0.5))

        if replay_buffer is None:
            replay_buffer = ReplayBuffer.init(
                self.transition_shapes_dtypes, self.hyperparameters.replay_buffer_size)

        # prefill replay buffer
        transitions, env_states = nnx.jit(self.rollout_transitions, static_argnames=('iter', 'actor'))(rngs,
            RandomActor(self.env.action_space, self.env.observation_space),
            math.ceil(prefill_steps / self.hyperparameters.n_envs),
        )

        replay_buffer = replay_buffer.insert(transitions)

        return TrainingState(
            steps = jnp.array(0, dtype=jnp.int32),
            env_states = env_states,

            networks = networks,
            optimizer = optimizer,

            target_networks = target_networks,
            replay_buffer = replay_buffer,
        )
    
    def train(self, rngs: nnx.Rngs, training_state: TrainingState[TEnvState, TEnvObs, TTrunkOut], steps: int) \
            -> tuple[TrainingState[TEnvState, TEnvObs, TTrunkOut], dict[Any, Any]]:
        """Train from the given `training_state`, returning an updated `training_state` and metrics."""

        steps_per_env_per_iter = math.ceil(self.hyperparameters.train_freq / self.hyperparameters.n_envs)
        total_steps_per_iter = steps_per_env_per_iter * self.hyperparameters.n_envs
        learn_steps_per_iter = math.ceil(self.hyperparameters.n_envs / self.hyperparameters.train_freq)

        def train_iteration(training_state: TrainingState[TEnvState, TEnvObs, TTrunkOut], rngs: nnx.Rngs) \
                -> tuple[TrainingState[TEnvState, TEnvObs, TTrunkOut], dict[Any, Any]]:
            ## sample transitions from environment ##
            set_algo_phase(training_state.networks, AlgoPhase.ROLLOUT)

            actor = self.make_actor(training_state.networks, 
                try_call(self.hyperparameters.epsilon, training_state.steps))
            transitions, training_state.env_states = self.rollout_transitions(rngs, 
                actor, steps_per_env_per_iter, training_state.env_states)
            training_state.steps += total_steps_per_iter

            training_state.replay_buffer = training_state.replay_buffer.insert(transitions)

            # update optimizer schedules using env steps (rather than default grad steps)
            training_state.optimizer.opt_state.hyperparams['learning_rate'].value \
                = try_call(self.hyperparameters.learning_rate, training_state.steps)

            ## update q functions ##
            set_algo_phase(training_state.networks, AlgoPhase.OPTIMIZE)

            def learn_step(carry: tuple[Networks, Networks, nnx.Optimizer], rngs: nnx.Rngs) \
                    -> tuple[tuple[Networks, Networks, nnx.Optimizer], dict[Any, Any]]:
                networks, target_networks, optimizer = carry

                sampled_transitions = training_state.replay_buffer.sample(
                    rngs.transitions(), (self.hyperparameters.batch_size,))

                next_qs = optionally_pass(target_networks, rngs=rngs)(sampled_transitions.next_obs)
                max_next_qs = jnp.max(next_qs, axis=-1)
                # zero out q_val if terminated
                max_next_qs = max_next_qs * jnp.logical_not(sampled_transitions.terminated)

                target_qs = sampled_transitions.reward \
                    + try_call(self.hyperparameters.discount_rate, training_state.steps)*max_next_qs

                def loss_func(networks: nnx.Module, rngs: nnx.Rngs):
                    pred_qs_all_actions = optionally_pass(networks, rngs=rngs)(sampled_transitions.obs)
                        # q-net returns a q-value for every action
                    pred_qs = pred_qs_all_actions[jnp.arange(self.hyperparameters.batch_size), sampled_transitions.action]
                        # take only the q-value corresponding to the chosen action

                    # simple MSE loss
                    return jnp.mean(jnp.power(target_qs - pred_qs, 2))

                loss_grad_func = nnx.value_and_grad(loss_func)
                loss, grads = loss_grad_func(networks, rngs)
                optimizer.update(grads) 

                # update target network if using polyak averaging
                if self.hyperparameters.polyak_tau is not None:
                    tau = try_call(self.hyperparameters.polyak_tau, training_state.steps)
                    nnx.update(target_networks, optax.incremental_update(
                        nnx.state(networks), nnx.state(target_networks), tau))

                return carry, { 'q_loss': loss }

            _, metrics = nnx.scan(learn_step)(
                (training_state.networks, training_state.target_networks, training_state.optimizer), 
                rngs.fork(split=learn_steps_per_iter)
            )

            # update target if enough steps have passed (not using polyak averaging)
            if self.hyperparameters.polyak_tau is None:
                update_target = training_state.steps % self.hyperparameters.target_update_interval < total_steps_per_iter
                nnx.update(training_state.target_networks, jax.lax.cond(update_target, 
                    lambda opt_state, target_state: opt_state, 
                    lambda opt_state, target_state: target_state,
                    nnx.state(training_state.networks), nnx.state(training_state.target_networks)
                ))

            return training_state, jax.tree.map(lambda x: jnp.mean(x), metrics)

        if self.hyperparameters.polyak_tau is not None: # convert datatypes to floats
            nnx.update(training_state.target_networks, optax.incremental_update(
                nnx.state(training_state.target_networks), nnx.state(training_state.target_networks), 0.5))

        # phases must match phases at the end of train_iteration
        set_algo_phase(training_state.target_networks, AlgoPhase.EVAL)
        set_algo_phase(training_state.networks, AlgoPhase.OPTIMIZE)

        iterations = math.ceil(steps / total_steps_per_iter)
        training_state, metrics = nnx.scan(train_iteration)(training_state, rngs.fork(split=iterations))

        # set into eval mode for the user
        set_algo_phase(training_state.networks, AlgoPhase.EVAL)

        return training_state, jax.tree.map(lambda x: jnp.mean(x), metrics)