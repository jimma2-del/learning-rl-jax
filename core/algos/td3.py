from typing import TypeVar, Generic, Any, Sequence, Self, Callable, Mapping, Literal

import math

import jax.numpy as jnp
import jax
from jax.typing import ArrayLike

from chex import dataclass
from dataclasses import field

from flax import nnx
import optax

from core.utils import RunningMeanVar
from core.utils.buffers import CircularBufferWithOptionalData
from core.utils.func_utils import try_call, optionally_pass, override_signature
from core.utils.nnx_modules import MLP, RunningMeanVarNorm, Pipe

from core.algos.base import Scheduleable, AlgoPhase, set_algo_phase, DeterministicPolicyActor, with_grad_clip

from core.envs.base import Environment, Space
from core.envs.wrappers import AutoResetWrapper
from core.envs.utils import rollout, Actor, RandomActor

@dataclass(frozen=True)
class Hyperparameters:
    n_envs: int = 256

    discount_rate: Scheduleable[float] = 0.99

    learning_rate: Scheduleable[float] = 2.5e-4
    max_grad_norm: Scheduleable[float] | None = 10.0

    policy_optimizer_params: Mapping[str, Scheduleable[float]] = field(
        default_factory=lambda: { 'weight_decay': 0.0 })
    q_func_optimizer_params: Mapping[str, Scheduleable[float]] = field(
        default_factory=lambda: { 'weight_decay': 0.0 })

    batch_size: int = 32

    replay_buffer_size: int = 1_000_000
    truncated_frac: float = 1.0 # fraction of timesteps expected to be truncated
        # lowering this saves memory by allocating less space for truncated observations in the replay buffer
        # however, truncated timesteps exceeding the specified limit will be treated as terminated

    train_freq: int = 4 # does around 1 gradient step per train_freq env steps
        # NOTE: if n_envs > train_freq, we take 1 step in each env, followed by multiple gradient steps
        # NOTE: will round up or down if not divisible evenly
    policy_delay: int = 2 # updates the policy once every `policy_delay` Q-function updates

    polyak_tau: Scheduleable[float] | None = None # if not None, OVERRIDES `target_update_interval``
        # target is updated after EVERY gradient step if not None

    exploration_noise: Scheduleable[float] = 0.1 # std
    target_noise: Scheduleable[float] = 0.2 # std
    target_noise_clip: Scheduleable[float] = 0.5

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")
TEnvAction = TypeVar("TEnvAction")
TTrunkOut = TypeVar("TTrunkOut")

class Networks(nnx.Module, Generic[TEnvObs, TEnvAction, TTrunkOut]):
    """Module containing networks for TD3.

    Observations are first processed using a shared `obs_trunk`.
        By default, `obs_trunk` only applies standardization and flattening, with no learnable parameters.

    The trunk output is then fed to the output heads: `policy_head`, `q1_head`, and `q2_head`.
        Additionally, an action is passed to `q1_head` and `q2_head` as a second argument. 
        Actions are given raw; the head is responsible for flattening.
    
    NOTE: Unlike in on-policy algorithms, in off-policy algorithms including TD3, the shared `obs_trunk` 
        is only updated during Q function updates; it is kept frozen during policy updates.
    """

    def __init__(self, 
        obs_trunk: Callable[[TEnvObs], TTrunkOut], 
        policy_head: Callable[[TTrunkOut], TEnvAction], 
        q1_head: Callable[[TTrunkOut, TEnvAction], jax.Array],
        q2_head: Callable[[TTrunkOut, TEnvAction], jax.Array],
    ) -> None:
        self.obs_trunk = obs_trunk
        self.policy_head = policy_head
        self.q1_head = q1_head
        self.q2_head = q2_head

    def __call__(self, obs: TEnvObs, action: TEnvAction,
            rngs: nnx.Rngs | None = None) -> tuple[TEnvAction, jax.Array]:
        trunk_out = optionally_pass(self.obs_trunk, rngs=rngs)(obs)

        out_action = optionally_pass(self.policy_head, rngs=rngs)(trunk_out)
        q1 = optionally_pass(self.q1_head, action, rngs=rngs)(trunk_out)
        q2 = optionally_pass(self.q2_head, action, rngs=rngs)(trunk_out)

        return out_action, q1, q2

    @classmethod
    def make_default(cls, rngs: nnx.Rngs, observation_space: Space[TEnvObs], action_space: Space[ArrayLike]) -> Self:
        return cls(
            cls.make_default_obs_trunk(observation_space),
            cls.make_default_policy_head(rngs, observation_space.flattened_dim, action_space),
            cls.make_default_q_head(rngs, observation_space.flattened_dim, action_space),
            cls.make_default_q_head(rngs, observation_space.flattened_dim, action_space)
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

        return Pipe(*layers)

    @staticmethod
    def make_default_policy_head(
        rngs: nnx.Rngs, input_dim: int, action_space: Space[ArrayLike],
        hidden_dims: Sequence[int] = (128, 128), do_layer_norm: bool = True, activation_func=nnx.tanh
    ) -> Callable[[TTrunkOut], TEnvAction]:
        mlp = MLP(
            rngs, (input_dim, *hidden_dims, action_space.flattened_dim), 
            do_layer_norm=do_layer_norm, activation_func=activation_func
        )

        return Pipe(mlp, action_space.unflatten, action_space.unsquash_continuous_from_bounds)    

    @staticmethod
    def make_default_q_head(
        rngs: nnx.Rngs, input_dim: int, action_space: Space[ArrayLike],
        hidden_dims: Sequence[int] = (128, 128), do_layer_norm: bool = True, activation_func=nnx.relu
    ) -> Callable[[TTrunkOut], jax.Array]:
        return Pipe(
            lambda trunk_out, action: jnp.concatenate((trunk_out, action_space.flatten(action)), axis=-1),
            MLP(
                rngs, (input_dim + action_space.flattened_dim, *hidden_dims, 1), 
                do_layer_norm=do_layer_norm, activation_func=activation_func
            ),
            lambda x: jnp.squeeze(x, axis=-1)
        )


@dataclass(frozen=True)
class ReplayTimestep(Generic[TEnvObs, TEnvAction]):
    obs: TEnvObs
    action: TEnvAction
    reward: ArrayLike
    terminated: ArrayLike

@dataclass
class TrainingState(Generic[TEnvState, TEnvObs, TEnvAction, TTrunkOut]):
    steps: ArrayLike
    env_states: TEnvState

    networks: Networks[TEnvObs, TEnvAction, TTrunkOut]
    policy_optimizer: nnx.Optimizer # policy updates only affect the policy head, NOT the shared trunk
    q_func_optimizer: nnx.Optimizer # q func updates affect both the q heads and the shared trunk

    target_networks: Networks[TEnvObs, TEnvAction, TTrunkOut]
    replay_buffer: CircularBufferWithOptionalData[ReplayTimestep[TEnvObs, TEnvAction], TEnvObs]

class TD3(Generic[TEnvState, TEnvObs, TEnvAction]):
    """Implementation of TD3."""

    def __init__(self, 
        env: Environment[TEnvState, TEnvObs, TEnvAction],
        hyperparameters: Hyperparameters = Hyperparameters()
    ) -> None:
        """IMPORTANT: `env` must already be batched; eg. wrap with `VmapWrapper` BEFORE passing in."""

        assert jax.tree.map(lambda s_dt: jnp.issubdtype(s_dt.dtype, jnp.floating), 
            env.action_space.shapes_dtypes), "Action space for TD3 must be continuous (jnp.floating)."

        self.env = env
        self.hyperparameters = hyperparameters

        # make replay buffer data shape
        self.replay_timestep_shapes_dtypes = ReplayTimestep(
            obs = self.env.observation_space.shapes_dtypes,
            action = self.env.action_space.shapes_dtypes,
            reward = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
            terminated = jax.ShapeDtypeStruct(shape=(), dtype=jnp.bool),
        )

    def make_optax_optimizer(self, 
        network_name: Literal['policy', 'q_func'],
        base: Callable[..., optax.GradientTransformation] = optax.adamw
    ) -> optax.GradientTransformationExtraArgs:
        optimizer_params = self.resolve_optimizer_params(network_name, 0)

        @optax.inject_hyperparams
        @override_signature(**optimizer_params)
        def make_optimizer(**kwargs):
            return with_grad_clip(base)(**kwargs)

        return make_optimizer(**optimizer_params)

    def resolve_optimizer_params(self, network_name: Literal['policy', 'q_func'], steps: int = 0) -> dict[str, Any]:
        additional_params = { 
            'policy': self.hyperparameters.policy_optimizer_params, 
            'q_func': self.hyperparameters.q_func_optimizer_params 
        }

        return jax.tree.map(lambda x: try_call(x, steps), {
            'learning_rate': self.hyperparameters.learning_rate,
            'max_grad_norm': self.hyperparameters.max_grad_norm,
            **additional_params[network_name]
        })

    def make_actor(self, 
        networks: Networks[TEnvObs, TTrunkOut] | None = None, 
        noise: ArrayLike = jnp.array(0.0), 
        rngs: nnx.Rngs | None = None
    ) -> DeterministicPolicyActor[TEnvObs, TEnvAction]:
        """`rngs` is only necessary if `networks` is not provided."""

        if networks is None: 
            networks = Networks.make_default(rngs, self.env.observation_space, self.env.action_space)

        return DeterministicPolicyActor(
            Pipe(networks.obs_trunk, networks.policy_head), 
            self.env.action_space,
            noise=noise
        )

    def init_training_state(self,
        rngs: nnx.Rngs,
        networks: Networks[TEnvObs, TEnvAction, TTrunkOut] | None = None,
        policy_optax_optimizer: optax.GradientTransformationExtraArgs | None = None,
        q_func_optax_optimizer: optax.GradientTransformationExtraArgs | None = None,
        replay_buffer: CircularBufferWithOptionalData[ReplayTimestep[TEnvObs, TEnvAction], TEnvObs] | None = None,
        prefill_steps: int = 10_000,
    ) -> TrainingState[TEnvState, TEnvObs, TEnvAction, TTrunkOut]:
        if networks is None: 
            networks = Networks.make_default(rngs, self.env.observation_space, self.env.action_space)

        optimizers = []

        for network_name, network, optax_optimizer in (
            ('policy', networks.policy_head, policy_optax_optimizer), 
            ('q_func', networks, q_func_optax_optimizer)
        ):
            if optax_optimizer is None: optax_optimizer = self.make_optax_optimizer(network_name)
            optimizer = nnx.Optimizer(network, optax_optimizer)

            assert hasattr(optimizer.opt_state, 'hyperparams'), \
                "`optax_optimizer` must be initialized using a `optax.inject_hyperparams()`-wrapped function."

            handled_keys = set(optimizer.opt_state.hyperparams)
            missing_keys = set(self.resolve_optimizer_params(network_name, 0)) - handled_keys
            assert not missing_keys, f"`optax_optimizer` missing hyperparams {missing_keys}; available: {handled_keys}."

            optimizers.append(optimizer)

        policy_optimizer, q_func_optimizer = optimizers

        target_networks = nnx.clone(networks)
        set_algo_phase(target_networks, AlgoPhase.EVAL)

        if self.hyperparameters.polyak_tau is not None: # convert datatypes to floats
            nnx.update(target_networks, optax.incremental_update(
                nnx.state(target_networks), nnx.state(target_networks), 0.5))

        if replay_buffer is None:
            replay_buffer = CircularBufferWithOptionalData.init(
                self.replay_timestep_shapes_dtypes, 
                self.env.observation_space.shapes_dtypes,
                int(self.hyperparameters.replay_buffer_size / self.hyperparameters.n_envs),
                optional_data_frac = self.hyperparameters.truncated_frac,
                batch_dims = self.hyperparameters.n_envs
            )

        # prefill replay buffer
        jitted_rollout = nnx.jit(self.rollout, static_argnames=('iter', 'actor'))
        timesteps, trunc, trunc_obs, env_states = jitted_rollout(rngs,
            RandomActor(self.env.action_space, self.env.observation_space),
            math.ceil(prefill_steps / self.hyperparameters.n_envs),
        )

        replay_buffer = replay_buffer.insert(timesteps, trunc, trunc_obs)

        return TrainingState(
            steps = jnp.array(0, dtype=jnp.int32),
            env_states = env_states,

            networks = networks,
            policy_optimizer = policy_optimizer,
            q_func_optimizer = q_func_optimizer,

            target_networks = target_networks,
            replay_buffer = replay_buffer,
        )

    def rollout(self,
        rngs: nnx.Rngs, 
        actor: Actor[TEnvObs, TEnvAction], 
        iter: int,
        initial_env_states: TEnvState | None = None,
    ) -> tuple[ReplayTimestep[TEnvObs, TEnvAction], jax.Array, TEnvObs, TEnvState]:
        """Collect a rollout of `ReplayTimestep`, truncated, truncated obs.

        Runs `n_envs` environments in parallel for `iter` steps each,
            for a total of `iter * n_envs` transitions.

        Initializes initial environment states if none given.
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

        replay_timesteps = ReplayTimestep(obs=timesteps.obs, action=timesteps.action, 
            reward=timesteps.reward, terminated=timesteps.terminated)

        return replay_timesteps, timesteps.truncated, next_obs, env_states
    
    def train(self, rngs: nnx.Rngs, training_state: TrainingState[TEnvState, TEnvObs, TEnvAction, TTrunkOut], steps: int) \
            -> tuple[TrainingState[TEnvState, TEnvObs, TEnvAction, TTrunkOut], dict[Any, Any]]:
        """Train from the given `training_state`, returning an updated `training_state` and metrics."""

        steps_per_env_per_iter = math.ceil(self.hyperparameters.train_freq / self.hyperparameters.n_envs)
        total_steps_per_iter = steps_per_env_per_iter * self.hyperparameters.n_envs
        learn_steps_per_iter = math.ceil(self.hyperparameters.n_envs / self.hyperparameters.train_freq)

        def train_iteration(training_state: TrainingState[TEnvState, TEnvObs, TEnvAction, TTrunkOut], rngs: nnx.Rngs) \
                -> tuple[TrainingState[TEnvState, TEnvObs, TEnvAction, TTrunkOut], dict[Any, Any]]:
            ## sample transitions from environment ##
            set_algo_phase(training_state.networks, AlgoPhase.ROLLOUT)

            actor = self.make_actor(training_state.networks, 
                noise=try_call(self.hyperparameters.exploration_noise, training_state.steps))
            timesteps, trunc, trunc_obs, training_state.env_states = self.rollout(rngs, 
                actor, steps_per_env_per_iter, training_state.env_states)
            training_state.steps += total_steps_per_iter

            training_state.replay_buffer = training_state.replay_buffer.insert(timesteps, trunc, trunc_obs)

            # update optimizer schedules using env steps (rather than default grad steps)
            for network_name, optimizer in (
                ('policy', training_state.policy_optimizer), 
                ('q_func', training_state.q_func_optimizer)
            ):
                optimizer_params = self.resolve_optimizer_params(network_name, training_state.steps)
                for key, new_val in optimizer_params.items():
                    optimizer.opt_state.hyperparams[key].value = new_val

            ## update q functions ##
            set_algo_phase(training_state.networks, AlgoPhase.OPTIMIZE)

            target_noise = try_call(self.hyperparameters.target_noise, steps)
            target_noise_clip = try_call(self.hyperparameters.target_noise_clip, steps)

            def learn_step(carry: tuple[Networks, Networks, nnx.Optimizer, nnx.Optimizer], rngs: nnx.Rngs) \
                    -> tuple[tuple[Networks, Networks, nnx.Optimizer, nnx.Optimizer], dict[Any, Any]]:
                networks, target_networks, policy_optimizer, q_func_optimizer = carry

                # sample replay buffer
                samp_timesteps, samp_trunc, samp_trunc_obs = training_state.replay_buffer.sample(
                    rngs.optimize_samples(), seq_len=2, batch_dims=self.hyperparameters.batch_size)

                first_timestep = jax.tree.map(lambda x: x[:, 0], samp_timesteps)
                next_obs = jax.tree.map(lambda main, trunc, sd: 
                        jnp.where(samp_trunc[(slice(None), 0) + (None,)*len(sd.shape)], trunc[:, 0], main[:, 1]), 
                    samp_timesteps.obs, samp_trunc_obs, self.env.observation_space.shapes_dtypes)

                # optimize networks
                target_trunk_out = optionally_pass(target_networks.obs_trunk, rngs=rngs)(next_obs)

                target_action = optionally_pass(target_networks.policy_head, rngs=rngs)(target_trunk_out)
                target_action = self.env.action_space.add_noise_to_continuous(rngs.actions(), target_action,
                    noise_std=target_noise, noise_clip=target_noise_clip)

                next_q = jnp.minimum(
                    optionally_pass(target_networks.q1_head, target_action, rngs=rngs)(target_trunk_out),
                    optionally_pass(target_networks.q2_head, target_action, rngs=rngs)(target_trunk_out)
                )

                # zero out q value if terminated
                next_q = next_q * jnp.logical_not(first_timestep.terminated)

                target_qs = first_timestep.reward \
                    + try_call(self.hyperparameters.discount_rate, training_state.steps)*next_q

                def q_loss_func(networks: nnx.Module, rngs: nnx.Rngs):
                    _, q1, q2 = optionally_pass(networks, rngs=rngs)(first_timestep.obs, first_timestep.action)
                    q_losses = [ jnp.mean(jnp.power(target_qs - q, 2)) for q in (q1, q2) ] # MSE loss

                    return sum(q_losses), { f'q1_loss': q_losses[0], 'q2_loss': q_losses[1] }

                q_loss_grad_func = nnx.grad(q_loss_func, has_aux=True)
                q_grads, metrics = q_loss_grad_func(networks, rngs)
                q_func_optimizer.update(q_grads) 

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

            metrics = jax.tree.map(lambda x: jnp.mean(x), metrics)
            metrics['steps'] = training_state.steps

            # update target if enough steps have passed (not using polyak averaging)
            if self.hyperparameters.polyak_tau is None:
                update_target = training_state.steps % self.hyperparameters.target_update_interval < total_steps_per_iter
                nnx.update(training_state.target_networks, jax.lax.cond(update_target, 
                    lambda opt_state, target_state: opt_state, 
                    lambda opt_state, target_state: target_state,
                    nnx.state(training_state.networks), nnx.state(training_state.target_networks)
                ))

            return training_state, metrics

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

        return training_state, metrics