import math

import jax.numpy as jnp
import jax

from jax.typing import ArrayLike
from chex import dataclass
from typing import TypeVar, Generic, Any

import functools

from flax import nnx
import optax

from core.algos.base import Scheduleable, resolve_scheduleable

from core.envs.base import Environment
from core.envs.wrappers import AutoResetWrapper
from core.envs.utils import parallel_rollout, Timestep

from core.sample_networks import MLP, MLPFeatureExtractor

@dataclass(frozen=True)
class Hyperparameters:
    n_envs: int = 32

    discount_rate: Scheduleable[float] = 0.99
    learning_rate: Scheduleable[float] = 1e-3
    
    n_steps: int = 5 # steps per env per update (batch size is n_steps * n_envs)

    vf_coef: Scheduleable[float] = 0.5 # value function coefficient for the loss calculation
    ent_coef: Scheduleable[float] = 0.001 # entropy coefficient for the loss calculation
        # 0.01 to 0.001; possibly schedule
    
    gae_lambda: Scheduleable[float] = 0.95

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")

@dataclass(frozen=True)
class TrainingState(Generic[TEnvState, TEnvObs]):
    steps: ArrayLike
    env_states: TEnvState

    policy: nnx.Module 
    critic: nnx.Module 
    optimizer: nnx.Optimizer

class A2C(Generic[TEnvState, TEnvObs]):
    """Implementation of A2C."""

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

    def get_action(self, rngs: nnx.Rngs, policy: nnx.module, obs: TEnvObs, deterministic: bool = False) -> ArrayLike:
        logits = policy(obs, rngs=rngs)

        if deterministic:
            return jnp.argmax(logits)

        return jax.random.categorical(rngs.actions(), logits)

    def create_default_networks(self, rngs: nnx.Rngs) -> tuple[nnx.Module, nnx.Module]:
        FEATURE_EXTRACTOR_OUTPUT_DIM = 256

        feature_extractor = MLPFeatureExtractor[TEnvObs](rngs, 
            self.env.observation_space.shapes_dtypes, output_dim=FEATURE_EXTRACTOR_OUTPUT_DIM),

        actor = nnx.Sequential(feature_extractor,
            MLP(rngs, input_dim=FEATURE_EXTRACTOR_OUTPUT_DIM, output_dim=self.num_actions))

        critic = nnx.Sequential(feature_extractor,
            MLP(rngs, input_dim=FEATURE_EXTRACTOR_OUTPUT_DIM, output_dim=1))

        return actor, critic

    def create_default_policy(self, rngs: nnx.Rngs) -> nnx.Module:
        actor, _ = self.create_default_networks(rngs)
        return actor

    def create_default_critic(self, rngs: nnx.Rngs) -> nnx.Module:
        _, critic = self.create_default_networks(rngs)
        return critic

    def init_training_state(self,
        rngs: nnx.Rngs,
        policy: nnx.Module | None = None,
        critic: nnx.Module | None = None,
    ) -> TrainingState[TEnvState, TEnvObs]:

        # create default networks if none given
        if policy is None and critic is None:
            policy, critic = self.create_default_networks(rngs)
        else:
            if policy is None:
                policy = self.create_default_policy(rngs)
            if critic is None:
                critic = self.create_default_critic(rngs)

        # shared optimizer for both actor & critic (loss is combined)
        optimizer = nnx.Optimizer((policy, critic), optax.inject_hyperparams(optax.adamw)(
            learning_rate=resolve_scheduleable(self.hyperparameters.learning_rate, 0)))
        #optimizer = nnx.Optimizer(q_net, optax.adamw(learning_rate=2.5e-4))

        env_states = jax.vmap(self.env.reset)(
            jax.random.split(rngs.env(), iter * self.hyperparameters.n_envs))

        return TrainingState(
            steps = jnp.array(0, dtype=jnp.int32),
            env_states = env_states,

            policy = policy,
            target = nnx.clone(policy),
            optimizer = optimizer
        )
    
    @functools.partial(nnx.jit, static_argnames=('self', 'epoch_steps'))
    def train_epoch(self, 
        rngs: nnx.Rngs,
        training_state: TrainingState[TEnvState, TEnvObs],
        epoch_steps: int,
        bootstrap_truncated: bool = False
    ) -> tuple[TrainingState[TEnvState, TEnvObs], dict[Any, Any]]:
        """Train for one 'epoch' -- one fully JIT compiled segment.
        
        `bootstrap_truncated`: If False, truncation is treated the same as termination.
            If True, the critic is run an extra time for every environment sample
                to compute next values; this is slightly slower.
        """

        total_steps_per_iter = self.hyperparameters.n_envs * self.hyperparameters.n_steps

        def train_iteration(training_state: TrainingState[TEnvState, TEnvObs], rngs: nnx.Rngs) \
                -> tuple[TrainingState[TEnvState, TEnvObs], dict[Any, Any]]:
            env_states = training_state.env_states
            steps = training_state.steps

            policy = training_state.policy
            critic = training_state.critic
            optimizer = training_state.optimizer    
            
            ## sample transitions from environment ##

            timesteps, env_states = parallel_rollout(
                rngs, self.env,
                nnx.vmap(lambda rngs, obs: self.get_action(rngs, policy, obs)),
                self.hyperparameters.n_steps, self.hyperparameters.n_envs,
                env_states
            )

            steps += total_steps_per_iter

            # update optimizer schedules using env steps (rather than default grad steps)
            optimizer.opt_state.hyperparams['learning_rate'].value \
                = resolve_scheduleable(self.hyperparameters.learning_rate, steps)

            ## update policy ##
            discount = resolve_scheduleable(self.hyperparameters.discount_rate, steps)
            gae_lambda = resolve_scheduleable(self.hyperparameters.gae_lambda, steps)

            def loss_func(networks: tuple[nnx.Module, nnx.Module], rngs: nnx.Rngs):
                policy, critic = networks

                values = critic(timesteps.obs, rngs=rngs)

                if bootstrap_truncated:
                    next_values = critic(timesteps.info[AutoResetWrapper.NEXT_STATE_INFO_KEY], rngs=rngs)
                else:
                    final_obs = jax.vmap(self.env.get_obs)(
                        jax.random.split(rngs.env(), self.hyperparameters.n_envs),
                        timesteps[-1].info[AutoResetWrapper.NEXT_STATE_INFO_KEY]
                    )

                    next_values = jnp.append(values[1:], final_obs, axis=0)

                timesteps.truncated = timesteps.truncated.at[-1].set(True)
                    # last timesteps should be considered truncated, so bootstrapping is used

                def gae_iter(next_gae: jax.Array, timestep: Timestep[TEnvState, TEnvObs, ArrayLike],
                    value: jax.Array, next_value: jax.Array):

                    not_terminated = jnp.logical_not(timestep.terminated)
                    not_truncated = jnp.logical_not(timestep.truncated)

                    next_gae = next_gae * not_terminated * not_truncated
                    td_err = -value + timestep.reward + discount*next_value*not_terminated

                    gae = td_err + discount*gae_lambda*next_gae

                    return gae, gae

                _, target_values = nnx.scan(gae_iter, in_axes=(nnx.Carry, 0, 0, 0), reverse=True)(
                    jnp.zeros(self.hyperparameters.n_envs),
                    timesteps, values, next_values
                )



                # simple MSE loss
                return jnp.mean(jnp.power(target_qs - pred_qs, 2))

            loss_grad_func = nnx.value_and_grad(loss_func, has_aux=True)
            (comb_loss, metrics), grads = loss_grad_func(policy, rngs)
            optimizer.update(grads) 

            return TrainingState(
                steps=steps,
                env_states=env_states,

                policy=policy,
                critic=critic,
                optimizer=optimizer
            ), jax.tree.map(lambda x: jnp.mean(x), metrics)

        iterations = math.ceil(epoch_steps / total_steps_per_iter)
        training_state, metrics = nnx.scan(train_iteration)(training_state, rngs.fork(split=iterations))

        return training_state, jax.tree.map(lambda x: jnp.mean(x), metrics)