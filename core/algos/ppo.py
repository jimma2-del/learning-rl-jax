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
from core.envs.wrappers import AutoResetWrapper, SquashContinuousActionsToBoundsWrapper
from core.envs.utils import rollout, Timestep

from core.sample_networks import MLP, MLPFeatureExtractor, StochasticPolicy

@dataclass(frozen=True)
class Hyperparameters:
    n_envs: int = 32

    discount_rate: Scheduleable[float] = 0.99
    learning_rate: Scheduleable[float] = 2.5e-4

    gae_lambda: Scheduleable[float] = 0.95

    rollout_length: int = 32 # steps per env per update (batch size is rollout_length * n_envs)
    n_minibatches: int = 8 # number of minibatches to split each batch into
        # minibatch size is rollout_length * n_envs / n_minibatches; must divide evenly
    n_epochs: int = 4 # number of full run throughs of the entire batch

    clip_epsilon: Scheduleable[float] = 0.25

    vf_coef: Scheduleable[float] = 0.5 # value function coefficient for the loss calculation
    ent_coef: Scheduleable[float] = 0.001 # conservative default; 0.01 to 0.001 (possibly schedule)

    ent_weight_continuous: Scheduleable[float] = 1
        # if using both discrete and continuous actions, it may be helpful to reduce the weight
            # of the continuous (differential) entropy, since it tends to have a higher scale
            # than discrete (Shannon's) entropy

    normalize_advantages: bool = True

    bootstrap_truncated: bool = False # if False, truncation is treated the same as termination.
        # if True, the critic is run an extra time for every environment sample
        # to compute next values; this is slightly slower.

    recompute_advantages: bool = False # if True, recomputes GAEs before every epoch.
        # see https://arxiv.org/pdf/2006.05990 (Appendix B.1)

    target_kl: Scheduleable[float] | None = None # if not None, stops further training epochs 
        # if the average approx_kl of the previous epoch exceeds this threshold
    
TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")
TEnvAction = TypeVar("TEnvAction")

class Networks(Generic[TEnvObs, TEnvAction], nnx.Module):
    def __init__(self, policy: nnx.Module, critic: nnx.Module) -> None:
        self.policy = policy
        self.critic = critic
            # NOTE: critic returns an array with 1 element instead of a scalar in current setup

@dataclass(frozen=True)
class TrainingState(Generic[TEnvState, TEnvObs]):
    steps: ArrayLike
    env_states: TEnvState
    policy: nnx.Module # to match standard api

    networks: Networks[TEnvObs, TEnvAction]
    optimizer: nnx.Optimizer

class PPO(Generic[TEnvState, TEnvObs]):
    """Implementation of PPO."""

    def __init__(self, 
        env: Environment[TEnvState, TEnvObs, TEnvAction],
        hyperparameters: Hyperparameters = Hyperparameters()
    ) -> None:
        """IMPORTANT: `env` must already be batched; eg. wrap with `VmapWrapper` BEFORE passing in."""
        self.env = env

        self.hyperparameters = hyperparameters
        assert hyperparameters.rollout_length*hyperparameters.n_envs % hyperparameters.n_minibatches == 0, \
            "Total rollout samples (`rollout_length * n_envs`) must be divisible by `n_minibatches`."

    def get_action(self, rngs: nnx.Rngs, policy: nnx.module, obs: TEnvObs, deterministic: bool = False,
            squash_continuous=True) -> TEnvAction:
        action_distribution = optionally_pass(policy, rngs=rngs)(obs)

        return self.env.action_space.sample_distribution(rngs.actions(), action_distribution, 
            squash_continuous=squash_continuous, log_stds=True, deterministic=deterministic)

    def create_default_networks(self, rngs: nnx.Rngs) -> tuple[nnx.Module, nnx.Module]:
        FEATURE_EXTRACTOR_OUTPUT_DIM = 256

        feature_extractor = MLPFeatureExtractor[TEnvObs](rngs, 
            self.env.observation_space.shapes_dtypes, output_dim=FEATURE_EXTRACTOR_OUTPUT_DIM)

        actor = nnx.Sequential(feature_extractor,
            StochasticPolicy(rngs, self.env.action_space, input_dim=FEATURE_EXTRACTOR_OUTPUT_DIM))

        feature_extractor = MLPFeatureExtractor[TEnvObs](rngs, 
            self.env.observation_space.shapes_dtypes, output_dim=FEATURE_EXTRACTOR_OUTPUT_DIM)

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

        networks = Networks(policy, critic)

        # shared optimizer
        optimizer = nnx.Optimizer(networks, optax.inject_hyperparams(optax.adamw)(
            learning_rate=try_call(self.hyperparameters.learning_rate, 0)))

        env_states, infos = self.env.reset(jax.random.split(rngs.env(), self.hyperparameters.n_envs))

        return TrainingState(
            steps = jnp.array(0, dtype=jnp.int32),
            env_states = env_states,

            policy = policy,
            networks = networks,
            optimizer = optimizer,
        )
    
    def train(self, rngs: nnx.Rngs, training_state: TrainingState[TEnvState, TEnvObs], steps: int) \
            -> tuple[TrainingState[TEnvState, TEnvObs], dict[Any, Any]]:
        """Train from the given `training_state`, returning an updated `training_state` and metrics."""

        total_steps_per_iter = self.hyperparameters.n_envs * self.hyperparameters.rollout_length

        def train_iteration(training_state: TrainingState[TEnvState, TEnvObs], rngs: nnx.Rngs) \
                -> tuple[TrainingState[TEnvState, TEnvObs], dict[Any, Any]]:
            env_states = training_state.env_states
            steps = training_state.steps

            networks = training_state.networks
            optimizer = training_state.optimizer
            
            ## sample transitions from environment ##
            def actor(obs: TEnvObs, rngs: nnx.Rngs) -> tuple[TEnvAction, dict[Any, Any]]:
                action_dist = optionally_pass(networks.policy, rngs=rngs)(obs)
                value = optionally_pass(networks.critic, rngs=rngs)(obs).squeeze(axis=-1)

                action = self.env.action_space.sample_distribution(rngs.actions(), action_dist, 
                    squash_continuous=False, log_stds=True)
                
                log_p = self.env.action_space.log_probability(action, action_dist, 
                    continuous_squashed=False, log_stds=True)

                return action, { 'value': value, 'log_p': log_p }

            (unreset_obs, timesteps), env_states, final_infos = rollout(
                rngs, SquashContinuousActionsToBoundsWrapper(self.env), actor,
                self.hyperparameters.rollout_length, self.hyperparameters.n_envs,
                env_states,

                take_func = lambda timesteps, rngs: (
                    self.env.get_obs(
                        jax.random.split(rngs.env(), self.hyperparameters.n_envs),
                        timesteps.info[AutoResetWrapper.UNRESET_STATE_INFO_KEY]
                    ), 
                    timesteps.replace(state=None, info=None) # remove unnecessary fields to save memory
                ),
                actor_returns_info=True
            )

            steps += total_steps_per_iter

            if not self.hyperparameters.bootstrap_truncated: # treat truncation as termination 
                timesteps.terminated = jnp.logical_or(timesteps.truncated, timesteps.terminated)

            timesteps.truncated = timesteps.truncated.at[-1].set(True)
                # last timesteps should be considered truncated, so bootstrapping is used

            # update optimizer schedules using env steps (rather than default grad steps)
            lr = try_call(self.hyperparameters.learning_rate, steps)
            optimizer.opt_state.hyperparams['learning_rate'].value = lr

            ## update policy ##
            discount = try_call(self.hyperparameters.discount_rate, steps)
            gae_lambda = try_call(self.hyperparameters.gae_lambda, steps)

            clip_epsilon = try_call(self.hyperparameters.clip_epsilon, steps)

            vf_coef = try_call(self.hyperparameters.vf_coef, steps)
            ent_coef = try_call(self.hyperparameters.ent_coef, steps)
            ent_weight_continuous = try_call(self.hyperparameters.ent_weight_continuous, steps)

            # compute advantage estimates and target values
            if self.hyperparameters.bootstrap_truncated:
                next_values = optionally_pass(networks.critic, rngs=rngs)(
                    jax.tree.map(lambda x: x[1:], unreset_obs)).squeeze(axis=-1)
            else:
                next_values = timesteps.action_info['value'][1:]

            final_obs = self.env.get_obs(
                jax.random.split(rngs.env(), self.hyperparameters.n_envs),
                final_infos[AutoResetWrapper.UNRESET_STATE_INFO_KEY]
            )

            final_values = optionally_pass(networks.critic, rngs=rngs)(final_obs).squeeze(axis=-1)
            next_values = jnp.append(next_values, final_values[None, ...], axis=0)

            def gae_iter(next_gae: jax.Array, timestep: Timestep[TEnvState, TEnvObs, TEnvAction],
                value: jax.Array, next_value: jax.Array):

                not_terminated = jnp.logical_not(timestep.terminated)
                not_truncated = jnp.logical_not(timestep.truncated)

                next_gae = next_gae * not_terminated * not_truncated
                td_err = -value + timestep.reward + discount*next_value*not_terminated

                gae = td_err + discount*gae_lambda*next_gae

                return gae, gae

            _, advantages = nnx.scan(gae_iter, in_axes=(nnx.Carry, 0, 0, 0), reverse=True)(
                jnp.zeros(self.hyperparameters.n_envs),
                timesteps, timesteps.action_info['value'], next_values
            )

            target_values = advantages + timesteps.action_info['value']

            train_samples = (timesteps.obs, timesteps.action, timesteps.action_info['log_p'], advantages, target_values)
            train_samples = jax.tree.map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), train_samples)

            # do epochs to update the networks
            def train_epoch(carry, rngs):
                shuffled_is = jax.random.permutation(rngs.learn(), len(train_samples[0]))
                minibatches = jax.tree.map(lambda x: 
                    jnp.reshape(x[shuffled_is], (self.hyperparameters.n_minibatches, -1, *x.shape[1:])), train_samples)

                def train_minibatch(carry, rngs, minibatch):
                    networks, optimizer = carry
                    obs, action, old_log_p, adv, target_values = minibatch

                    normed_adv = adv
                    if self.hyperparameters.normalize_advantages:
                        normed_adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)

                    def loss_func(networks: Networks[TEnvObs, TEnvAction], rngs: nnx.Rngs):
                        action_distribution = optionally_pass(networks.policy, rngs=rngs)(obs)

                        # policy loss
                        log_probabilities = self.env.action_space.log_probability(
                            action, action_distribution, continuous_squashed=False, log_stds=True)

                        log_ratio = log_probabilities - old_log_p
                        ratio = jnp.exp(log_ratio)
                        clipped_ratio = jnp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

                        policy_loss = - jnp.mean(jnp.minimum(ratio * normed_adv, clipped_ratio * normed_adv))

                        # value loss
                        pred_values = optionally_pass(networks.critic, rngs=rngs)(obs).squeeze(axis=-1)
                        value_loss = jnp.mean(jnp.power(target_values - pred_values, 2)) # MSE
                        
                        # entropy loss
                        feature_ents = self.env.action_space.entropies(action_distribution, 
                            log_stds=True, monte_carlo_n_samples=1, monte_carlo_key=rngs.actions())
                        scaled_feature_ents = jax.tree.map( # reduce continuous entropy weighting
                            lambda leaf, s_dt: 
                                (1 if jnp.issubdtype(s_dt.dtype, jnp.integer) else ent_weight_continuous) * leaf,
                            feature_ents, self.env.action_space.shapes_dtypes
                        )
                        comb_ents = jax.tree.reduce(lambda tot, cur: tot + cur, # sum entropies
                            jax.tree.map(lambda leaf, s_dt: jnp.sum(leaf, axis=tuple(range(-len(s_dt.shape), 0))),
                                scaled_feature_ents, self.env.action_space.shapes_dtypes))
                        mean_entropy = jnp.mean(comb_ents)

                        comb_loss = policy_loss + vf_coef*value_loss - ent_coef*mean_entropy

                        # extra metrics
                        approx_kl = jnp.mean(ratio - 1 - log_ratio) # http://joschu.net/blog/kl-approx.html
                            # monte carlo estimate of kl divergence; only used for metrics
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > clip_epsilon)
                            # fraction of data where the clipped objective was used instead of the regular

                        metrics = { 'loss': comb_loss, 'policy_loss': policy_loss, 'value_loss': value_loss, 
                            'entropy': mean_entropy, 'approx_kl': approx_kl, 'clip_frac': clip_frac }

                        return comb_loss, metrics

                    loss_grad_func = nnx.value_and_grad(loss_func, has_aux=True)
                    (comb_loss, metrics), grads = loss_grad_func(networks, rngs)
                    optimizer.update(grads)

                    return (networks, optimizer), metrics

                carry, metrics = nnx.scan(train_minibatch, in_axes=(nnx.Carry, 0, 0))(carry, 
                    rngs.fork(split=self.hyperparameters.n_minibatches), minibatches)

                return carry, metrics

            (networks, optimizer), metrics = nnx.scan(train_epoch)((networks, optimizer), 
                rngs.fork(split=self.hyperparameters.n_epochs))

            return TrainingState(
                steps=steps,
                env_states=env_states,

                policy=networks.policy,
                networks=networks,
                optimizer=optimizer,
            ), jax.tree.map(lambda x: jnp.mean(x), metrics)

        iterations = math.ceil(steps / total_steps_per_iter)
        training_state, metrics = nnx.scan(train_iteration)(training_state, rngs.fork(split=iterations))

        return training_state, jax.tree.map(lambda x: jnp.mean(x), metrics)