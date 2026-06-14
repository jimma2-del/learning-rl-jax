from typing import Any, Generic, Callable
from typing_extensions import TypeVar
from abc import ABC, abstractmethod

import chex

import functools

import jax
from jax.typing import ArrayLike

import jax.numpy as jnp
import numpy as np

from .base import Environment, Space
from core.utils.batch_utils import get_tree_vmap_dim, dummy_vmap

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")
TEnvAction = TypeVar("TEnvAction")
TRenderFrame = TypeVar("TRenderFrame", default=None)

class Wrapper(
    Generic[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
    Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame]
):
    """Abstract base class for wrappers, which modify environment attributes in some way.
    
    Wrappers should generally be written to support batched environments,
        though users may forego this in their custom wrappers.
    """

    def __init__(self, env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame]):
        self.env = env

    @property
    def unwrapped(self) -> Environment: # there is no way to type this properly
        """Get the underlying Environment, without any wrappers."""

        if isinstance(self.env, Wrapper):
            return self.env.unwrapped

        return self.env

    # forwards all Environment methods/properties to the internal env by default

    def reset(self, key: chex.PRNGKey) -> tuple[TEnvState, dict[Any, Any]]:
        return self.env.reset(key)

    def step(self, key: chex.PRNGKey, state: TEnvState, action: TEnvAction) \
        -> tuple[TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        return self.env.step(key, state, action)

    def get_obs(self, key: chex.PRNGKey, state: TEnvState) -> TEnvObs:
        return self.env.get_obs(key, state)

    def render(self, state: TEnvState, action: ArrayLike) -> TRenderFrame:
        return self.env.render(state, action)

    @property
    def observation_space(self) -> Space[TEnvObs]:
        return self.env.observation_space

    @property
    def action_space(self) -> Space[TEnvAction]:
        return self.env.action_space

    @property
    def name(self) -> str:
        """Environment name."""
        return self.env.name

    def __getattr__(self, name):
        return getattr(self.env, name)

TWrapperType = TypeVar('TWrapperType', bound=Wrapper)

def find_wrapper(
    wrapper_type: type[TWrapperType], 
    env: Environment | Wrapper
) -> TWrapperType | None:

    while isinstance(env, Wrapper):
        if isinstance(env, wrapper_type):
            return env
            
        env = env.env

class ObsRangeNormalizeWrapper(
    Generic[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
    Wrapper[TEnvState, TEnvObs, TEnvAction, TRenderFrame]
):
    """Translates and scales observation features to normalize.

    Features which had both sides bounded are normalized to the range
        [-1, 1] if (previously) discrete and [-1, 1) if continuous.
    Features which had one side bounded are normalized to the range [0, inf).

    Ignores features with both sides unbounded (-inf, inf), keeping the values unaltered.
        NOTE: Keep in mind when using this wrapper that unbounded observation features
        are very common; this wrapper may not be suitable for many environments.

    NOTE: May convert discrete (np.integer) data types to continuous (np.floating).
    """

    def __init__(self, 
        env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
        normalize_obs_space: Space[TEnvObs] | None = None
    ):
        """`normalize_obs_space`: If provided, normalizes based on bounds in this instead of `env.observation_space`."""

        super().__init__(env)

        self.normalize_obs_space: Space[TEnvObs] = normalize_obs_space \
            if normalize_obs_space is not None else super().observation_space

        chex.assert_trees_all_equal_structs(env.observation_space.low, self.normalize_obs_space.low,
            "`normalize_obs_space` treedef does not match with `env.observation_space`.")

        def handle_leaf(cur_low, cur_high):
            # handle both NOT unbounded
            translate = - (cur_low + cur_high) / 2
            scale = 1 / ((cur_high + cur_low) / 2)

            # only one side unbounded
            translate = jnp.where(np.isinf(cur_low), -cur_high, translate)
            scale = jnp.where(np.isinf(cur_low), -1, scale)
            
            translate = jnp.where(np.isinf(cur_high), -cur_low, translate)

            # handle both unbounded -> no-op, return original
            both_unbounded = np.logical_and(np.isinf(cur_low), np.isinf(cur_high))
            translate = jnp.where(both_unbounded, np.zeros_like(cur_low), translate)
            scale = jnp.where(both_unbounded, np.ones_like(cur_low), scale)

            return translate, scale

        self.translate, self.scale = jax.tree.map(handle_leaf, 
            self.normalize_obs_space.low, self.normalize_obs_space.high)

    def get_obs(self, key: chex.PRNGKey, state: TEnvState) -> TEnvObs:
        return jax.tree.map(lambda obs, translate, scale: (obs + translate) * scale, 
            super().get_obs(key, state), self.translate, self.scale)

    @property
    def observation_space(self) -> Space[TEnvObs]:
        def handle_leaf(cur_low, cur_high):
            # handle both NOT unbounded
            low = -np.ones_like(cur_low)
            high = np.ones_like(cur_high)

            # only one side unbounded
            either_unbounded = np.logical_or(np.isinf(cur_low), np.isinf(cur_high))
            low = jnp.where(either_unbounded, np.zeros_like(cur_low), low)
            high = jnp.where(either_unbounded, np.full_like(cur_high, np.inf), high)

            # handle both unbounded 
            both_unbounded = np.logical_and(np.isinf(cur_low), np.isinf(cur_high))
            low = jnp.where(both_unbounded, np.full_like(cur_high, -np.inf), low)
            high = jnp.where(both_unbounded, np.full_like(cur_high, np.inf), high)

            return low, high

        low, high = jax.tree.map(handle_leaf, 
            self.normalize_obs_space.low, self.normalize_obs_space.high)

        return Space(low=low, high=high)

class ActionsRangeNormalizeWrapper(
    Generic[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
    Wrapper[TEnvState, TEnvObs, TEnvAction, TRenderFrame]
):
    """Translates and scales action features to normalize.

    Translates discrete (integer) features to make low=0.

    For continuous (floating) features:
        Features which had both sides bounded are normalized to the range [-1, 1).
        Features which had one side bounded are normalized to the range [0, inf).

        Ignores features with both sides unbounded (-inf, inf), keeping the values unaltered.
    """

    def __init__(self, 
        env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
        normalize_actions_space: Space[TEnvObs] | None = None
    ):
        """`normalize_actions_space`: If provided, normalizes based on bounds in this instead of `env.action_space`."""
        super().__init__(env)

        self.normalize_actions_space: Space[TEnvObs] = normalize_actions_space \
            if normalize_actions_space is not None else super().action_space

        chex.assert_trees_all_equal_structs(env.action_space.low, self.normalize_actions_space.low,
            "`normalize_actions_space` treedef does not match with `env.action_space`.")

        def handle_leaf(cur_low, cur_high, shape_dtype):

            if np.issubdtype(shape_dtype.dtype, np.integer):
                return cur_low, np.ones(shape_dtype.shape)

            else: # continuous
                # handle both NOT unbounded
                translate = (cur_low + cur_high) / 2
                scale = ((cur_high + cur_low) / 2)

                # only one side unbounded
                translate = jnp.where(np.isinf(cur_low), cur_high, translate)
                scale = jnp.where(np.isinf(cur_low), -1, scale)
                
                translate = jnp.where(np.isinf(cur_high), cur_low, translate)

                # handle both unbounded -> no-op, return original
                both_unbounded = np.logical_and(np.isinf(cur_low), np.isinf(cur_high))
                translate = jnp.where(both_unbounded, np.zeros_like(cur_low), translate)
                scale = jnp.where(both_unbounded, np.ones_like(cur_low), scale)

                return translate, scale

        self.translate, self.scale = jax.tree.map(handle_leaf, 
            self.normalize_actions_space.low, self.normalize_actions_space.high, self.normalize_actions_space.shapes_dtypes)

    def step(self, key: chex.PRNGKey, state: TEnvState, action: TEnvAction) \
            -> tuple[TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:

        action = jax.tree.map(lambda obs, translate, scale: obs*scale + translate, 
            action, self.translate, self.scale)
        
        return super().step(key, state, action)

    @property
    def action_space(self) -> Space[TEnvObs]:
        def handle_leaf(cur_low, cur_high, shape_dtype):
            if np.issubdtype(shape_dtype.dtype, np.integer):
                return np.zeros(shape_dtype.shape), cur_high - cur_low + 1

            else: # continuous
                # handle both NOT unbounded
                low = -np.ones_like(cur_low)
                high = np.ones_like(cur_high)

                # only one side unbounded
                either_unbounded = np.logical_or(np.isinf(cur_low), np.isinf(cur_high))
                low = jnp.where(either_unbounded, np.zeros_like(cur_low), low)
                high = jnp.where(either_unbounded, np.full_like(cur_high, np.inf), high)

                # handle both unbounded 
                both_unbounded = np.logical_and(np.isinf(cur_low), np.isinf(cur_high))
                low = jnp.where(both_unbounded, np.full_like(cur_high, -np.inf), low)
                high = jnp.where(both_unbounded, np.full_like(cur_high, np.inf), high)

                return low, high

        low, high = jax.tree.map(handle_leaf, 
            self.normalize_actions_space.low, self.normalize_actions_space.high, self.normalize_actions_space.shapes_dtypes)

        return Space(low=low, high=high)

class JitWrapper(
    Generic[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
    Wrapper[TEnvState, TEnvObs, TEnvAction, TRenderFrame]
):
    """JITs the `reset`, `step`, and `get_obs` methods.
    Does not alter `observation_space` or `action_space`.
    Does not alter the `render` method as it may not be jittable."""

    def reset(self, key: chex.PRNGKey) -> tuple[TEnvState, dict[Any, Any]]:
        return jax.jit(super().reset)(key)

    def step(self, key: chex.PRNGKey, state: TEnvState, action: TEnvAction) \
            -> tuple[TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        return jax.jit(super().step)(key, state, action)

    def get_obs(self, key: chex.PRNGKey, state: TEnvState) -> TEnvObs:
        return jax.jit(super().get_obs)(key, state)

class VmapWrapper(
    Generic[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
    Wrapper[TEnvState, TEnvObs, TEnvAction, TRenderFrame]
):
    """Vmaps the `reset`, `step`, and `get_obs` methods.
    Does not alter `observation_space` or `action_space` as the batch size is unknown.
    Does not alter the `render` method as it may not be jittable.
    """

    def reset(self, key: chex.PRNGKey) -> tuple[TEnvState, dict[Any, Any]]:
        return jax.vmap(super().reset)(key)

    def step(self, key: chex.PRNGKey, state: TEnvState, action: TEnvAction) \
            -> tuple[TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        return jax.vmap(super().step)(key, state, action)

    def get_obs(self, key: chex.PRNGKey, state: TEnvState) -> TEnvObs:
        return jax.vmap(super().get_obs)(key, state)

class DummyVmapWrapper(
    Generic[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
    Wrapper[TEnvState, TEnvObs, TEnvAction, TRenderFrame]
):
    """Mimics the behavior of `VmapWrapper()`, but does not actually apply vmap transformation; instead, 
        for inputs, simply takes the first element in the batch axis,
        and for outputs, adds a dummy batch axis of length 1."""

    def reset(self, key: chex.PRNGKey) -> tuple[TEnvState, dict[Any, Any]]:
        return dummy_vmap(super().reset)(key)

    def step(self, key: chex.PRNGKey, state: TEnvState, action: TEnvAction) \
            -> tuple[TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        return dummy_vmap(super().reset)(key, state, action)

    def get_obs(self, key: chex.PRNGKey, state: TEnvState) -> TEnvObs:
        return dummy_vmap(super().reset)(key, state)

class AutoResetWrapper(
    Generic[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
    Wrapper[TEnvState, TEnvObs, TEnvAction, TRenderFrame]
):
    """Automatically resets the environment if terminated or truncated, returning the resetted state as the next state
    Places the original, unresetted state into `info[UNRESET_STATE_INFO_KEY]` (useful eg. for truncation)
        This will be the same as the returned new_state if not terminated and not truncated.

    NOTE: If vectorizing the environment, apply BEFORE this wrapper:
        eg. do `AutoResetWrapper(VmapWrapper(env))`; do NOT do `VmapWrapper(AutoResetWrapper(env))`.

    If computing environment resets is expensive, for vectorization, it may be beneficial to
        wrap the environment with `PrecomputedResetsPoolWrapper()` before passing it to this wrapper.
    """

    UNRESET_STATE_INFO_KEY = 'unreset_state'

    def reset(self, key: chex.PRNGKey) -> tuple[TEnvState, dict[Any, Any]]:
        state, info = super().reset(key)
        info[self.UNRESET_STATE_INFO_KEY] = state
        return state, info

    def step(self, key: chex.PRNGKey, state: TEnvState, action: TEnvAction) \
            -> tuple[TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        if jnp.isscalar(key): step_key, reset_key = jax.random.split(key)
        else: step_key, reset_key = jax.vmap(jax.random.split)(key).T

        next_state, reward, terminated, truncated, info = super().step(step_key, state, action)
        info[self.UNRESET_STATE_INFO_KEY] = next_state

        done = jnp.logical_or(terminated, truncated)

        new_state = jax.lax.cond(done.any(), # cond to avoid computing resets when no envs done
            lambda: jax.tree.map(lambda reset, next: 
                    jnp.where(done[(slice(None),) + (None,)*(next.ndim - 1)], reset, next), 
                self.env.reset(reset_key)[0], next_state), 
            lambda: next_state
        )

        return new_state, reward, terminated, truncated, info

class PrecomputedResetsPoolWrapper(
    Generic[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
    Wrapper[TEnvState, TEnvObs, TEnvAction, TRenderFrame]
):
    """Returns a random choice from a pool of precomputed reset states and infos when calling `env.reset()`,
        rather than computing new reset states on the fly.
    Useful if computing environment resets is expensive, especially when using with `AutoResetWrapper()`.

    NOTE: If vectorizing the environment, apply BEFORE this wrapper:
        eg. do `PrecomputedResetsPoolWrapper(VmapWrapper(env))`; 
            do NOT do `VmapWrapper(PrecomputedResetsPoolWrapper(env))`.
    """

    def __init__(self, env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame], 
            resets_pool_state_infos: tuple[TEnvState, dict[Any, Any]] | None = None) -> None:
        super().__init__(env)
        self.resets_pool_state_infos = resets_pool_state_infos

    def reset(self, key: chex.PRNGKey) -> tuple[TEnvState, dict[Any, Any]]:
        resets_pool_size = get_tree_vmap_dim(self.resets_pool_state_infos)

        get_reset_is = lambda key: jax.random.randint(key, (), minval=0, maxval=resets_pool_size)
        if not jnp.isscalar(key): get_reset_is = jax.vmap(get_reset_is) # if env vmapped

        reset_is = get_reset_is(key)
        return jax.tree.map(lambda x: x[reset_is], self.resets_pool_state_infos)

class SquashContinuousActionsToBoundsWrapper(
    Generic[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
    Wrapper[TEnvState, TEnvObs, TEnvAction, TRenderFrame]
):
    """Squashes unbounded real values (-inf, inf) in actions to the bounds defined by the action space.
    Ignores discrete values.

    See `Space.squash_continuous_to_bounds(x)`.
    """

    def __init__(self, 
        env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
        normalize_actions_space: Space[TEnvObs] | None = None
    ):
        """`normalize_actions_space`: If provided, normalizes based on bounds in this instead of `env.action_space`."""
        super().__init__(env)

        self.normalize_actions_space: Space[TEnvObs] = normalize_actions_space \
            if normalize_actions_space is not None else super().action_space

        chex.assert_trees_all_equal_structs(env.action_space.low, self.normalize_actions_space.low,
            "`normalize_actions_space` treedef does not match with `env.action_space`.")

    def step(self, key: chex.PRNGKey, state: TEnvState, action: TEnvAction) \
            -> tuple[TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        action = self.normalize_actions_space.squash_continuous_to_bounds(action)
        return super().step(key, state, action)

    @property
    def action_space(self) -> Space[TEnvAction]:
        return Space(
            low = jax.tree.map(lambda leaf: leaf if np.issubdtype(leaf, np.integer) else np.full_like(leaf, -np.inf), 
                self.normalize_actions_space.low),
            high = jax.tree.map(lambda leaf: leaf if np.issubdtype(leaf, np.integer) else np.full_like(leaf, np.inf),
                self.normalize_actions_space.high)
        )

class ClipActionsToBoundsWrapper(
    Generic[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
    Wrapper[TEnvState, TEnvObs, TEnvAction, TRenderFrame]
):
    """Clips values in actions to the bounds defined by the action space.
        Values above or below bounds will be set to the bounds.

    The new action space will have (-inf, inf) bounds for all continuous values.
        Discrete bounds will be left unchanged because there is no way to mark unbounded discrete bounds,
        but any integer will be a valid input.

    NOTE: Spaces are typically not inclusive of `high` for continuous (floating) features. However,
        This wrapper will clip values above `high` to exactly `high`.
    """

    def __init__(self, 
        env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
        normalize_actions_space: Space[TEnvObs] | None = None
    ):
        """`normalize_actions_space`: If provided, normalizes based on bounds in this instead of `env.action_space`."""
        super().__init__(env)

        self.normalize_actions_space: Space[TEnvObs] = normalize_actions_space \
            if normalize_actions_space is not None else super().action_space

        chex.assert_trees_all_equal_structs(env.action_space.low, self.normalize_actions_space.low,
            "`normalize_actions_space` treedef does not match with `env.action_space`.")

    def step(self, key: chex.PRNGKey, state: TEnvState, action: TEnvAction) \
            -> tuple[TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        action = jax.tree.map(lambda x, low, high: jnp.clip(x, low, high), 
            action, self.normalize_actions_space.low, self.normalize_actions_space.high)
        return super().step(key, state, action)

    @property
    def action_space(self) -> Space[TEnvAction]:
        return Space(
            low = jax.tree.map(lambda leaf: leaf if np.issubdtype(leaf, np.integer) else np.full_like(leaf, -np.inf), 
                self.normalize_actions_space.low),
            high = jax.tree.map(lambda leaf: leaf if np.issubdtype(leaf, np.integer) else np.full_like(leaf, np.inf),
                self.normalize_actions_space.high)
        )

@chex.dataclass
class EpisodeStepCountState(Generic[TEnvState]):
    state: TEnvState
    episode_steps: jax.Array

class EpisodeStepCountWrapper(
    Generic[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
    Wrapper[EpisodeStepCountState[TEnvState], TEnvObs, TEnvAction, TRenderFrame]
):
    """Stores the step count of the current episode in `info[STEP_COUNT_INFO_KEY]`,
    and optionally truncates episodes upon reaching a maximum length."""

    STEP_COUNT_INFO_KEY = 'episode_steps'

    def __init__(self,
        env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
        max_eps_len: int | None = None
    ) -> None:
        super().__init__(env)
        self.max_eps_len = max_eps_len

    def reset(self, key: chex.PRNGKey) -> tuple[EpisodeStepCountState[TEnvState], dict[Any, Any]]:
        state, info = super().reset(key)

        steps = jnp.zeros_like(key, dtype=jnp.int32)

        info[self.STEP_COUNT_INFO_KEY] = steps
        return EpisodeStepCountState(state=state, episode_steps=steps), info

    def step(self, key: chex.PRNGKey, state: EpisodeStepCountState[TEnvState], action: TEnvAction) \
            -> tuple[EpisodeStepCountState[TEnvState], jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        next_state, reward, terminated, truncated, info = super().step(key, state.state, action)

        steps = state.episode_steps + 1
        info[self.STEP_COUNT_INFO_KEY] = steps

        if self.max_eps_len is not None:
            truncated = jnp.logical_or(
                truncated,
                jnp.logical_and(
                    jnp.logical_not(terminated), # don't truncate if already terminated
                    steps >= self.max_eps_len
                )
            )

        return EpisodeStepCountState(state=next_state, episode_steps=steps), reward, terminated, truncated, info

    def get_obs(self, key: chex.PRNGKey, state: EpisodeStepCountState[TEnvState]) -> TEnvObs:
        return super().get_obs(key, state.state)

    def render(self, state: EpisodeStepCountState[TEnvState], action: ArrayLike) -> TRenderFrame:
        return super().render(state.state, action)

