from typing import Any, Generic
from typing_extensions import TypeVar
from abc import ABC, abstractmethod

import chex

import functools

import jax
from jax.typing import ArrayLike

import jax.numpy as jnp

from core.utils.math import inv_softplus

TSpaceElement = TypeVar("TSpaceElement")

class Space(Generic[TSpaceElement]):
    """Defines bounds and structure, such as for observations/actions. 

    Similar to gymnasium.Space, but more flexible as it allows elements to be arbitrary PyTrees,
        rather than requiring custom Tuple/Dict spaces.

    Integer bounds denote discrete values, while float bounds denote continuous values.
    Float bounds can be infinite to indicate an unbounded leaf.
        However, sampling is currently not supported for unbounded leaves.

    `low`: Lower bound, inclusive.
    `high`: Upper bound. Inclusive for integer type, exclusive for float type.

    `treedef`: Jax PyTree treedef for an element of the space. Can be used to unflatten elements.
    `shapes_dtypes`: PyTree of `jax.ShapeDtypeStruct`, storing the shape and dtype of each leaf.
    """

    def __init__(self, low: TSpaceElement, high: TSpaceElement) -> None:
        chex.assert_trees_all_equal_structs(low, high,
            custom_message="`low` and `high` must have the same treedef (structure).")

        self.treedef = jax.tree.structure(low)

        self.shapes_dtypes = jax.tree.map(lambda cur_low, cur_high: jax.ShapeDtypeStruct(
            dtype = jnp.result_type(cur_low, cur_high),
            shape = jnp.broadcast_shapes(cur_low.shape, cur_high.shape)
        ), low, high)

        self.low = jax.tree.map(
            lambda leaf, shape_dtype: jnp.broadcast_to(leaf, shape_dtype.shape).astype(shape_dtype.dtype),
            low, self.shapes_dtypes
        )

        self.high = jax.tree.map(
            lambda leaf, shape_dtype: jnp.broadcast_to(leaf, shape_dtype.shape).astype(shape_dtype.dtype),
            high, self.shapes_dtypes
        )

    def contains(self, x: TSpaceElement, batched: bool = False) -> bool:
        """Check if `x` is a valid member of this space. 
        If batched=True, disregards leading batch dimensions"""

        if jax.tree.structure(x) != self.treedef: return False

        def leaf_matches(x_leaf, cur_low, cur_high, shape_dtype):
            shape_compare_start = -len(shape_dtype.shape) if batched else 0
            if x_leaf.shape[shape_compare_start:] != shape_dtype.shape: return False

            if not bool(jnp.all(x_leaf >= cur_low)): return False

            if jnp.issubdtype(shape_dtype.dtype, jnp.integer):
                if not jnp.issubdtype(x_leaf.dtype, jnp.integer): return False
                if not bool(jnp.all(x_leaf <= cur_high)): return False
            else:
                if not jnp.issubdtype(x_leaf.dtype, jnp.floating): return False
                if not bool(jnp.all(x_leaf < cur_high)): return False

        return jax.tree.all(leaf_matches, x, self.low, self.high, self.shapes_dtypes)

    #@functools.partial(jax.jit, static_argnames=('self'))
    def sample(self, key: chex.PRNGKey) -> TSpaceElement:
        """Samples a single element from the space, according to a uniform distribution.
        
        Continuous (jnp.floating) items can be unbounded by setting 
        `low` to `-jnp.inf` and/or `high` to `jnp.inf`.
            - If one bound is infinite, samples from a standard exponential distribution.
            - If both bounds are infinite, samples from standard normal distribution."""

        keys = jax.random.split(key, num=self.treedef.num_leaves)
        keys_tree = jax.tree.unflatten(self.treedef, keys)

        def sample_leaf(cur_low, cur_high, shape_dtype: jax.ShapeDtypeStruct, key: chex.PRNGKey):

            if jnp.issubdtype(shape_dtype.dtype, jnp.integer):
                return jax.random.randint(key, shape=shape_dtype.shape, dtype=shape_dtype.dtype,
                    minval=cur_low, maxval=cur_high + 1)

            else:
                sample = jax.random.uniform(key, shape=shape_dtype.shape, dtype=shape_dtype.dtype,
                    minval=cur_low, maxval=cur_high)

                exp = jax.random.exponential(key, shape=shape_dtype.shape)
                sample = jnp.where(jnp.isinf(cur_low), cur_high - exp, sample)
                sample = jnp.where(jnp.isinf(cur_high), cur_low + exp, sample)

                sample = jnp.where(jnp.logical_and(jnp.isinf(cur_low), jnp.isinf(cur_high)), 
                    jax.random.normal(key, shape=shape_dtype.shape), sample)

                return sample

        return jax.tree.map(sample_leaf, self.low, self.high, self.shapes_dtypes, keys_tree)

    def squash_continuous_to_bounds(self, x: TSpaceElement) -> TSpaceElement:
        """Squashes unbounded real values (-inf, inf) to the bounds defined by the Space.
        Ignores discrete values."""

        def map_func(leaf, cur_low, cur_high, shape_dtype: jax.ShapeDtypeStruct):
            if jnp.issubdtype(shape_dtype.dtype, jnp.integer): return leaf

            # extra sanitization due to issue with jnp.where and NaNs for gradients
            safe_cur_low = jnp.where(jnp.isinf(cur_low), 1, cur_low)
            safe_cur_high = jnp.where(jnp.isinf(cur_high), 2, cur_high)

            leaf = jnp.where(jnp.isinf(cur_low), safe_cur_high - jax.nn.softplus(leaf), leaf)
            leaf = jnp.where(jnp.isinf(cur_high), safe_cur_low + jax.nn.softplus(leaf), leaf)

            leaf = jnp.where(jnp.logical_or(jnp.isinf(cur_low), jnp.isinf(cur_high)), 
                leaf,
                safe_cur_low + jnp.tanh(leaf)*(safe_cur_high - safe_cur_low)
            )

            return leaf

        return jax.tree.map(map_func, x, self.low, self.high, self.shapes_dtypes)

    def unsquash_continuous_from_bounds(self, x: TSpaceElement) -> TSpaceElement:
        """Undos the `space.squash_continuous_to_bounds(x)` method."""
        
        def map_func(leaf, cur_low, cur_high, shape_dtype: jax.ShapeDtypeStruct):
            if jnp.issubdtype(shape_dtype.dtype, jnp.integer): return leaf

            # extra sanitization due to issue with jnp.where and NaNs for gradients
            safe_cur_low = jnp.where(jnp.isinf(cur_low), 1, cur_low)
            safe_cur_high = jnp.where(jnp.isinf(cur_high), 2, cur_high)

            leaf = jnp.where(jnp.isinf(cur_low), 
                inv_softplus(jnp.where(jnp.isinf(cur_low), safe_cur_high - leaf, 10)), 
                leaf
            )

            leaf = jnp.where(jnp.isinf(cur_high), 
                inv_softplus(jnp.where(jnp.isinf(cur_high), leaf - safe_cur_low, 10)), 
                leaf
            )

            any_inf = jnp.logical_or(jnp.isinf(cur_low), jnp.isinf(cur_high))
            safe_ranges = jnp.where(any_inf, 1, safe_cur_high - safe_cur_low)
            leaf = jnp.where(any_inf, 
                leaf,
                jnp.arctanh(jnp.where(any_inf, 0.5, (leaf - safe_cur_low) / safe_ranges))
            )

            return leaf

        return jax.tree.map(map_func, x, self.low, self.high, self.shapes_dtypes)

    def sample_distribution(self, key: chex.PRNGKey, distribution: TSpaceElement, 
            ignore_continuous_bounds=False, deterministic=False) -> TSpaceElement:
        """Samples a single element from the space, according to the given distribution.
        
        The distribution should have the same treedef as the Space. Each leaf should have the same shape,
        but with an extra trailing axis as follows:

            - If discrete (jnp.integer): values will be sampled according to a categorical distribution.
                Trailing axis should have size equal to the maximum number of possible values 
                    for any feature in the Space in that leaf: `max(high - low + 1)`
                Values along trailing axis should be logits. The 0th logit should correspond to the
                    mininum possible value of that feature. Pad with zeros at the end as necessary.

            If continuous (jnp.floating): values will be sampled according to a normal distribution.
                Trailing axis should have size 2. The first element should be the mean, 
                    and the second should be the standard deviation.
                If bounded, values will be clamped using the softplus (one side bounded) 
                    or tanh (both sides bounded) function.

        `ignore_continuous_bounds`: If True, assumes all continuous values are unbounded.
            Useful, eg. for raw network outputs, before softplus or tanh.

        `deterministic`: If True, takes the mode.
            Discrete (jnp.integer) leaves: selects the choice with the highest probability.
            Continuous (jnp.floating) leaves: takes the mean of the distribution.
        """

        keys = jax.random.split(key, num=self.treedef.num_leaves)
        keys_tree = jax.tree.unflatten(self.treedef, keys)

        def sample_leaf(cur_dist, cur_low, shape_dtype: jax.ShapeDtypeStruct, key: chex.PRNGKey):
            if jnp.issubdtype(shape_dtype.dtype, jnp.integer):
                if deterministic: indices = jnp.argmax(cur_dist, axis=-1)
                else: indices = jax.random.categorical(key, cur_dist, axis=-1)

                return cur_low + indices
            else:
                if deterministic: z = 0
                else: z = jax.random.normal(key, shape=shape_dtype.shape, dtype=shape_dtype.dtype)

                return cur_dist[..., 0] + cur_dist[..., 1]*z

        sample = jax.tree.map(sample_leaf, distribution, self.low, self.shapes_dtypes, keys_tree)

        if not ignore_continuous_bounds:
            sample = self.squash_continuous_to_bounds(sample)

        return sample

    def log_probability(self, x: TSpaceElement, 
            distribution: TSpaceElement, ignore_continuous_bounds=False) -> ArrayLike:
        """Computes the log probability of sampling `x` from `distribution`, assuming features are independent.
        See `space.sample_distribution` for details on the structure of `distribution`.

        `ignore_continuous_bounds`: If True, assumes all continuous values are unbounded.
            Useful, eg. for raw network outputs, before softplus or tanh.
        """

        def leaf_log_probability(x_leaf, cur_dist, cur_low, shape_dtype: jax.ShapeDtypeStruct):

            if jnp.issubdtype(shape_dtype.dtype, jnp.integer):
                all_log_probs = jax.nn.log_softmax(cur_dist, axis=-1)

                dist_is = x_leaf - cur_low
                log_probs = jnp.take_along_axis(all_log_probs, dist_is[..., None], axis=-1).squeeze(axis=-1)
            else: 
                log_probs = jax.scipy.stats.norm.logpdf(x_leaf, loc=cur_dist[..., 0], scale=cur_dist[..., 1])

            return jnp.sum(log_probs)

        if not ignore_continuous_bounds:
            x = self.unsquash_continuous_from_bounds(x)

        log_probabilities = jax.tree.map(leaf_log_probability, x, distribution, self.low, self.shapes_dtypes)

        return jax.tree.reduce(lambda tot, cur: tot + cur, log_probabilities)

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")
TEnvAction = TypeVar("TEnvAction")
TRenderFrame = TypeVar("TRenderFrame", default=None)

class Environment(ABC, Generic[TEnvState, TEnvObs, TEnvAction, TRenderFrame]):
    """Abstract base class for environments."""

    @abstractmethod
    def reset(self, key: chex.PRNGKey) -> tuple[TEnvState, dict[Any, Any]]:
        """Performs resetting of environment.
        Returns: state, info"""

    @abstractmethod
    def step(self, key: chex.PRNGKey, state: TEnvState, action: TEnvAction) \
        -> tuple[TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        """Performs step transitions in the environment.
        Returns: state, reward, terminated, truncated, info"""

    @abstractmethod
    def get_obs(self, key: chex.PRNGKey, state: TEnvState) -> TEnvObs:
        """Applies observation function to state."""

    def render(self, state: TEnvState, action: ArrayLike) -> TRenderFrame:
        """Compute a render frame from the state-action pair.
        Intended for human interpretation (visualization, debugging); should not be used as a policy input.
        Implementations may or may not be jittable.
        Unimplemented by default, returning None."""
        return None

    @property
    @abstractmethod
    def observation_space(self) -> Space[TEnvObs]:
        """Observation space of the environment."""

    @property
    @abstractmethod
    def action_space(self) -> Space[TEnvAction]:
        """Action space of the environment."""

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__