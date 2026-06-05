from typing import Any, Generic
from typing_extensions import TypeVar
from abc import ABC, abstractmethod

import chex

import functools

import jax
from jax.typing import ArrayLike

import jax.numpy as jnp

from core.utils.math import inv_softplus, normal_entropy
from core.utils.batch_utils import get_tree_batch_dims

TSpaceElement = TypeVar("TSpaceElement")

class Space(Generic[TSpaceElement]):
    """Defines bounds and structure, such as for observations/actions. 

    Similar to gymnasium.Space, but more flexible as it allows elements to be arbitrary PyTrees,
        rather than requiring custom Tuple/Dict spaces.

    Integer bounds denote discrete values, while float bounds denote continuous values.
    Float bounds can be infinite to indicate an unbounded leaf.

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

        # assert low != +inf and high != -inf
        assert jax.tree.all(jax.tree.map(lambda cur_low: jnp.all(jnp.logical_not(jnp.isposinf(cur_low))), self.low)), \
            "`low` values cannot be positive infinity (`+jnp.inf`)."
        assert jax.tree.all(jax.tree.map(lambda cur_high: jnp.all(jnp.logical_not(jnp.isneginf(cur_high))), self.high)), \
            "`high` values cannot be negative infinity (`-jnp.inf`)."

        # assert low != high for continuous leafs
        assert jax.tree.all(jax.tree.map(
            lambda cur_low, cur_high: 
                not (jnp.issubdtype(cur_low.dtype, jnp.floating) \
                    and jnp.any(cur_low == cur_high)), 
            self.low, self.high
        )), "`low` cannot equal `high` for continuous leaves, since `high` bound is exclusive while `low` is inclusive."

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

            # NOTE: extra sanitization due to issue with jnp.where and NaNs for gradients
                # must ensure no NaNs ever get created in either branch; replace infs with dummy values

            safe_cur_low = jnp.where(jnp.isinf(cur_low), -0.5, cur_low)
            safe_cur_high = jnp.where(jnp.isinf(cur_high), 0.5, cur_high)
            squashed = (safe_cur_low+safe_cur_high)/2 + jnp.tanh(leaf)*(safe_cur_high - safe_cur_low)/2
                # handle both NOT unbounded -> tanh

            softplused = jax.nn.softplus(leaf) # only one side unbounded
            squashed = jnp.where(jnp.isinf(cur_low), cur_high - softplused, squashed)
            squashed = jnp.where(jnp.isinf(cur_high), cur_low + softplused, squashed)

            squashed = jnp.where(jnp.logical_and(jnp.isinf(cur_low), jnp.isinf(cur_high)),
                leaf, squashed) # handle both unbounded -> no-op, return original

            return squashed

        return jax.tree.map(map_func, x, self.low, self.high, self.shapes_dtypes)

    def unsquash_continuous_from_bounds(self, x: TSpaceElement) -> TSpaceElement:
        """Undos the `space.squash_continuous_to_bounds(x)` method."""
        
        def map_func(leaf, cur_low, cur_high, shape_dtype: jax.ShapeDtypeStruct):
            if jnp.issubdtype(shape_dtype.dtype, jnp.integer): return leaf

            # NOTE: extra sanitization due to issue with jnp.where and NaNs for gradients
                # must ensure no NaNs ever get created in either branch; replace infs with dummy values
    
            # both NOT unbounded -> tanh
            safe_cur_low = jnp.where(jnp.isinf(cur_low), -0.5, cur_low)
            safe_cur_high = jnp.where(jnp.isinf(cur_high), 0.5, cur_high)
            mid = (safe_cur_low+safe_cur_high)/2
            unsquashed = jnp.arctanh(jnp.clip(
                (leaf - jnp.where(jnp.isinf(mid), 0, mid)) / (safe_cur_high-safe_cur_low) * 2,
                min=-1, max=1
            ))

            # only one side unbounded -> softplus
            unsquashed = jnp.where(jnp.isinf(cur_low), 
                inv_softplus(jnp.maximum(0, cur_high - leaf)), unsquashed)
            unsquashed = jnp.where(jnp.isinf(cur_high), 
                inv_softplus(jnp.maximum(0, leaf - cur_low)), unsquashed)

            # both unbounded -> no-op, return original
            unsquashed = jnp.where(jnp.logical_and(jnp.isinf(cur_low), jnp.isinf(cur_high)),
                leaf, unsquashed)

            return unsquashed

        return jax.tree.map(map_func, x, self.low, self.high, self.shapes_dtypes)

    def sample_distribution(self, key: chex.PRNGKey, distribution: TSpaceElement, 
            squash_continuous=True, deterministic=False, log_stds=False) -> TSpaceElement:
        """Samples a single element from the space, according to the given distribution.
        
        The distribution should have the same treedef as the Space. Each leaf should have the same shape,
        but with an extra trailing axis as follows:

            - If discrete (jnp.integer): values will be sampled according to a categorical distribution.
                Trailing axis should have size equal to the maximum number of possible values 
                    for any feature in the Space in that leaf: `max(high - low + 1)`
                Values along trailing axis should be logits. The 0th logit should correspond to the
                    mininum possible value of that feature. Pad with `-jnp.inf` at the end as necessary.

            If continuous (jnp.floating): values will be sampled according to a normal distribution.
                Trailing axis should have size 2. The first element should be the mean, 
                    and the second should be the standard deviation.
                If bounded, values will be squashed to bounds using the softplus (one side bounded) 
                    or tanh (both sides bounded) function.

        `squash_continuous`: If False, does not squash continuous values, leaving them unbounded.
            Useful, eg. for sampling raw outputs, before softplus or tanh.

        `deterministic`: If True, takes the mode.
            Discrete (jnp.integer) leaves: selects the choice with the highest probability.
            Continuous (jnp.floating) leaves: takes the mean of the distribution.

        `log_stds`: If True, treats stds as log stds: uses exp(feature[1]) as the standard deviation.
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

                std = jnp.exp(cur_dist[..., 1]) if log_stds else cur_dist[..., 1]

                return cur_dist[..., 0] + std*z

        sample = jax.tree.map(sample_leaf, distribution, self.low, self.shapes_dtypes, keys_tree)

        if squash_continuous:
            sample = self.squash_continuous_to_bounds(sample)

        return sample

    def log_probabilities(self, x: TSpaceElement, distribution: TSpaceElement, 
            continuous_squashed=True, log_stds=False) -> TSpaceElement:
        """Computes the individual log probability of sampling each feature of `x` from `distribution`.
        See `space.sample_distribution` for details on the structure of `distribution`.

        `continuous_squashed`: If False, assumes all continuous values are unsquashed and unbounded.
            Useful, eg. for processing raw outputs, before softplus or tanh.

        `log_stds`: If True, treats stds as log stds: uses exp(feature[1]) as the standard deviation.
        """

        def leaf_log_probabilities(unsquashed_leaf, cur_dist, 
                cur_low, cur_high, shape_dtype: jax.ShapeDtypeStruct) -> TSpaceElement:

            if jnp.issubdtype(shape_dtype.dtype, jnp.integer):
                all_log_probs = jax.nn.log_softmax(cur_dist, axis=-1)
                dist_is = unsquashed_leaf - cur_low # unsquashed and squashed are the same
                return jnp.take_along_axis(all_log_probs, dist_is[..., None], axis=-1).squeeze(axis=-1)

            else: 
                std = jnp.exp(cur_dist[..., 1]) if log_stds else cur_dist[..., 1]
                norm_prob = jax.scipy.stats.norm.logpdf(unsquashed_leaf, loc=cur_dist[..., 0], scale=std)

                # NOTE: extra sanitization due to issue with jnp.where and NaNs for gradients
                    # must ensure no NaNs ever get created in either branch; replace infs with dummy values

                ## adjust for squashing transformation for bounded features using jacobian
                adjust = -jnp.log(1 - jnp.square(jnp.tanh(unsquashed_leaf)) + 1e-6) # neither side unbounded -> tanh

                adjust = jnp.where(jnp.logical_or(jnp.isinf(cur_low), jnp.isinf(cur_high)), 
                    jnp.logaddexp(0, -unsquashed_leaf), adjust) # handle one side unbounded -> softplus
                    
                adjust = jnp.where(jnp.logical_and(jnp.isinf(cur_low), jnp.isinf(cur_high)), 
                    0, adjust) # handle both unbounded -> no transformation

                return norm_prob + adjust

        if continuous_squashed:
            x = self.unsquash_continuous_from_bounds(x)

        return jax.tree.map(leaf_log_probabilities, x, distribution, self.low, self.high, self.shapes_dtypes)

    def log_probability(self, x: TSpaceElement, distribution: TSpaceElement, 
            continuous_squashed=True, log_stds=False) -> ArrayLike:
        """Computes the total log probability of sampling `x` from `distribution`, assuming features are independent.
        See `space.sample_distribution` for details on the structure of `distribution`.

        `continuous_squashed`: If False, assumes all continuous values are unsquashed and unbounded.
            Useful, eg. for processing raw outputs, before softplus or tanh.

        `log_stds`: If True, treats stds as log stds: uses exp(feature[1]) as the standard deviation.
        """

        log_probabilities = self.log_probabilities(x, distribution, continuous_squashed, log_stds)

        log_probabilities = jax.tree.map(
            lambda leaf, s_dt: jnp.sum(leaf, axis=tuple(range(-len(s_dt.shape), 0))),
            log_probabilities, self.shapes_dtypes
        )

        return jax.tree.reduce(lambda tot, cur: tot + cur, log_probabilities)

    def try_compute_entropies(self, distribution: TSpaceElement, 
            approximate_entropies: TSpaceElement | None = None, log_stds=False) -> tuple[TSpaceElement, TSpaceElement]:
        """Computes the individual entropy of each feature of `distribution`,
        if an analytical solution is available: discrete and unbounded continuous features.

        Returns 0 if an analytical solution is not available: bounded (one or both sides) continuous features.
        Returns a mask containing True if an analytical solution available, otherwise false.

        See `space.sample_distribution` for details on the structure of `distribution`.

        `approximate_entropies`: If given, returns the specified approximated entropy instead of 0 
            if no analytical solution available.
        `log_stds`: If True, treats stds as log stds: uses exp(feature[1]) as the standard deviation.
        """

        def leaf_entropies(cur_dist, approx_ent, cur_low, cur_high, shape_dtype: jax.ShapeDtypeStruct):
            if jnp.issubdtype(shape_dtype.dtype, jnp.integer):
                log_probs = jax.nn.log_softmax(cur_dist, axis=-1)
                log_probs = jnp.where(jnp.isneginf(log_probs), 0, log_probs) # handle 0 probability
                return - jnp.exp(log_probs) * log_probs
            else: 
                std = jnp.exp(cur_dist[..., 1]) if log_stds else cur_dist[..., 1]
                return jnp.where(jnp.logical_and(jnp.isinf(cur_low), jnp.isinf(cur_high)), 
                    normal_entropy(std), approx_ent)

        if approximate_entropies is None:
            approximate_entropies = jax.tree.map(lambda sd: jnp.zeros(sd.shape), self.shapes_dtypes)

        ents = jax.tree.map(leaf_entropies, distribution, approximate_entropies, self.low, self.high, self.shapes_dtypes)

        mask = jax.tree.map(
            lambda cur_low, cur_high, sdt: 
                jnp.full(sdt.shape, True) if jnp.issubdtype(sdt.dtype, jnp.integer)
                    else jnp.logical_and(jnp.isinf(cur_low), jnp.isinf(cur_high)),
            self.low, self.high, self.shapes_dtypes
        )

        return ents, mask

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