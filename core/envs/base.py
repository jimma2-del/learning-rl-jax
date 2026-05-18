"""Abstract base class for environments."""

from typing import Any, Generic
from typing_extensions import TypeVar
from abc import ABC, abstractmethod

import chex

import functools

import jax
from jax.typing import ArrayLike

import jax.numpy as jnp

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

        self.low = low
        self.high = high

        self.treedef = jax.tree.structure(low)

        self.shapes_dtypes = jax.tree.map(lambda cur_low, cur_high: jax.ShapeDtypeStruct(
            dtype = jnp.result_type(cur_low, cur_high),
            shape = jnp.broadcast_shapes(cur_low.shape, cur_high.shape)
        ), low, high)

    #@functools.partial(jax.jit, static_argnames=('self'))
    def sample(self, key: jax.Array) -> TSpaceElement:
        """Samples a single element from the space, according to a uniform distribution.
        Does not currently support unbounded leaves (low or high are infinity)"""

        keys = jax.random.split(key, num=self.treedef.num_leaves)
        keys_tree = jax.tree.unflatten(self.treedef, keys)

        def sample_leaf(cur_low, cur_high, shape_dtype: jax.ShapeDtypeStruct, key: jax.Array):
            if jnp.issubdtype(shape_dtype.dtype, jnp.integer):
                return jax.random.randint(key, shape=shape_dtype.shape, dtype=shape_dtype.dtype,
                    minval=cur_low, maxval=cur_high + 1)
            else:
                return jax.random.uniform(key, shape=shape_dtype.shape, dtype=shape_dtype.dtype,
                    minval=cur_low, maxval=cur_high)

        return jax.tree.map(sample_leaf, self.low, self.high, self.shapes_dtypes, keys_tree)

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

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")
TEnvAction = TypeVar("TEnvAction")
TRenderFrame = TypeVar("TRenderFrame", default=None)

class Environment(ABC, Generic[TEnvState, TEnvObs, TEnvAction, TRenderFrame]):
    """Abstract base class for environments."""

    @abstractmethod
    def reset(self, key: jax.Array) -> tuple[TEnvState, dict[Any, Any]]:
        """Performs resetting of environment.
        Returns: state, info"""

    @abstractmethod
    def step(self, key: jax.Array, state: TEnvState, action: TEnvAction) \
        -> tuple[TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        """Performs step transitions in the environment.
        Returns: state, reward, terminated, truncated, info"""

    @abstractmethod
    def get_obs(self, key: jax.Array, state: TEnvState) -> TEnvObs:
        """Applies observation function to state."""

    def render(self, state: TEnvState, action: ArrayLike) -> TRenderFrame:
        """Compute a render frame from the state-action pair.
        Intended for human interpretation (visualization, debugging); should not be used as a policy input.
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