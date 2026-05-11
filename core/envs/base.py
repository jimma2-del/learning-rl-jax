"""Abstract base class for environments."""

from typing import Any, Generic
from typing_extensions import TypeVar
from abc import ABC, abstractmethod

import functools

import jax
from jax.typing import ArrayLike

import jax.numpy as jnp

TSpaceElement = TypeVar("TSpaceElement")

class Space(Generic[TSpaceElement]):

    def __init__(self, low: TSpaceElement, high: TSpaceElement) -> None:
        """inclusive, inclusive for integer dtype; inclusive, exclusive for float dtype;"""

        low_leaves, low_treedef = jax.tree.flatten(low)
        high_leaves, high_treedef = jax.tree.flatten(high)

        assert low_treedef == high_treedef, "'low' and 'high' must have the same treedef (shape)"

        self.low = low
        self.high = high

        self.treedef = low_treedef

        self.low_leaves = low_leaves
        self.high_leaves = high_leaves

        self.leaf_dtypes = [ jnp.result_type(cur_low, cur_high) 
            for cur_low, cur_high in zip(self.low_leaves, self.high_leaves) ]
        self.leaf_shapes = [ jnp.broadcast_shapes(cur_low.shape, cur_high.shape) 
            for cur_low, cur_high in zip(self.low_leaves, self.high_leaves) ] 

    #@functools.partial(jax.jit, static_argnames=('self'))
    def sample(self, key: jax.Array) -> TSpaceElement:
        """Samples a single element from the space, according to a uniform distribution."""

        keys = jax.random.split(key, num=len(self.low_leaves))

        def sample_leaf(i):
            if jnp.issubdtype(self.leaf_dtypes[i], jnp.integer):
                return jax.random.randint(keys[i], shape=self.leaf_shapes[i], dtype=self.leaf_dtypes[i],
                    minval=self.low_leaves[i], maxval=self.high_leaves[i] + 1)
            else:
                return jax.random.uniform(keys[i], shape=self.leaf_shapes[i], dtype=self.leaf_dtypes[i],
                    minval=self.low_leaves[i], maxval=self.high_leaves[i])

        sampled_leaves = [ sample_leaf(i) for i in range(len(self.low_leaves)) ]

        return self.treedef.unflatten(sampled_leaves)

    def contains(self, x: TSpaceElement, batched: bool = False) -> bool:
        """Check if `x` is a valid member of this space. 
        If batched=True, disregards leading batch dimensions"""

        x_leaves, x_treedef = jax.tree.flatten(x)
        if x_treedef != self.treedef: return False

        for i in range(len(x_leaves)):
            if x_leaves[i].dtype != self.leaf_dtypes[i]: return False

            shape_compare_start = -len(self.leaf_shapes[i]) if batched else 0
            if x_leaves[i].shape[shape_compare_start:] != self.leaf_shapes[i]: return False

            if not bool(jnp.all(x_leaves[i] >= self.low_leaves[i])): return False

            if jnp.issubdtype(self.leaf_dtypes[i], jnp.integer):
                if not bool(jnp.all(x_leaves[i] <= self.high_leaves[i])): return False
            else:
                if not bool(jnp.all(x_leaves[i] < self.high_leaves[i])): return False

        return True

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