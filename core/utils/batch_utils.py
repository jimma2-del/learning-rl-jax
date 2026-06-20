from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import chex

def shape_matches_excluding_batch_dims(unbatched_shape: Sequence[int], x_shape: Sequence[int]) -> bool:
    if len(unbatched_shape) == 0: return True
    if len(unbatched_shape) > len(x_shape): return False
    return x_shape[-len(unbatched_shape):] == unbatched_shape

def shape_is_batched(unbatched_shape: Sequence[int], x_shape: Sequence[int]) -> bool:
    return len(x_shape) > len(unbatched_shape) \
        and shape_matches_excluding_batch_dims(unbatched_shape, x_shape)

def is_batched(unbatched_shape: Sequence[int], x: jax.Array) -> bool:
    return shape_is_batched(unbatched_shape, x.shape)

def get_shape_batch_dims(unbatched_shape: Sequence[int], x_shape: Sequence[int]) -> Sequence[int]:
    if len(unbatched_shape) == 0: return x_shape
    assert shape_matches_excluding_batch_dims(unbatched_shape, x_shape)
    return x_shape[:-len(unbatched_shape)]

def get_batch_dims(unbatched_shape: Sequence[int], x: jax.Array) -> Sequence[int]:
    return get_shape_batch_dims(unbatched_shape, x.shape)

def get_tree_batch_dims(unbatched_shapes_dtypes, x) -> Sequence[int]:
    chex.assert_trees_all_equal_structs(x, unbatched_shapes_dtypes,
        custom_message="'x' must have the same treedef (structure) as 'unbatched_shapes_dtypes'.")

    assert len(jax.tree.leaves(x)) != 0, "Trees cannot be empty."
    
    assert jax.tree.all(jax.tree.map(
        lambda shape_dtype, x_leaf: shape_matches_excluding_batch_dims(shape_dtype.shape, x_leaf.shape), 
        unbatched_shapes_dtypes, x
    )), "Leaves of 'x' must have the shape defined in the corresponding leaf of 'unbatched_shapes_dtypes'."

    batch_dims_shapes_dtypes = jax.tree.map(
        lambda unbatched_shape_dtype, x_leaf: jax.ShapeDtypeStruct(
            shape = get_batch_dims(unbatched_shape_dtype.shape, x_leaf),
            dtype = unbatched_shape_dtype.dtype
        ),
        unbatched_shapes_dtypes, x
    )

    batch_dims_shapes_dtypes_leaves = jax.tree.leaves(batch_dims_shapes_dtypes)

    assert all([ batch_dims_shape_dtype.shape == batch_dims_shapes_dtypes_leaves[0].shape 
            for batch_dims_shape_dtype in batch_dims_shapes_dtypes_leaves ] ), \
        "Leaf batch dimensions are not the same."

    return batch_dims_shapes_dtypes_leaves[0].shape


def flatten_batched_tree(unbatched_shapes_dtypes, x) -> jax.Array:
    """Flattens a batched PyTree into a single array while preserving leading batch axes.
    
    Args:
        unbatched_shapes_dtypes: A PyTree of jax.ShapeDtypeStruct leaves, eg. from jax.eval_shape().
        x: A PyTree with the same structure as unbatched_shapes_dtypes and leaves with corresponding shapes, 
            but potentially with extra leading batch axes.
        
    Returns:
        A jax.Array of shape (*batch_dims, total_flattened_size)."""

    leaves_x = jax.tree.leaves(x)

    if not leaves_x:
        return jnp.array([]) # convention is to return a empty 1D array if empty PyTree

    batch_dims = get_tree_batch_dims(unbatched_shapes_dtypes, x)

    return jnp.concatenate([ leaf.reshape((*batch_dims, -1)) for leaf in leaves_x ], axis=-1)

def get_tree_vmap_dim(tree) -> int:
    """Finds the length of the leading axis of the PyTree leaves, ensuring that these lengths are all equal.
    This dimension would be the batch dimension if passed into a vmap'ed function.
    """
    
    leaves = jax.tree.leaves(tree)
    assert leaves, "`tree` cannot be empty"
    dim = leaves[0].shape[0] # all leaves must be arrays to be usable in vmap

    # assert all([ leaf.shape[0] == leaves[0].shape[0] for leaf in leaves ] ), \
    #     "Leaf batch dimensions are not the same."
        # NOT TRUE for mjx warp backend

    return dim

def split_key_if_batched(key: chex.PRNGKey, batch_num: int | None = None) -> chex.PRNGKey:
    """Splits key into an array of length `batch_num`.
    If `batch_num` is None, does nothing, returning `key` unaltered."""
    return key if batch_num is None else jax.random.split(key, batch_num)

def dummy_vmap(f: Callable) -> Callable:
    """Mimics the behavior of `jax.vmap`, but does not actually apply vmap transformation; instead, 
        for inputs, simply takes the first element in the batch axis,
        and for outputs, adds a dummy batch axis of length 1."""

    def dummy_vmapped(*args, **kwargs):
        unbatched_args, unbatched_kwargs = jax.tree.map(lambda x: x[0], (args, kwargs))
        return jax.tree.map(lambda x: x[None, ...], f(*unbatched_args, **unbatched_kwargs))

    return dummy_vmapped

def batched_index(arr: jax.Array, indices: jax.Array) -> tuple[jax.Array, ...]:
    """Converts an array containing a batch of indices into a tuple of indices along individual 
        axes of `arr`, ready for use in indexing into `arr`: `arr[batched_index(arr, indices)]`.

    `arr` can have leading batch dimensions if they match with the rightmost batch dimensions of `indices`.
        In this case, `indices` will select the corresponding batch slice from `arr`.

    `arr`: Array to index into, of shape `(arr_batch_dims, *arr_dims)`.

    `indices`: Array of shape `(*batch_dims, len(arr_dims))`; values along the last axis represent
        indices along the corresponding axis of `arr`, while axes on the left are batch axes.

    Returns: Tuple of length `len(arr.shape)`, where each item is an 
        array of indices along the corresponding axis of `arr`.
    """

    indices_batch_dims = indices.shape[:-1]
    n_indices_batch_dims = len(indices_batch_dims)

    n_arr_batch_dims = len(arr.shape) - indices.shape[-1]
    assert n_arr_batch_dims >= 0, (
        "`indices` contains too many indices: "
        f"{len(arr.shape)}-dimensional array cannot be indexed with {indices.shape[-1]} indices."
    )

    arr_batch_dims = arr.shape[:n_arr_batch_dims]
    indices_trailing_batch_dims = indices_batch_dims[n_indices_batch_dims - n_arr_batch_dims:]
    assert arr_batch_dims == indices_batch_dims[n_indices_batch_dims - n_arr_batch_dims:], (
        f"Batch dimensions of `arr` {arr_batch_dims} do not match with " 
        f"rightmost batch dimensions of `indices` {indices_trailing_batch_dims}."
    )

    extra_index_dims = []
    for i, size in enumerate(arr_batch_dims):
        shape = [1] * n_indices_batch_dims
        shape[n_indices_batch_dims - n_arr_batch_dims + i] = size
        extra_index_dims.append(jnp.arange(size).reshape(shape))

    return tuple(extra_index_dims) + tuple(jnp.moveaxis(indices, -1, 0))