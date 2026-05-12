import jax
import jax.numpy as jnp

import chex

def shape_matches_excluding_batch_dims(unbatched_shape, x_shape):
    if len(unbatched_shape) == 0: return True
    if len(unbatched_shape) > len(x_shape): return False
    return x_shape[-len(unbatched_shape):] == unbatched_shape

def shape_is_batched(unbatched_shape, x_shape):
    return len(x_shape) > len(unbatched_shape) \
        and shape_matches_excluding_batch_dims(unbatched_shape, x_shape)

def is_batched(unbatched_shape, x):
    return shape_is_batched(unbatched_shape, x.shape)

def get_shape_batch_dims(unbatched_shape, x_shape):
    if len(unbatched_shape) == 0: return x_shape
    assert shape_matches_excluding_batch_dims(unbatched_shape, x_shape)
    return x_shape[:-len(unbatched_shape)]

def get_batch_dims(unbatched_shape, x):
    return get_shape_batch_dims(unbatched_shape, x.shape)

def get_tree_batch_dims(unbatched_shapes_dtypes, x):
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