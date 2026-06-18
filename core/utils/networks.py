import math

from jax.typing import ArrayLike
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp

import numpy as np

from flax import nnx

from core.envs.base import Space
from core.utils.batch_utils import flatten_batched_tree

class MLP(nnx.Module):

    def __init__(self, rngs: nnx.Rngs, 
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256, 
        num_hidden_layers: int = 1,
        do_layer_norm: bool = True,
        activation_func = nnx.relu
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        self.do_layer_norm = do_layer_norm
        self.activation_func = activation_func

        self.hidden_layers = [ nnx.Linear(input_dim if i == 0 else hidden_dim, hidden_dim, rngs=rngs)
            for i in range(num_hidden_layers) ]

        if do_layer_norm:
            self.hidden_norms = [ nnx.LayerNorm(hidden_dim, rngs=rngs)
                for i in range(num_hidden_layers) ]

        self.output_layer = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:

        for i in range(self.num_hidden_layers):
            x = self.hidden_layers[i](x)

            if self.do_layer_norm:
                x = self.hidden_norms[i](x)

            x = self.activation_func(x)

        return self.output_layer(x)

TEnvObs = TypeVar("TEnvObs")

class FlattenAndProject(nnx.Module, Generic[TEnvObs]):
    """Flattens an input PyTree and projects it to the specified dimensions (using an `nnx.Linear` layer)."""

    def __init__(self, rngs: nnx.Rngs, 
        shapes_dtypes,
        output_dim: int = 256,
    ) -> None:
        """
        shapes_dtypes: A PyTree of jax.ShapeDtypeStruct leaves, eg. from jax.eval_shape(),
            specifiying the structure of the inputs to flatten.
        """

        self.shapes_dtypes = shapes_dtypes

        self.flattened_len = jax.tree.reduce(
            lambda cum_len, shape_dtype: cum_len + (
                1 if len(shape_dtype.shape) == 0 else math.prod(shape_dtype.shape)), 
            shapes_dtypes, 0
        )
        self.output_dim = output_dim

        self.linear = nnx.Linear(self.flattened_len, output_dim, rngs=rngs)

    def __call__(self, x: TEnvObs, rngs: nnx.Rngs) -> jax.Array:
        x = flatten_batched_tree(self.shapes_dtypes, x)
        return self.linear(x)

TEnvAction = TypeVar("TEnvAction")

class ActionDistributionHead(nnx.Module, Generic[TEnvAction]):

    def __init__(self, rngs: nnx.Rngs, 
        action_space: Space[TEnvAction],
        input_dim: int = 256,
        do_state_independent_stds = True,
    ):
        self.treedef = action_space.treedef
        self.input_dim = input_dim
        self.do_state_independent_stds = do_state_independent_stds

        self.output_dim = 0
        self.num_continuous = 0
        self._leaves_descrip = []

        for cur_low, cur_high in zip(jax.tree.leaves(action_space.low), jax.tree.leaves(action_space.high)):
            if np.issubdtype(cur_low.dtype, np.integer):
                n_choices = cur_high - cur_low + 1
                output_layer_dim = int(np.sum(n_choices))
                self.output_dim += output_layer_dim

                self._leaves_descrip.append({ 
                    'discrete': True, 
                    'output_layer_dim': output_layer_dim,
                    'shape': (*cur_low.shape, int(np.max(n_choices))),
                    'n_choices': n_choices.tolist() # convert to list to mark as static
                })
            else:
                dim = math.prod(cur_low.shape)
                self.num_continuous += dim
                output_layer_dim = dim * (2 - do_state_independent_stds)
                self.output_dim += output_layer_dim

                self._leaves_descrip.append({ 
                    'discrete': False, 
                    'output_layer_dim': output_layer_dim,
                    'shape': (*cur_low.shape, 2), # mean, std
                })
        
        if do_state_independent_stds:
            self.state_independent_log_stds = nnx.Param(jnp.full(self.num_continuous, jnp.log(0.5)))
                # stds should be initialized small, 0.5 is best; https://arxiv.org/abs/2006.05990

        self.linear = nnx.Linear(input_dim, self.output_dim, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = self.linear(x)

        leaves = []
        i = 0
        state_indep_log_stds_i = 0

        for leaf_descrip in self._leaves_descrip:
            vals = x[..., i : i+leaf_descrip['output_layer_dim']]
            i += leaf_descrip['output_layer_dim']

            if leaf_descrip['discrete']:
                leaf = jnp.full(vals.shape[:-1] + leaf_descrip['shape'], -jnp.inf)
                vals_i = 0
                
                for path, cur_n_choices in np.ndenumerate(np.asarray(leaf_descrip['n_choices'])):
                    leaf = leaf.at[(..., *path, slice(cur_n_choices))].set(vals[..., vals_i : vals_i + cur_n_choices])
                    vals_i += cur_n_choices

                leaves.append(leaf)

            else:
                if self.do_state_independent_stds:
                    state_indep_log_stds = self.state_independent_log_stds.value \
                        [state_indep_log_stds_i : state_indep_log_stds_i + leaf_descrip['output_layer_dim']]
                    state_indep_log_stds_i += leaf_descrip['output_layer_dim']

                    means = vals.reshape(vals.shape[:-1] + leaf_descrip['shape'][:-1])

                    leaves.append(jnp.stack((
                        means, 
                        jnp.broadcast_to(state_indep_log_stds, means.shape)
                    ), axis=-1))
                else:
                    leaves.append(vals.reshape(leaf_descrip['shape']))

        return jax.tree.unflatten(self.treedef, leaves)