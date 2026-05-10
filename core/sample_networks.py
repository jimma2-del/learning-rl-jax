from jax.typing import ArrayLike
from typing import Any, Generic, TypeVar

from jax import flatten_util
import jax

from flax import nnx

class MLP(nnx.Module):

    def __init__(self, rngs: nnx.Rngs, 
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256, 
        num_hidden_layers: int = 2,
        do_layer_norm: bool = True,
        activation_func = nnx.swish
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        self.do_layer_norm = do_layer_norm
        self.activation_func = activation_func

        self.hidden_layers = [ nnx.Linear(input_dim if i == 0 else hidden_dim, hidden_dim, rngs=rngs)
            for i in range(num_hidden_layers) ]

        if do_layer_norm:
            self.hidden_norms = [ nnx.LayerNorm(input_dim if i == 0 else hidden_dim, rngs=rngs)
                for i in range(num_hidden_layers) ]

        self.output_layer = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, x: jax.Array, rngs: nnx.Rngs):

        for i in range(self.num_hidden_layers):
            x = self.hidden_layers[i](x)

            if self.do_layer_norm:
                x = self.hidden_norms[i](x)

            x = self.activation_func(x)

        return self.output_layer(x)

TInputType = TypeVar("TInputType")

class MLPFeatureExtractor(nnx.Module, Generic[TInputType]):

    def __init__(self, rngs: nnx.Rngs, 
        dummy_input: TInputType,

        output_dim: int = 256,
        output_activation_func = None, # same as activation_func by default

        hidden_dim: int = 256, 
        num_hidden_layers: int = 1,
        do_layer_norm: bool = True,
        activation_func = nnx.swish,
    ):
        dummy_flattened, unravel_func = flatten_util.ravel_pytree(dummy_input)

        self.batched_flatten_func = jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])
        self.unflatten_func = unravel_func

        self.flattened_len = len(dummy_flattened)
        self.output_dim = output_dim

        self.output_activation_func = output_activation_func \
            if output_activation_func is not None else activation_func

        self.do_layer_norm = do_layer_norm

        self.mlp = MLP(rngs, self.flattened_len, output_dim,
            hidden_dim, num_hidden_layers, do_layer_norm, activation_func)

        if self.do_layer_norm:
            self.output_norm = nnx.LayerNorm(output_dim, rngs)

    def __call__(self, x: Generic[TInputType], rngs: nnx.Rngs):
        x = self.batched_flatten_func(x)
        x = self.mlp(x, rngs)

        if self.do_layer_norm:
            x = self.output_norm(x)

        return self.output_activation_func(x)