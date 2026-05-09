import jax.numpy as jnp
import jax

from jax.typing import ArrayLike
from chex import dataclass
from typing import TypeVar

@dataclass(frozen=True)
class Hyperparameters:
    n_envs: int = 32,

    discount_rate: float = 0.99,
    learning_rate: float = 2.5e-4,

    batch_size: int = 32,


# from flax import nnx

# class Network(nnx.Module):
#   def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
#     self.linear = nnx.Linear(din, dmid, rngs=rngs)
#     self.bn = nnx.BatchNorm(dmid, rngs=rngs)
#     self.dropout = nnx.Dropout(0.2)
#     self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)

#   def __call__(self, x, rngs):
#     x = nnx.relu(self.dropout(self.bn(self.linear(x)), rngs=rngs))
#     return self.linear_out(x)