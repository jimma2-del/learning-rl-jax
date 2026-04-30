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
