from chex import dataclass
import chex
from jax.typing import ArrayLike
from typing import Any, Generic, TypeVar, Self, Sequence

import jax
from jax import flatten_util
import jax.numpy as jnp

from core.utils.batch_utils import get_tree_batch_dims

TReplayBufferItem = TypeVar("TReplayBufferItem")

@dataclass(frozen=True)
class ReplayBuffer(Generic[TReplayBufferItem]):
    data: TReplayBufferItem
    insert_i: ArrayLike
    filled_len: ArrayLike

    @classmethod
    def init(cls, item_shapes_dtypes: TReplayBufferItem, capacity: int = 10000) -> Self:
        """item_shapes_dtypes: A PyTree of jax.ShapeDtypeStruct leaves, eg. from jax.eval_shape()."""

        data = jax.tree.map(
            lambda shape_dtype: jnp.empty((capacity, *shape_dtype.shape), dtype=shape_dtype.dtype),
            item_shapes_dtypes
        )

        return cls(
            data = data,
            insert_i = jnp.array(0, dtype=jnp.int32),
            filled_len = jnp.array(0, dtype=jnp.int32)
        )

    def insert(self, items: TReplayBufferItem) -> Self:
        """`items`: Farther back means newer."""
        shapes_dtypes = jax.eval_shape(lambda: jax.tree.map(lambda x: x[0], self.data))
        capacity = get_tree_batch_dims(shapes_dtypes, self.data)[0]
            
        # flatten/add batch axis if number of batch axes is not 1
        items = jax.tree.map(
            lambda shape_dtype, items_leaf: items_leaf.reshape((-1, *shape_dtype.shape)),
            shapes_dtypes, items
        )

        insert_n = get_tree_batch_dims(shapes_dtypes, items)[0]

        if insert_n > capacity:
            return ReplayBuffer(
                data = jax.tree.map(lambda x: x[insert_n - capacity : ], items),
                insert_i = jnp.array(0),
                filled_len = jnp.array(capacity)
            )

        insert_indices = (self.insert_i + jnp.arange(insert_n)) % capacity

        data = jax.tree.map(lambda data_leaf, items_leaf: data_leaf.at[insert_indices].set(items_leaf), 
            self.data, items)

        return ReplayBuffer(
            data = data,
            insert_i = insert_indices[-1] + 1,
            filled_len = jnp.minimum(self.filled_len + insert_n, capacity)
        )

    def sample(self, key: chex.PRNGKey, batch_dims: Sequence[int]) -> TReplayBufferItem:
        indices = jax.random.randint(key, batch_dims, minval=0, maxval=self.filled_len)
        return jax.tree.map(lambda x: x[indices], self.data)

if __name__ == "__main__":
    @dataclass(frozen=True)
    class Sample:
        a: ArrayLike
        b: ArrayLike

    buffer = ReplayBuffer.init(jax.eval_shape(lambda: Sample(a=1, b=2)), 8)
    buffer = buffer.insert(Sample(a=jnp.array((1,2,3)), b=jnp.array((11,12,13))))
    buffer = buffer.insert(Sample(a=jnp.array((100,200,300)), b=jnp.array((1100,1200,1300))))
    print(buffer.data)

    key = jax.random.key(100)

    #print(jax.random.randint(key, (10, ), minval=0, maxval=5))

    print(buffer.sample(key, (20,)))