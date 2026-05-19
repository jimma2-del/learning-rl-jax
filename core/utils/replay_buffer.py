from chex import dataclass
from jax.typing import ArrayLike
from typing import Any, Generic, TypeVar

import functools

import jax
from jax import flatten_util
import jax.numpy as jnp

from core.utils import jax_utils

@dataclass(frozen=True)
class ReplayBufferState:
    data: jax.Array
    insert_i: ArrayLike
    filled_len: ArrayLike

TReplayBufferItem = TypeVar("TReplayBufferItem")

class ReplayBuffer(Generic[TReplayBufferItem]):
    def __init__(self, item_shapes_dtypes, capacity: int = 10000) -> None:
        """item_shapes_dtypes: A PyTree of jax.ShapeDtypeStruct leaves, eg. from jax.eval_shape()."""
        self.item_shapes_dtypes = item_shapes_dtypes
        self.capacity = capacity

    #@functools.partial(jax.jit, static_argnames=('self'))
    def init(self) -> ReplayBufferState:

        data = jax.tree.map(
            lambda shape_dtype: jnp.empty((self.capacity, *shape_dtype.shape), dtype=shape_dtype.dtype),
            self.item_shapes_dtypes
        )

        return ReplayBufferState(
            data = data,
            insert_i = jnp.array(0, dtype=jnp.int32),
            filled_len = jnp.array(0, dtype=jnp.int32)
        )

    #@functools.partial(jax.jit, static_argnames=('self'))
    def insert(self, state: ReplayBufferState, items: TReplayBufferItem) -> None:
        '''samples: farther back means newer'''
        
        # flatten/add batch axis if number of batch axes is not 1
        items = jax.tree.map(
            lambda shape_dtype, items_leaf: items_leaf.reshape((-1, *shape_dtype.shape)),
            self.item_shapes_dtypes, items
        )

        insert_n = jax_utils.get_tree_batch_dims(self.item_shapes_dtypes, items)[0]

        if insert_n > self.capacity:
            return ReplayBufferState(
                data = jax.tree.map(lambda x: x[insert_n - self.capacity : ], items),
                insert_i = 0,
                filled_len = self.capacity
            )

        insert_indices = (state.insert_i + jnp.arange(insert_n)) % self.capacity

        data = jax.tree.map(lambda data_leaf, items_leaf: data_leaf.at[insert_indices].set(items_leaf), 
            state.data, items)

        return ReplayBufferState(
            data = data,
            insert_i = insert_indices[-1] + 1,
            filled_len = jnp.clip(state.filled_len + insert_n, max=self.capacity)
        )

    #@functools.partial(jax.jit, static_argnames=('self', 'num_samples'))
    def sample(self, key: chex.PRNGKey, state: ReplayBufferState, num_samples: int) -> TReplayBufferItem:
        indices = jax.random.randint(key, (num_samples, ), minval=0, maxval=state.filled_len)
        return jax.tree.map(lambda x: x[indices], state.data)

if __name__ == "__main__":
    @dataclass(frozen=True)
    class Sample:
        a: ArrayLike
        b: ArrayLike

    dummy = Sample(a=1, b=1)

    buffer = ReplayBuffer(dummy, 8)
    buffer_state = buffer.init()
    buffer_state = buffer.insert(buffer_state, Sample(a=jnp.array((1,2,3)), b=jnp.array((11,12,13))))
    buffer_state = buffer.insert(buffer_state, Sample(a=jnp.array((100,200,300)), b=jnp.array((1100,1200,1300))))
    print(buffer_state.data)

    key = jax.random.key(100)

    #print(jax.random.randint(key, (10, ), minval=0, maxval=5))

    print(buffer.sample(key, buffer_state, 20))