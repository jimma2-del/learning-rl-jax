from chex import dataclass
from jax.typing import ArrayLike
from typing import Any, Generic, TypeVar

import functools

import jax
from jax import flatten_util
import jax.numpy as jnp

@dataclass(frozen=True)
class ReplayBufferState:
    data: jax.Array
    insert_i: ArrayLike
    filled_len: ArrayLike

TReplayBufferItem = TypeVar("TReplayBufferItem")

class ReplayBuffer(Generic[TReplayBufferItem]):
    def __init__(self, dummy_item: TReplayBufferItem, capacity: int = 10000) -> None:
        dummy_flattened, unravel_func = flatten_util.ravel_pytree(dummy_item)

        self.batched_flatten_func = jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])
        self.batched_unflatten_func = jax.vmap(unravel_func)

        self.capacity = capacity
        self.item_size = len(dummy_flattened)
        self.data_shape = (capacity, self.item_size)
        self.dtype = dummy_flattened.dtype

    @functools.partial(jax.jit, static_argnames=('self'))
    def init(self) -> ReplayBufferState:
        return ReplayBufferState(
            data = jnp.empty(self.data_shape, dtype=self.dtype),
            insert_i = jnp.array(0),
            filled_len = jnp.array(0)
        )

    @functools.partial(jax.jit, static_argnames=('self'))
    def insert(self, state: ReplayBufferState, samples: TReplayBufferItem) -> None:
        '''samples: farther back means newer'''

        flattened_samples = self.batched_flatten_func(samples)

        if len(flattened_samples) > self.capacity:
            return ReplayBufferState(
                data = flattened_samples[len(flattened_samples) - self.capacity : ],
                insert_i = 0,
                filled_len = self.capacity
            )

        insert_indices = (state.insert_i + jnp.arange(len(flattened_samples))) % self.capacity

        return ReplayBufferState(
            data = state.data.at[insert_indices].set(flattened_samples),
            insert_i = insert_indices[-1] + 1,
            filled_len = jnp.clip(state.filled_len + len(flattened_samples), max=self.capacity)
        )

    @functools.partial(jax.jit, static_argnames=('self', 'num_samples'))
    def sample(self, key: jax.Array, state: ReplayBufferState, num_samples: int) -> TReplayBufferItem:
        indices = jax.random.randint(key, (num_samples, ), minval=0, maxval=state.filled_len)
        flattened_samples = state.data[indices]
        return self.batched_unflatten_func(flattened_samples)

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