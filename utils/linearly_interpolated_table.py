import jax.numpy as jnp
from jax.typing import ArrayLike
import jax
import functools

class LinearlyInterpolatedTable:

    def __init__(self, min, max, step):
        self.min = min
        self.max = max
        self.step = step

        self.shape = ()

        for i in range(len(min)):
            self.shape = (*self.shape, (max[i] - min[i]) // step[i] + 1 
                + ((max[i] - min[i]) % step[i] != 0)) # add an extra if not perfectly ending on max

    @functools.partial(jax.jit, static_argnames=('self'))
    def init(self, init):
        return jnp.full(self.shape, init, dtype=jnp.float32)

    @functools.partial(jax.jit, static_argnames=('self'))
    def get(self, data: ArrayLike, pos: ArrayLike) -> jax.Array:
        assert data.shape == self.shape
        assert len(pos) == len(self.shape)

        lower_indices, offsets = self.get_lower_indices_and_offsets(pos)
        corner_indices, weights = self.get_corner_indices_and_weights(lower_indices, offsets)

        result = 0

        for corner, weight in zip(corner_indices, weights):
            result += data[corner] * weight

        return result

    @functools.partial(jax.jit, static_argnames=('self'))
    def adjust(self, data: ArrayLike, pos: ArrayLike, adjust_amount) -> jax.Array:
        assert data.shape == self.shape
        assert len(pos) == len(self.shape)

        lower_indices, offsets = self.get_lower_indices_and_offsets(pos)
        corner_indices, weights = self.get_corner_indices_and_weights(lower_indices, offsets)

        sum_squared_weights = sum(map(lambda x: x**2, weights))
        adjust_amounts = map(lambda x: x * adjust_amount / sum_squared_weights, weights)

        for corner, cur_adjust_amount in zip(corner_indices, adjust_amounts):
            data = data.at[corner].add(cur_adjust_amount)

        return data

    def adjust_get_corner_adjustments():
        pass

    def set_get_corner_adjustments():
        pass

    @functools.partial(jax.jit, static_argnames=('self'))
    def set(self, data: ArrayLike, pos: ArrayLike, value) -> jax.Array:
        cur_value = self.get(data, pos)
        return self.adjust(data, pos, value - cur_value)
    
    def get_lower_indices_and_offsets(self, pos: ArrayLike):
        assert len(pos) == len(self.shape)

        lower_indices = []
        offsets = []

        for i in range(len(self.shape)):
            index = (pos[i] - self.min[i]) // self.step[i]
            index = jnp.clip(index, 0, self.shape[i] - 2)
            lower_indices.append(jnp.rint(index).astype(int))

            offsets.append((pos[i] - (index*self.step[i] + self.min[i])) / self.step[i])

        return lower_indices, offsets

    def get_corner_indices_and_weights(self, lower_indices, offsets, chosen_indices=(), weight=1):
        assert len(offsets) == len(self.shape)
        
        if len(chosen_indices) == len(self.shape):
            return [ chosen_indices ], [ weight ]
        
        indices_list1, weights_list1 = self.get_corner_indices_and_weights(lower_indices, offsets, 
            (*chosen_indices, lower_indices[len(chosen_indices)]), weight * (1 - offsets[len(chosen_indices)]))
        indices_list2, weights_list2 = self.get_corner_indices_and_weights(lower_indices, offsets, 
            (*chosen_indices, lower_indices[len(chosen_indices)] + 1), weight * offsets[len(chosen_indices)])

        return indices_list1 + indices_list2, weights_list1 + weights_list2

if __name__ == "__main__":
    table = LinearlyInterpolatedTable((0,1), (9,11), (2,3))
    table_state = table.init(1)

    print(table_state)

    #table_state = table_state.at[1,1].set(5)

    key = jax.random.key(0)

    for i in range(1000):
        key, subkey1, subkey2, = jax.random.split(key, 3)
        coords = jax.random.uniform(subkey1, (2,), minval=jnp.array((-10,-10)), maxval=jnp.array((11,20)))
        set_val = jax.random.uniform(subkey2, (), minval=-10, maxval=10)

        table_state = table.set(table_state, coords, set_val)

        if abs(table.get(table_state, coords) - set_val) > 0.001:
            print(set_val, table.get(table_state, coords))

    print(table_state)

    #print(table.get(table_state, jnp.array((-1, 2.5))))