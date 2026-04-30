import jax.numpy as jnp
from jax.typing import ArrayLike
import jax
import functools

class LinearlyInterpolatedTable:

    def __init__(self, min: ArrayLike, max: ArrayLike, step: ArrayLike):
        self.min = jnp.array(min)
        self.max = jnp.array(max)
        self.step = jnp.array(step)

        self.shape = ()

        for i in range(len(min)):
            self.shape = (*self.shape, (max[i] - min[i]) // step[i] + 1 
                + ((max[i] - min[i]) % step[i] != 0)) # add an extra if not perfectly ending on max

    @functools.partial(jax.jit, static_argnames=('self'))
    def init(self, init: ArrayLike) -> jax.Array:
        return jnp.full(self.shape, init, dtype=jnp.float32)

    @functools.partial(jax.jit, static_argnames=('self'))
    def get(self, data: jax.Array, pos: ArrayLike) -> jax.Array:
        assert data.shape == self.shape
        assert len(pos) == len(self.shape)

        lower_indices, offsets = self.get_lower_indices_and_offsets(pos)
        corner_indices, weights = self.get_corner_indices_and_weights(lower_indices, offsets)

        result = 0

        for corner, weight in zip(corner_indices, weights):
            result += data[corner] * weight

        return result

    @functools.partial(jax.jit, static_argnames=('self'))
    def adjust_get_corner_adjustments(
        self, data: jax.Array, pos: ArrayLike, adjust_amount: ArrayLike
    ) -> tuple[jax.Array, jax.Array]:
        assert data.shape == self.shape
        assert len(pos) == len(self.shape)

        lower_indices, offsets = self.get_lower_indices_and_offsets(pos)
        corner_indices, weights = self.get_corner_indices_and_weights(lower_indices, offsets)
        
        corner_indices = jnp.array(corner_indices)
        weights = jnp.array(weights)

        sum_squared_weights = jnp.sum(weights**2)
        adjust_amounts = weights * adjust_amount / sum_squared_weights

        return corner_indices, adjust_amounts

    @functools.partial(jax.jit, static_argnames=('self'))
    def set_get_corner_adjustments(
        self, data: jax.Array, pos: ArrayLike, value: ArrayLike
    ) -> tuple[jax.Array, jax.Array]:
        cur_value = self.get(data, pos)
        return self.adjust_get_corner_adjustments(data, pos, value - cur_value)

    @functools.partial(jax.jit, static_argnames=('self'))
    def adjust(self, data: jax.Array, pos: ArrayLike, adjust_amount: ArrayLike) -> jax.Array:
        corner_indices, adjust_amounts = self.adjust_get_corner_adjustments(data, pos, adjust_amount)
        return data.at[tuple(corner_indices.T)].add(adjust_amounts)

    @functools.partial(jax.jit, static_argnames=('self'))
    def set(self, data: jax.Array, pos: ArrayLike, value: ArrayLike) -> jax.Array:
        cur_value = self.get(data, pos)
        return self.adjust(data, pos, value - cur_value)
    
    @functools.partial(jax.jit, static_argnames=('self'))
    def get_lower_indices_and_offsets(self, pos: ArrayLike):
        assert len(pos) == len(self.shape)

        indices = ((pos - self.min) // self.step).astype(int)
        indices = jnp.clip(indices, 0, jnp.array(self.shape) - 2)
            # clip to inside bounds; use extrapolation for those cases

        offsets = (pos - (indices*self.step + self.min)) / self.step
            # offset from lower_index, in units of indices (normalized by step size)

        return indices, offsets

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

    for i in range(100):
        key, subkey1, subkey2, = jax.random.split(key, 3)
        coords = jax.random.uniform(subkey1, (2,), minval=jnp.array((-10,-10)), maxval=jnp.array((11,20)))
        set_val = jax.random.uniform(subkey2, (), minval=-10, maxval=10)

        table_state = table.set(table_state, coords, set_val)

        if abs(table.get(table_state, coords) - set_val) > 0.001:
            print(set_val, table.get(table_state, coords))

    print(table_state)

    #print(table.get(table_state, jnp.array((-1, 2.5))))