import jax.numpy as jnp
import jax
import functools

import timeit

@functools.partial(jax.jit, donate_argnums=(1,))
def f(key, arr):
    def g(carry, _):
        key, policy, target, step = carry

        key, subkey1, subkey2 = jax.random.split(key, 3)

        N = 1000
        indices = jax.random.randint(subkey1, (4, N), minval=0, maxval=5)
        random_vals = jax.random.uniform(subkey2, (N,))

        vals = (policy[tuple(indices)] + target[tuple(indices)] + random_vals) / 3

        policy = policy.at[tuple(indices)].add(0.1 * vals)

        update = (step+1) % 100 == 0
        target = jnp.where(update, policy, target)
        #target = jax.lax.cond(update, lambda: policy, lambda: target)

        return (key, policy, target, step + 1), None

    return jax.lax.scan(g, (key, arr, arr, 0), length=1_000)[0][1]

key = jax.random.key(0)

key, subkey1, subkey2 = jax.random.split(key, 3)
small_arr = jax.random.uniform(subkey1, shape=(5,5,5,5))
large_arr = jax.random.uniform(subkey2, shape=(100,100,200,200))

small_arr.block_until_ready()
large_arr.block_until_ready()
print("start")

key, subkey = jax.random.split(key)
print(timeit.timeit(lambda: f(subkey, small_arr).block_until_ready(), number=1))

key, subkey = jax.random.split(key)
print(timeit.timeit(lambda: f(subkey, large_arr).block_until_ready(), number=1))
