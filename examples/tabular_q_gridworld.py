import jax
import jax.numpy as jnp

from core.envs.gridworld import GridworldEnv, State

from core.algos.tabular_q import TabularQ, TabularQHyperparameters

SEED = 2
key = jax.random.key(SEED)

MAP = "general"
env = GridworldEnv.default_map(MAP)

### TRAIN ###

hyperparameters = TabularQHyperparameters(
    discount_rate = 0.95,
    learning_rate = 0.01,

    epsilon_final = 0.05,

    replay_buffer_size = 1000,
    batch_size = 32,
    train_freq = 1,
    n_envs = 16, #256,

    target_update_interval = 1000,
)

algo = TabularQ(env, hyperparameters=hyperparameters)

STEPS = 10_000_000
LOG_INTERVAL_STEPS = 1_000_000

key, train_key = jax.random.split(key, 2)
q_vals = algo.train(train_key, STEPS, log_interval_steps=LOG_INTERVAL_STEPS)

print(q_vals)
print(env.visualize_q_table(jnp.moveaxis(q_vals, 0, 2)))