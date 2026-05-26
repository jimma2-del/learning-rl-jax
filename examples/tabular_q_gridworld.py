import time

import jax
import jax.numpy as jnp

from optax import schedules

from flax import nnx

from core.envs.gridworld import GridworldEnv, State

from core.algos import tabular_q
from core.envs.utils import evaluate_episodes

rngs = nnx.Rngs(0, params=1, env=2, actions=3, transitions=4)

MAP = "general"
env = GridworldEnv.default_map(MAP)

### TRAIN ###
STEPS = 200_000
LOG_INTERVAL_STEPS = 10_000
EVAL_EPS = 2048#32 
    # very number of eval eps for maze env since high variance due to random starting position

hyperparameters = tabular_q.Hyperparameters(
    discount_rate = 0.95,
    learning_rate = 0.1,

    epsilon = schedules.linear_schedule(1, 0.05, 0.1*STEPS),

    replay_buffer_size = 1000,
    batch_size = 32,
    train_freq = 4,
    n_envs = 32, #256,

    target_update_interval = 100,
)

algo = tabular_q.TabularQ(env, hyperparameters=hyperparameters)

training_state = algo.init_training_state(rngs)

while training_state.steps < STEPS:
    start_time = time.perf_counter()

    training_state, metrics = algo.train_epoch(rngs, training_state, LOG_INTERVAL_STEPS)

    elasped_time = time.perf_counter() - start_time
    sps = LOG_INTERVAL_STEPS / elasped_time
    print(f"Completed steps={training_state.steps}; sps={sps:,.1f}")
    print("Metrics: " + " ".join([ f"{key}={val}" for key, val in metrics.items() ]))

    # eval
    returns, lengths = nnx.jit(evaluate_episodes, static_argnums=(1, 2, 3, 4, 5))(
        rngs, env, 
        lambda rngs, obs: algo.get_action(rngs, training_state.policy, obs), 
        EVAL_EPS, hyperparameters.n_envs
    )

    print(f"Episode Return: mean={jnp.mean(returns)} std={jnp.std(returns, ddof=1)}")
    print(f"Episode Length: mean={jnp.mean(lengths)} std={jnp.std(lengths, ddof=1)}")

    print()

print(training_state.policy)
print(env.visualize_q_table(jnp.moveaxis(training_state.policy, 0, 2)))