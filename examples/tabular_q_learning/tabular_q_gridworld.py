import time

import jax
import jax.numpy as jnp

from optax import schedules

from flax import nnx

from core.envs.gridworld import GridworldEnv, State

from core.algos import tabular_q_learning
from core.envs.utils import evaluate_episodes
from core.envs.wrappers import VmapWrapper

rngs = nnx.Rngs(0, params=1, env=2, actions=3, transitions=4)

MAP = "general"
env = GridworldEnv.built_in_map(MAP)

### TRAIN ###
STEPS = 50_000

EVAL_EPS = 2048 # very high number of eval eps for gridworld bc high variance from random starting position
EVAL_INTERVAL = 5_000
N_LOGS_PER_EVAL = 2

hyperparameters = tabular_q_learning.Hyperparameters(
    discount_rate = 0.95,
    learning_rate = 0.1,
    epsilon = schedules.linear_schedule(1, 0.05, 0.1*STEPS),
    n_envs = 32, #256,
)

algo = tabular_q_learning.TabularQLearning(VmapWrapper(env), hyperparameters=hyperparameters)

training_state = algo.init_training_state(rngs)
train = nnx.jit(algo.train, static_argnames=('steps',), donate_argnames=('training_state'))

@nnx.jit
def evaluate(rngs, actor):
    return evaluate_episodes(
        rngs, VmapWrapper(env), actor, 
        EVAL_EPS, EVAL_EPS
    )

while training_state.steps < STEPS:
    start_time = time.perf_counter()
    training_state, metrics = train(rngs, training_state, EVAL_INTERVAL)
    elasped_time = time.perf_counter() - start_time

    print()

    avg_metrics = jax.tree.map(lambda x: list(map(jnp.mean, jnp.array_split(x, N_LOGS_PER_EVAL))), metrics)
    steps = avg_metrics.pop('steps')
    for i in range(N_LOGS_PER_EVAL):
        print(f"Step {steps[i]:.0f}: " + " ".join([ f"{key}={val[i]:.5g}" for key, val in avg_metrics.items() ]))

    print()

    sps = EVAL_INTERVAL / elasped_time
    print(f"COMPLETED steps={training_state.steps}; sps={sps:,.1f}")

    #eval
    actor = algo.make_actor(training_state.q_func, epsilon=0)
    returns, lengths = evaluate(rngs, actor)

    print(f"Episode Return: mean={jnp.mean(returns)} std={jnp.std(returns, ddof=1)}")
    print(f"Episode Length: mean={jnp.mean(lengths)} std={jnp.std(lengths, ddof=1)}")

print()

print("Q Table:")
print(training_state.q_func.q_table_values.value)
print()

print("Q Table Gridworld Visualization:")
print(env.visualize_q_table(jnp.moveaxis(training_state.q_func.q_table_values.value, 0, 2)))