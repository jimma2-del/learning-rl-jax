import time

import numpy as np

import jax
from jax.typing import ArrayLike
import chex

import jax.numpy as jnp

from optax import schedules

from flax import nnx

from core.envs.flappy_bird import FlappyBirdEnv, State

from core.envs.wrappers import Wrapper, VmapWrapper
from core.envs.base import Space

from core.envs.utils import evaluate_episodes

from core.algos import tabular_q_learning

DT = 0.1
env = FlappyBirdEnv(DT)

# remove pipe_dx from observations to simply

class FlappyBirdWrapper(Wrapper[State, jax.Array, ArrayLike, jax.Array]):

    def get_obs(self, key: chex.PRNGKey, state: State) -> jax.Array:
        obs = super().get_obs(key, state)
        return obs[:2]

    @property
    def observation_space(self):
        space = super().observation_space
        return Space(low=space.low[:2], high=space.high[:2])

env = FlappyBirdWrapper(env)

### TRAIN ###

# bird_pos_y, bird_vel_y, pipe1_pos_x, pipe1_pos_y, pipe2_pos_x, pipe2_pos_y
# low  = (   0, -600, -60, 175, 300, 175 )
# high = ( 800, 1500, 300, 625, 660, 625 ) 
# res  = (  25,  100,  30,  25,  30,  25 )
# #res  = (   5,   50,  15,   5,  15,   5 )

# bird_vel_y, pipe_dy
low  = (-600, -625)
high = (1500,  625)
res  = (  50,    5)

rngs = nnx.Rngs(0, params=10, env=20, actions=30, transitions=40)

STEPS = 2_000_000
LOG_INTERVAL_STEPS = 200_000
EVAL_EPS = 32

hyperparameters = tabular_q_learning.Hyperparameters(
    discount_rate = 0.95,
    learning_rate = 0.1,

    epsilon = schedules.linear_schedule(1, 0.05, 0.1*STEPS),

    replay_buffer_size = 100_000,
    batch_size = 32,
    train_freq = 4,
    n_envs = 32, #256,

    target_update_interval = 1000,
)

algo = tabular_q_learning.TabularQLearning(VmapWrapper(env), hyperparameters)
train = nnx.jit(algo.train, static_argnames=('steps',))

q_func = tabular_q_learning.LinInterpTabularQFunc(
    algo.num_actions, Space(np.array(low), np.array(high)), np.array(res))
training_state = algo.init_training_state(rngs, q_func)

@nnx.jit
def evaluate(rngs, actor):
    return evaluate_episodes(
        rngs, VmapWrapper(env), actor, 
        EVAL_EPS, hyperparameters.n_envs
    )

while training_state.steps < STEPS:
    start_time = time.perf_counter()

    training_state, metrics = train(rngs, training_state, LOG_INTERVAL_STEPS)

    elasped_time = time.perf_counter() - start_time
    sps = LOG_INTERVAL_STEPS / elasped_time
    print(f"Completed steps={training_state.steps}; sps={sps:,.1f}")
    print("Metrics: " + " ".join([ f"{key}={val}" for key, val in metrics.items() ]))

    # eval
    returns, lengths = evaluate(rngs, training_state.actor)

    print(f"Episode Return: mean={jnp.mean(returns)} std={jnp.std(returns, ddof=1)}")
    print(f"Episode Length: mean={jnp.mean(lengths)} std={jnp.std(lengths, ddof=1)}")

    print()

### ENJOY ###
from core.envs.utils import visualize_pygame

rngs = nnx.Rngs(0, params=1, env=5, actions=3, transitions=4)

FPS = round(1/DT)

visualize_pygame(
    rngs, env, training_state.actor, 
    fps=FPS, 
    verbose=False
)