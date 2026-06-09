import time

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

from core.algos import linearly_interpolated_tabular_q
from core.utils import LinearlyInterpolatedTable

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

# q_table = LinearlyInterpolatedTable(
#     min=( 0,  -600, -60,  175, 300, 175 ), 
#     max=( 800, 1500, 300, 625, 660, 625 ), 
#     step=( 25, 100,   30,  25,  30,  25 )
#     #step=( 5,  50,   15,  5,   15,  5 )
# ) # bird_pos_y, bird_vel_y, pipe1_pos_x, pipe1_pos_y, pipe2_pos_x, pipe2_pos_y

q_table = LinearlyInterpolatedTable(
    min=( -600, -625 ), 
    max=( 1500,  625 ), 
    step=(  50,    5 )
) # bird_vel_y, pipe_dy

rngs = nnx.Rngs(0, params=1, env=2, actions=3, transitions=4)

STEPS = 2_000_000
LOG_INTERVAL_STEPS = 200_000
EVAL_EPS = 32

hyperparameters = linearly_interpolated_tabular_q.Hyperparameters(
    discount_rate = 0.95,
    learning_rate = 0.1,

    epsilon = schedules.linear_schedule(1, 0.05, 0.1*STEPS),

    replay_buffer_size = 100_000,
    batch_size = 32,
    train_freq = 4,
    n_envs = 32, #256,

    target_update_interval = 1000,
)

algo = linearly_interpolated_tabular_q.LinearlyInterpolatedTabularQ(VmapWrapper(env), q_table, hyperparameters)

training_state = algo.init_training_state(rngs)

@nnx.jit
def evaluate(rngs, policy):
    return evaluate_episodes(
        rngs, VmapWrapper(env), 
        nnx.vmap(lambda obs, rngs: algo.get_action(rngs, policy, obs)), 
        EVAL_EPS, hyperparameters.n_envs
    )

while training_state.steps < STEPS:
    start_time = time.perf_counter()

    training_state, metrics = algo.train_epoch(rngs, training_state, LOG_INTERVAL_STEPS)

    elasped_time = time.perf_counter() - start_time
    sps = LOG_INTERVAL_STEPS / elasped_time
    print(f"Completed steps={training_state.steps}; sps={sps:,.1f}")
    print("Metrics: " + " ".join([ f"{key}={val}" for key, val in metrics.items() ]))

    # eval
    returns, lengths = evaluate(rngs, training_state.policy)

    print(f"Episode Return: mean={jnp.mean(returns)} std={jnp.std(returns, ddof=1)}")
    print(f"Episode Length: mean={jnp.mean(lengths)} std={jnp.std(lengths, ddof=1)}")

    print()

### ENJOY ###
from core.envs.utils import visualize_pygame

rngs = nnx.Rngs(0, params=1, env=5, actions=3, transitions=4)

def policy(obs, rngs):
    #print(jax.vmap(algo.q_table.get, in_axes=[0, None])(training_state.policy, obs))
    return algo.get_action(rngs, training_state.policy, obs)

FPS = round(1/DT)

visualize_pygame(
    rngs, env, policy, 
    fps=FPS, 
    verbose=False
)