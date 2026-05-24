import jax
from jax.typing import ArrayLike
import chex

import jax.numpy as jnp

from optax import schedules

from core.envs.flappy_bird import FlappyBirdEnv, State

from core.envs.wrappers import Wrapper
from core.envs.base import Space

from core.algos.linearly_interpolated_tabular_q import LinearlyInterpolatedTabularQ, TabularQHyperparameters
from core.utils import LinearlyInterpolatedTable

SEED = 2
key = jax.random.key(SEED)

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

STEPS = 10_000_000
LOG_INTERVAL_STEPS = 1_000_000

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

algo = LinearlyInterpolatedTabularQ(env, q_table, hyperparameters)

key, train_key = jax.random.split(key, 2)
q_vals = algo.train(train_key, STEPS, log_interval_steps=LOG_INTERVAL_STEPS)

### ENJOY ###
from flax import nnx
from core.envs.utils import visualize_pygame

rngs = nnx.Rngs(0, params=1, env=5, actions=3, transitions=4)

def policy(rngs, obs):
    return algo.get_greedy_action(q_vals, obs)

FPS = round(1/DT)
window_size = (env.settings.window_size[1], env.settings.window_size[0])

visualize_pygame(
    rngs, env, policy, 
    window_size, FPS, 
    verbose=False
)