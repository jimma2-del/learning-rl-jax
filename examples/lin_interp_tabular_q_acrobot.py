import jax
import jax.numpy as jnp

from gymnax.environments import Acrobot, CartPole
from core.envs.gymnax_wrapper import GymnaxWrapper, Space

from core.algos.linearly_interpolated_tabular_q import LinearlyInterpolatedTabularQ, TabularQHyperparameters
from core.utils import LinearlyInterpolatedTable

SEED = 2
key = jax.random.key(SEED)

class CustomGymnaxWrapper(GymnaxWrapper):
    def step(self, key, state, action):
        obs, state, reward, terminated, info = self.gymnax_env.step_env(key, state, action, self.gymnax_params)

        height = (-obs[0] - (obs[0] * obs[2] - obs[1] * obs[3])) / 2 # [-1, 0.5]
        h_d = ((obs[1] * (1 + obs[2]) + obs[0] * obs[3]) * obs[4] 
            + (obs[1] * obs[2] + obs[0] * obs[3]) * obs[5]) / 2

        reward = reward + (height - 1)/2 + jnp.abs(h_d)/5

        #jax.debug.print("reward={r} height={h} h_d={h_d}", r=reward, h=height, h_d=h_d)
        #print(f"reward={reward} height={height} h_d={h_d}")

        return state, reward, terminated, False, info

    def get_obs(self, key, state):
        gymnax_obs = self.gymnax_env.get_obs(state=state, params=self.gymnax_params, key=key)
        return jnp.array((
            jnp.atan2(gymnax_obs[1], gymnax_obs[0]), 
            jnp.atan2(gymnax_obs[3], gymnax_obs[2]), 
            gymnax_obs[4], gymnax_obs[5]
        ))

    @property
    def observation_space(self):
        return Space(
            low=jnp.array((-jnp.pi, -jnp.pi, -13, -29)),
            high=jnp.array((jnp.pi, jnp.pi, 13, 29))
        )

gymnax_env = Acrobot()
gymnax_env_params = gymnax_env.default_params
env = CustomGymnaxWrapper(gymnax_env, gymnax_env_params)

### TRAIN ###
Q_TABLE_GRIDPOINTS_PER_AXIS = 10

# q_table = LinearlyInterpolatedTable(
#     min=env.observation_space.low, 
#     max=env.observation_space.high, 
#     step=(env.observation_space.high - env.observation_space.low) / Q_TABLE_GRIDPOINTS_PER_AXIS
# )

# ACROBOT
# q_table = LinearlyInterpolatedTable(
#     min=(-1, -1, -1, -1, -13, -29), 
#     max=(1, 1, 1, 1, 13, 29), 
#     step=(0.2, 0.2, 0.2, 0.2, 2, 2)
# )
# q_table = LinearlyInterpolatedTable(
#     min=(-3.2, -3.2, -13, -29), 
#     max=(3.2, 3.2, 13, 29), 
#     step=(0.4, 0.4, 2, 4)
# )
q_table = LinearlyInterpolatedTable(
    min=(-3.2, -3.2, -6, -15), 
    max=(3.2, 3.2, 6, 15), 
    step=(0.2, 0.4, 0.25, 0.25)
)

# # CARTPOLE
# q_table = LinearlyInterpolatedTable(
#     min=(-2.4, -2.4, -0.2095, -2.4), 
#     max=(2.4, 2.4, 0.2095, 2.4), 
#     #step=(0.1, 0.1, 0.005, 0.05)
#     step=(0.2, 0.2, 0.02, 0.2)
# )

hyperparameters = TabularQHyperparameters(
    discount_rate = 0.98,
    learning_rate = 0.01,

    epsilon = 0.01,

    replay_buffer_size = 4096, #1024,
    batch_size = 256, #64,
    train_freq = 1,
    n_envs = 256, #64,

    target_update_interval = 4096, #512,
)

algo = LinearlyInterpolatedTabularQ(env, q_table, hyperparameters)

#q_vals = algo.init_q_table_vals()

STEPS = 1_000_000_000
LOG_INTERVAL_STEPS = 10_000_000

key, train_key = jax.random.split(key, 2)
q_vals = algo.train(train_key, STEPS, log_interval_steps=LOG_INTERVAL_STEPS)

### ENJOY ###

# NOTE: visualizer is broken in gymnax main branch; use PR https://github.com/RobertTLange/gymnax/pull/84
    # edit  _render_and_close() to remove 'with env:' statement to avoid closing pygame early
from gymnax.visualize import Visualizer

from gymnax.visualize.vis_gym import render_acrobot

import pygame
pygame.init()

DT = 0.2
FPS = round(1/DT)
clock = pygame.time.Clock()

states = []
rewards = []

key, reset_key = jax.random.split(key)
state, info = env.reset(reset_key)

steps = 0
MAX_STEPS = 500
truncated = False

# pygame.display.set_caption("Environment Visualization")
# screen = pygame.display.set_mode(render_acrobot(None, gymnax_env_params, state).swapaxes(0,1).shape[:2])

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            truncated = True

    states.append(state)

    # get agent action
    key, obs_key = jax.random.split(key, 2)
    obs = env.get_obs(obs_key, state)
    #print(obs)

    action = algo.get_greedy_action(q_vals, obs)

    # get human action
    # action = 1

    # keys = pygame.key.get_pressed()
    
    # if keys[pygame.K_a]:
    #     action -= 1
    # else:
    #     action += 1

    key, step_key = jax.random.split(key, 2)
    state, reward, terminated, truncated, info = env.step(step_key, state, action)
    rewards.append(reward)

    #print(reward)

    steps += 1

    # image_array = render_acrobot(None, gymnax_env_params, state)

    # pygame_surface = pygame.surfarray.make_surface(image_array.swapaxes(0,1))
    # screen.blit(pygame_surface, (0,0))
    # pygame.display.flip()

    # clock.tick(FPS)

    if terminated or truncated or steps >= MAX_STEPS:
        if terminated: print("terminated at steps=" + str(steps))
        break

cum_rewards = jnp.cumsum(jnp.array(rewards))
vis = Visualizer(gymnax_env, gymnax_env_params, states, cum_rewards)
vis.animate("./examples/lin_interp_tabular_q_acrobot_anim.gif")
#vis.animate(save_fname=None, view=True)