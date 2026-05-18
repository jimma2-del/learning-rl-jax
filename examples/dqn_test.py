import jax
import jax.numpy as jnp

from jax.typing import ArrayLike

from flax import nnx
from optax import schedules

from gymnax.environments import Acrobot, CartPole
from core.envs.gymnax_wrapper import GymnaxWrapper, Space

from core.envs.flappy_bird import FlappyBirdEnv, State as FlappyBirdState

from core.envs.wrappers import Wrapper

from core.algos.dqn import DQN, DQNHyperparameters
from core.utils import LinearlyInterpolatedTable

#jax.config.update("jax_log_compiles", True)

rngs = nnx.Rngs(0, params=1, env=2, actions=3, transitions=4)

# gymnax_env = CartPole()
# gymnax_env_params = gymnax_env.default_params

# #env = CustomGymnaxWrapper(gymnax_env, gymnax_env_params)
# env = GymnaxWrapper(gymnax_env)

DT = 0.1
env = FlappyBirdEnv(DT)

class FlappyBirdNormalizeWrapper(Wrapper[FlappyBirdEnv, FlappyBirdState, jax.Array, ArrayLike, jax.Array]):
    def get_obs(self, key, state):
        obs = super().get_obs(key, state)

        return jnp.array(((obs[0] - 450) / 2100 * 2 * 2, obs[1] / 625 * 2))

env = FlappyBirdNormalizeWrapper(env)

### TRAIN ###

STEPS = 1_000_000#10_000_000
LOG_INTERVAL_STEPS = 100_000#1_000_000

hyperparameters = DQNHyperparameters(
    learning_rate = 1e-3,
    train_freq = 4,
    n_envs = 256,
    epsilon = schedules.linear_schedule(1, 0.1, 100_000),
)

algo = DQN(env, hyperparameters)

q_net = algo.train(rngs, STEPS, log_interval_steps=LOG_INTERVAL_STEPS)

### ENJOY ###

rngs = nnx.Rngs(0, params=1, env=2, actions=3, transitions=4)

# # NOTE: visualizer is broken in gymnax main branch; use PR https://github.com/RobertTLange/gymnax/pull/84
#     # edit  _render_and_close() to remove 'with env:' statement to avoid closing pygame early
# from gymnax.visualize import Visualizer

# from gymnax.visualize.vis_gym import render_acrobot

# import pygame
# pygame.init()

# DT = 0.2
# FPS = round(1/DT)
# clock = pygame.time.Clock()

# states = []
# rewards = []

# state, info = env.reset(rngs.env())

# steps = 0
# MAX_STEPS = 500
# truncated = False

# # pygame.display.set_caption("Environment Visualization")
# # screen = pygame.display.set_mode(render_acrobot(None, gymnax_env_params, state).swapaxes(0,1).shape[:2])

# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             truncated = True

#     states.append(state)

#     # get agent action
#     obs = env.get_obs(rngs.env(), state)
#     #print(obs)

#     action = algo.get_greedy_action(rngs, q_net, obs)

#     # get human action
#     # action = 1

#     # keys = pygame.key.get_pressed()
    
#     # if keys[pygame.K_a]:
#     #     action -= 1
#     # else:
#     #     action += 1

#     state, reward, terminated, truncated, info = env.step(rngs.env(), state, action)
#     rewards.append(reward)

#     #print(reward)

#     steps += 1

#     # image_array = render_acrobot(None, gymnax_env_params, state)

#     # pygame_surface = pygame.surfarray.make_surface(image_array.swapaxes(0,1))
#     # screen.blit(pygame_surface, (0,0))
#     # pygame.display.flip()

#     # clock.tick(FPS)

#     if terminated or truncated or steps >= MAX_STEPS:
#         if terminated: print("terminated at steps=" + str(steps))
#         break

# cum_rewards = jnp.cumsum(jnp.array(rewards))
# vis = Visualizer(gymnax_env, gymnax_env_params, states, cum_rewards)
# vis.animate("./examples/dqn_acrobot_anim.gif")
# #vis.animate(save_fname=None, view=True)

import numpy as np

import pygame, sys
pygame.init()

FPS = round(1/DT)
clock = pygame.time.Clock()

state, info = env.reset(rngs.env())

cur_return = 0
done = False
done_pause = 0

pygame.display.set_caption("Flappy Bird")
screen = pygame.display.set_mode((env.unwrapped.settings.window_size[1], env.unwrapped.settings.window_size[0]))

prev_flap_pressed = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
    
    if done_pause == 0:
        obs = env.get_obs(rngs.env(), state)
        q_vals = q_net(obs, rngs=rngs)
        print(q_vals)
        action = jnp.argmax(q_vals)
        state, reward, terminated, truncated, info = env.step(rngs.env(), state, action)

        cur_return += reward
        done = terminated or truncated

        if reward != 0:
            print(cur_return)

    image_array = np.array(env.render(state, 0))
    pygame_surface = pygame.surfarray.make_surface(image_array.swapaxes(0,1))
    screen.blit(pygame_surface, (0,0))
    pygame.display.flip()

    if done and done_pause == 0:
        done_pause = 30

    if done and done_pause == 10:
        state, info = env.reset(rngs.env())
        cur_return = 0
        done = False

    if done_pause > 0:
        done_pause -= 1

    clock.tick(FPS)