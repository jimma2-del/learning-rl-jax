import jax
import jax.numpy as jnp

from flax import nnx

from gymnax.environments import Acrobot, CartPole
from core.envs.gymnax_wrapper import GymnaxWrapper, Space

from core.algos.dqn import DQN, DQNHyperparameters
from core.utils import LinearlyInterpolatedTable

#jax.config.update("jax_log_compiles", True)

rngs = nnx.Rngs(0, params=1, env=2, actions=3, transitions=4)

# class CustomGymnaxWrapper(GymnaxWrapper):
#     def step(self, key, state, action):
#         obs, state, reward, terminated, info = self.gymnax_env.step_env(key, state, action, self.gymnax_params)

#         height = (-obs[0] - (obs[0] * obs[2] - obs[1] * obs[3])) / 2 # [-1, 0.5]
#         h_d = ((obs[1] * (1 + obs[2]) + obs[0] * obs[3]) * obs[4] 
#             + (obs[1] * obs[2] + obs[0] * obs[3]) * obs[5]) / 2

#         reward = reward + (height - 1)/2 + jnp.abs(h_d)/5

#         #jax.debug.print("reward={r} height={h} h_d={h_d}", r=reward, h=height, h_d=h_d)
#         #print(f"reward={reward} height={height} h_d={h_d}")

#         return state, reward, terminated, False, info

#     def get_obs(self, key, state):
#         gymnax_obs = self.gymnax_env.get_obs(state=state, params=self.gymnax_params, key=key)
#         return jnp.array((
#             jnp.atan2(gymnax_obs[1], gymnax_obs[0]), 
#             jnp.atan2(gymnax_obs[3], gymnax_obs[2]), 
#             gymnax_obs[4], gymnax_obs[5]
#         ), dtype=jnp.float32)

#     @property
#     def observation_space(self):
#         return Space(
#             low=jnp.array((-jnp.pi, -jnp.pi, -13, -29), dtype=jnp.float32),
#             high=jnp.array((jnp.pi, jnp.pi, 13, 29), dtype=jnp.float32)
#         )

gymnax_env = CartPole()
gymnax_env_params = gymnax_env.default_params

#env = CustomGymnaxWrapper(gymnax_env, gymnax_env_params)
env = GymnaxWrapper(gymnax_env)

### TRAIN ###

hyperparameters = DQNHyperparameters(
    learning_rate = 1e-3,
    train_freq = 1,
    n_envs = 256
)

algo = DQN(env, hyperparameters)

STEPS = 10_000_000
LOG_INTERVAL_STEPS = 1_000_000

q_net = algo.train(rngs, STEPS, log_interval_steps=LOG_INTERVAL_STEPS)

### ENJOY ###

rngs = nnx.Rngs(0, params=1, env=2, actions=3, transitions=4)

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

state, info = env.reset(rngs.env())

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
    obs = env.get_obs(rngs.env(), state)
    #print(obs)

    action = algo.get_greedy_action(rngs, q_net, obs)

    # get human action
    # action = 1

    # keys = pygame.key.get_pressed()
    
    # if keys[pygame.K_a]:
    #     action -= 1
    # else:
    #     action += 1

    state, reward, terminated, truncated, info = env.step(rngs.env(), state, action)
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
vis.animate("./examples/dqn_acrobot_anim.gif")
#vis.animate(save_fname=None, view=True)