import jax

from envs.flappy_bird import FlappyBirdEnv, State

from algos.linearly_interpolated_tabular_q import LinearlyInterpolatedTabularQ, TabularQHyperparameters
from utils import LinearlyInterpolatedTable

SEED = 2
key = jax.random.key(SEED)

DT = 0.1
env = FlappyBirdEnv(DT)

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

hyperparameters = TabularQHyperparameters(
    discount_rate = 0.95,
    learning_rate = 0.01,

    epsilon = 0.05,

    replay_buffer_size = 1000,
    batch_size = 32,
    train_freq = 1,
    n_envs = 16, #256,

    target_update_interval = 1000,
)

algo = LinearlyInterpolatedTabularQ(env, q_table, hyperparameters)

STEPS = 10_000_000
LOG_INTERVAL_STEPS = 1_000_000

key, train_key = jax.random.split(key, 2)
q_vals = algo.train(train_key, STEPS, log_interval_steps=LOG_INTERVAL_STEPS)

### ENJOY ###
import numpy as np

import pygame, sys
pygame.init()

FPS = round(1/DT)
clock = pygame.time.Clock()

ENJOY_SEED = 0
key = jax.random.key(ENJOY_SEED)

key, reset_key = jax.random.split(key, 2)
state, info = env.reset(reset_key)

cur_return = 0
done = False
done_pause = 0

pygame.display.set_caption("Flappy Bird")
screen = pygame.display.set_mode((env.settings.window_size[1], env.settings.window_size[0]))

prev_flap_pressed = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
    
    if done_pause == 0:
        key, obs_key, step_key = jax.random.split(key, 3)
        action = algo.get_greedy_action(q_vals, env.get_obs(obs_key, state))
        state, reward, terminated, truncated, info = env.step(step_key, state, action)

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
        key, reset_key = jax.random.split(key, 2)
        state, info = env.reset(reset_key)
        cur_return = 0
        done = False

    if done_pause > 0:
        done_pause -= 1

    clock.tick(FPS)