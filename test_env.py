import jax.numpy as jnp
import jax
import numpy as np

from envs.flappy_bird import FlappyBirdEnv, Action, State

import pygame, sys
pygame.init()

DT = 0.1
FPS = 1#round(1/DT)
clock = pygame.time.Clock()

rng_key = jax.random.key(0)

env = FlappyBirdEnv(dt=DT)
rng_key, reset_key = jax.random.split(rng_key, 2)
state = env.reset(reset_key)

def get_observation(state: State):
    use_pipe2 = state.pipe1_pos_x + env.settings.pipe_width/2 + env.settings.bird_size < env.settings.bird_pos_x

    pipe_pos_x = jnp.where(use_pipe2, state.pipe2_pos_x, state.pipe1_pos_x)
    pipe_pos_y = jnp.where(use_pipe2, state.pipe2_pos_y, state.pipe1_pos_y)

    pipe_dx = pipe_pos_x - env.settings.bird_pos_x
    pipe_dy = pipe_pos_y - state.bird_pos_y

    return jnp.array((state.bird_vel_y, pipe_dy, pipe_dx))

cur_return = 0
terminated = False

pygame.display.set_caption("Flappy Bird")
screen = pygame.display.set_mode((env.settings.window_size[1], env.settings.window_size[0]))

prev_flap_pressed = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    if not terminated:
        keys = pygame.key.get_pressed()
        
        flap = False
        
        if keys[pygame.K_SPACE]:
            flap = not prev_flap_pressed
            prev_flap_pressed = True
        else:
            prev_flap_pressed = False
        
        rng_key, step_key = jax.random.split(rng_key, 2)
        state, reward, terminated = env.step(state, Action(flap=flap), step_key)

        print(get_observation(state))

        cur_return += reward

        if reward != 0:
            print(cur_return)

    image_array = np.array(env.render(state))
    pygame_surface = pygame.surfarray.make_surface(image_array.swapaxes(0,1))
    screen.blit(pygame_surface, (0,0))
    pygame.display.flip()

    clock.tick(FPS)