import jax.numpy as jnp
import jax
import numpy as np

from core.envs.flappy_bird import FlappyBirdEnv, State

import pygame, sys
pygame.init()

DT = 0.1
FPS = round(1/DT)
clock = pygame.time.Clock()

rng_key = jax.random.key(0)

env = FlappyBirdEnv(dt=DT)
rng_key, reset_key = jax.random.split(rng_key, 2)
state, info = env.reset(reset_key)

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
        
        rng_key, step_key, obs_key = jax.random.split(rng_key, 3)
        state, reward, terminated, truncated, info = env.step(step_key, state, flap)

        print(env.get_obs(obs_key, state))

        cur_return += reward

        if reward != 0:
            print(cur_return)

    else:
        rng_key, reset_key = jax.random.split(rng_key, 2)
        state, info = env.reset(reset_key)
        terminated = False

    image_array = np.array(env.render(state, 0))
    pygame_surface = pygame.surfarray.make_surface(image_array.swapaxes(0,1))
    screen.blit(pygame_surface, (0,0))
    pygame.display.flip()

    clock.tick(FPS)