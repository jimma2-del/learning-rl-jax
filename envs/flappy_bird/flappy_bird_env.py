from chex import dataclass
from jax.typing import ArrayLike
from jax import Array
import jax.numpy as jnp
import jax
import functools
import dataclasses

@dataclass(frozen=True)
class State:
    # positions refer to the CENTER
    bird_pos_y: ArrayLike
    bird_vel_y: ArrayLike

    pipe1_pos_x: ArrayLike
    pipe1_pos_y: ArrayLike

    pipe2_pos_x: ArrayLike
    pipe2_pos_y: ArrayLike

@dataclass(frozen=True)
class Action:
    flap: ArrayLike

@dataclass(frozen=True)
class Settings:
    window_size: tuple[int, int] = (800, 600) # height, width

    bird_size: int = 40

    pipe_width: int = 120
    pipe_gap_height: int = 250
    pipe_min_height: int = 50

    #dist_between_pipe_centers: int = 400

    bird_vel_x: int = 150 # per second
    bird_accel_y: int = 1500
    bird_flap_vel_y: int = -600

    bird_pos_x = 150

@dataclass(frozen=True)
class Rewards:
    pass_pipe: int = 1
    death: int = -1

@dataclass(frozen=True)
class RenderSettings:
    background_color = jnp.array((139, 212, 245), dtype=jnp.uint8)
    pipe_color = jnp.array((114, 197, 100), dtype=jnp.uint8)
    bird_color = jnp.array((253, 218, 66), dtype=jnp.uint8)

class FlappyBirdEnv:

    def __init__(self, dt=0.1, settings=Settings(), rewards=Rewards(), render_settings=RenderSettings()):
        self.DT = dt
        self.settings = settings
        self.rewards = rewards
        self.render_settings = render_settings

    @functools.partial(jax.jit, static_argnames=('self'))
    def reset(self, rng_key) -> State:
        pipe1_key, pipe2_key = jax.random.split(rng_key, 2)

        return State(
            bird_pos_y = self.settings.window_size[0] / 2,
            bird_vel_y = 0,

            pipe1_pos_x = self.settings.bird_pos_x - self.settings.pipe_width/2,
            pipe1_pos_y = self.gen_pipe_y(pipe1_key),

            pipe2_pos_x = self.settings.window_size[1]/2 + self.settings.bird_pos_x,
            pipe2_pos_y = self.gen_pipe_y(pipe2_key)
        )

    @functools.partial(jax.jit, static_argnames=('self'))
    def step(self, state: State, action: Action, rng_key) -> tuple[State, Array, Array]:
        '''returns new_state, reward, terminated'''

        # update velocities & positions
        new_bird_vel_y = jnp.where(action.flap != 0, 
            self.settings.bird_flap_vel_y,
            state.bird_vel_y + self.settings.bird_accel_y*self.DT
        )

        prev_pipe1_pos_x = state.pipe1_pos_x

        state = dataclasses.replace(state,
            bird_pos_y = state.bird_pos_y + new_bird_vel_y*self.DT,
            bird_vel_y = new_bird_vel_y,

            pipe1_pos_x = state.pipe1_pos_x - self.settings.bird_vel_x*self.DT,
            pipe2_pos_x = state.pipe2_pos_x - self.settings.bird_vel_x*self.DT,
        )

        # remove pipe and spawn in new pipe if left pipe (1) left screen
        pipe1_left_screen = state.pipe1_pos_x + self.settings.pipe_width/2 < 0

        state = dataclasses.replace(state,
            pipe1_pos_x = jnp.where(pipe1_left_screen, state.pipe2_pos_x, state.pipe1_pos_x),
            pipe1_pos_y = jnp.where(pipe1_left_screen, state.pipe2_pos_y, state.pipe1_pos_y),

            pipe2_pos_x = jnp.where(pipe1_left_screen, self.settings.window_size[1] + self.settings.pipe_width/2, state.pipe2_pos_x),
            pipe2_pos_y = jnp.where(pipe1_left_screen, self.gen_pipe_y(rng_key), state.pipe2_pos_y),
        )

        reward = 0
        terminated = False

        # check for collisions
        hit_bot = state.bird_pos_y + self.settings.bird_size/2 > self.settings.window_size[0]
        hit_top = state.bird_pos_y - self.settings.bird_size/2 < 0
        
        x_in_pipe = jnp.logical_and(
            self.settings.bird_pos_x + self.settings.bird_size/2 >= state.pipe1_pos_x - self.settings.pipe_width/2,
            self.settings.bird_pos_x - self.settings.bird_size/2 <= state.pipe1_pos_x + self.settings.pipe_width/2
        )

        y_in_pipe_gap = jnp.logical_and(
            state.bird_pos_y + self.settings.bird_size/2 <= state.pipe1_pos_y + self.settings.pipe_gap_height/2,
            state.bird_pos_y - self.settings.bird_size/2 >= state.pipe1_pos_y - self.settings.pipe_gap_height/2
        )

        hit_pipe = jnp.logical_and(x_in_pipe, jnp.logical_not(y_in_pipe_gap))

        hit = jnp.logical_or(jnp.logical_or(hit_bot, hit_top), hit_pipe)

        reward += jnp.where(hit, self.rewards.death, 0)
        terminated = hit

        # check if passed pipe
        passed_pipe = jnp.logical_and(prev_pipe1_pos_x >= self.settings.bird_pos_x, state.pipe1_pos_x < self.settings.bird_pos_x)
        reward += jnp.where(passed_pipe, self.rewards.pass_pipe, 0)

        return state, reward, terminated
    
    
    @functools.partial(jax.jit, static_argnames=('self'))
    def gen_pipe_y(self, rng_key):
        height_to_center = self.settings.pipe_min_height + self.settings.pipe_gap_height/2

        return jax.random.uniform(rng_key, shape=(), 
            minval=height_to_center, 
            maxval=self.settings.window_size[0] - height_to_center + 1
        )

    @functools.partial(jax.jit, static_argnames=('self'))
    def render(self, state: State) -> Array:
        image = jnp.full((*self.settings.window_size, 3), self.render_settings.background_color, dtype=jnp.uint8)

        # pipes
        pipes_pos = ((state.pipe1_pos_y, state.pipe1_pos_x), (state.pipe2_pos_y, state.pipe2_pos_x))

        for pipe_y, pipe_x in pipes_pos:
            y_vals, x_vals = jnp.mgrid[0:self.settings.window_size[0], 0:self.settings.window_size[1]]

            x_in_pipe = jnp.logical_and(
                x_vals <= pipe_x + self.settings.pipe_width/2,
                x_vals >= pipe_x - self.settings.pipe_width/2
            )

            y_in_pipe_gap = jnp.logical_and(
                y_vals <= pipe_y + self.settings.pipe_gap_height/2,
                y_vals >= pipe_y - self.settings.pipe_gap_height/2
            )

            pipe_mask = jnp.logical_and(x_in_pipe, jnp.logical_not(y_in_pipe_gap))
            
            image = jnp.where(pipe_mask[:, :, None], self.render_settings.pipe_color[None, None, :], image)

        # bird
        rect = jnp.full((self.settings.bird_size, self.settings.bird_size, 3), 
            self.render_settings.bird_color, dtype=jnp.uint8)

        bird_top = jnp.rint(state.bird_pos_y - self.settings.bird_size/2).astype(int)
        bird_left = jnp.rint(self.settings.bird_pos_x - self.settings.bird_size/2).astype(int)

        image = jax.lax.dynamic_update_slice(image, rect, (bird_top, bird_left, 0))

        return image