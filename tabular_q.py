import jax.numpy as jnp
import jax
from jax.typing import ArrayLike

from chex import dataclass
import functools

from envs.flappy_bird import FlappyBirdEnv, Action, State
from utils import ReplayBuffer, ReplayBufferState, LinearlyInterpolatedTable

@dataclass(frozen=True)
class Transition:
    cur_obs: ArrayLike
    action: ArrayLike
    reward: ArrayLike
    new_obs: ArrayLike

@dataclass(frozen=True)
class TrainingState:
    env_steps: ArrayLike
    prev_target_qs_update_steps: ArrayLike
    env_states: State
    policy_q_vals: ArrayLike
    target_q_vals: ArrayLike

DISCOUNT_RATE = 0.95
EPSILON = 0.05

LEARNING_RATE = 0.01

REPLAY_BUFFER_SIZE = 1000
BATCH_SIZE = 32
    # NOTE: we don't average the update (divide by batch size), so higher batch size -> higher learning rate
TRAIN_FREQ = 1
N_ENVS = 16#256

TARGET_UPDATE_INTERVAL = 1000

STEPS = 10_000_000
LOG_INTERVAL_STEPS = 1_000_000

SEED = 2
key = jax.random.key(SEED)

DT = 0.1
env = FlappyBirdEnv(DT)

def get_observation(state: State):
    use_pipe2 = state.pipe1_pos_x + env.settings.pipe_width/2 + env.settings.bird_size < env.settings.bird_pos_x

    pipe_pos_x = jnp.where(use_pipe2, state.pipe2_pos_x, state.pipe1_pos_x)
    pipe_pos_y = jnp.where(use_pipe2, state.pipe2_pos_y, state.pipe1_pos_y)

    pipe_dx = pipe_pos_x - env.settings.bird_pos_x
    pipe_dy = pipe_pos_y - state.bird_pos_y

    return jnp.array((state.bird_vel_y, pipe_dy))

key, subkey = jax.random.split(key, 2)
dummy_state = env.reset(subkey)
dummy_obs = get_observation(dummy_state)
dummy_transition = Transition(
    cur_obs = dummy_obs,
    action = jnp.array(0),
    reward = jnp.array(0),
    new_obs = dummy_obs
)

replay_buffer = ReplayBuffer(dummy_transition, REPLAY_BUFFER_SIZE)
replay_buffer_state = replay_buffer.init()

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

def get_greedy_action(observation, q_vals):
    q_vals = jax.vmap(q_table.get, in_axes=[0, None])(q_vals, observation)
    return jnp.argmax(q_vals)

def get_action(key, observation, q_vals):
    do_greedy_key, random_action_key = jax.random.split(key)

    random_action = jax.random.randint(random_action_key, shape=(), minval=0, maxval=2)
    greedy_action = get_greedy_action(observation, q_vals)

    return jnp.where(jax.random.uniform(do_greedy_key) < EPSILON, random_action, greedy_action)

@functools.partial(jax.jit, static_argnames=('iterations'))
def train_loop(key, training_state: TrainingState, replay_buffer_state: ReplayBufferState, iterations: int):
    def train_iteration(carry, _):
        key, training_state, replay_buffer_state = carry

        env_states = training_state.env_states
        env_steps = training_state.env_steps
        prev_target_qs_update_steps = training_state.prev_target_qs_update_steps
        policy_q_vals = training_state.policy_q_vals
        target_q_vals = training_state.policy_q_vals
        
        ## sample transitions from environment ##

        key, step_key = jax.random.split(key, 2)
        step_keys = jax.random.split(step_key, N_ENVS)

        def env_step(carry, _):
            key, env_state = carry

            key, action_key, step_key, reset_key = jax.random.split(key, 4)

            cur_obs = get_observation(env_state)
            action = get_action(action_key, cur_obs, policy_q_vals)
            new_state, reward, terminated = env.step(env_state, Action(flap=action), step_key)
            new_obs = get_observation(new_state)

            # reset env if terminated, don't otherwise
            next_state = jax.lax.cond(terminated, lambda: env.reset(reset_key), lambda: new_state)

            return (key, next_state), Transition(cur_obs=cur_obs, action=action, reward=reward, new_obs=new_obs)

        carry, transitions = jax.lax.scan(jax.vmap(env_step), (step_keys, env_states), None, length=TRAIN_FREQ)
        _, env_states = carry
        env_steps += TRAIN_FREQ * N_ENVS

        transitions = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), transitions)
            # flatten to remove axis 0
        replay_buffer_state = replay_buffer.insert(replay_buffer_state, transitions)

        ## update policy q-table ##

        key, sample_key = jax.random.split(key, 2)
        sampled_transitions = replay_buffer.sample(replay_buffer_state, BATCH_SIZE, sample_key)

        def update_q_get_corner_adjustments(transition):
            next_q_vals = jax.vmap(q_table.get, in_axes=[0, None])(target_q_vals, transition.new_obs)
            new_q = transition.reward + DISCOUNT_RATE*jnp.max(next_q_vals)

            old_q = q_table.get(policy_q_vals[transition.action], transition.cur_obs)
            adjust = LEARNING_RATE * (new_q - old_q)

            adjust_is, adjusts = q_table.adjust_get_corner_adjustments(policy_q_vals[transition.action], transition.cur_obs, adjust)

            return transition.action, adjust_is, adjusts

        actions, adjust_is, adjusts = jax.vmap(update_q_get_corner_adjustments)(sampled_transitions)

        # flatten to remove axis 0; duplicate values in actions to match
        actions = jnp.repeat(actions, adjust_is.shape[1])
        adjust_is = adjust_is.reshape(-1, *adjust_is.shape[2:])
        adjusts = adjusts.reshape(-1, *adjusts.shape[2:])

        policy_q_vals = policy_q_vals.at[(actions, ) + tuple(adjust_is.T)].add(adjusts)

        # update target_q_vals if enough steps have passed
        update_target_qs = env_steps - prev_target_qs_update_steps >= TARGET_UPDATE_INTERVAL
        target_q_vals = jnp.where(update_target_qs, policy_q_vals, target_q_vals)
        prev_target_qs_update_steps = jnp.where(update_target_qs, env_steps, prev_target_qs_update_steps)

        return (key, TrainingState(
            env_steps=env_steps,
            prev_target_qs_update_steps=prev_target_qs_update_steps,
            env_states=env_states,
            policy_q_vals=policy_q_vals,
            target_q_vals=target_q_vals,
        ), replay_buffer_state), None

    carry, _ = jax.lax.scan(train_iteration, (key, training_state, replay_buffer_state), length=iterations)
    key, training_state, replay_buffer_state = carry
    return training_state, replay_buffer_state

## initialize ##
key, reset_key = jax.random.split(key, 2)
reset_keys = jax.random.split(reset_key, N_ENVS)

env_states = jax.vmap(env.reset)(reset_keys)

init_q_vals = jnp.array((q_table.init(0), q_table.init(0)))

training_state = TrainingState(
    env_steps = jnp.array(0),
    prev_target_qs_update_steps = jnp.array(0),
    env_states = env_states,
    policy_q_vals = init_q_vals,
    target_q_vals = init_q_vals,
)

while training_state.env_steps < STEPS:
    key, train_key = jax.random.split(key, 2)
    training_state, replay_buffer_state = train_loop(train_key, training_state, replay_buffer_state, 
        LOG_INTERVAL_STEPS // (TRAIN_FREQ*N_ENVS))

    print(f"Possibly finished env_steps={training_state.env_steps}")


print(training_state.policy_q_vals)

### ENJOY ###
import numpy as np

import pygame, sys
pygame.init()

FPS = round(1/DT)
clock = pygame.time.Clock()

rng_key = jax.random.key(0)

rng_key, reset_key = jax.random.split(rng_key, 2)
state = env.reset(reset_key)

cur_return = 0
terminated = False
terminated_pause = 0

pygame.display.set_caption("Flappy Bird")
screen = pygame.display.set_mode((env.settings.window_size[1], env.settings.window_size[0]))

prev_flap_pressed = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
    
    if terminated_pause == 0:
        flap = get_greedy_action(get_observation(state), training_state.policy_q_vals)
        
        rng_key, step_key = jax.random.split(rng_key, 2)
        state, reward, terminated = env.step(state, Action(flap=flap), step_key)

        cur_return += reward

        if reward != 0:
            print(cur_return)

    image_array = np.array(env.render(state))
    pygame_surface = pygame.surfarray.make_surface(image_array.swapaxes(0,1))
    screen.blit(pygame_surface, (0,0))
    pygame.display.flip()

    if terminated and terminated_pause == 0:
        terminated_pause = 30

    if terminated and terminated_pause == 10:
        rng_key, reset_key = jax.random.split(rng_key, 2)
        state = env.reset(reset_key)
        cur_return = 0
        terminated = False

    if terminated_pause > 0:
        terminated_pause -= 1

    clock.tick(FPS)