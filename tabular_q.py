import jax.numpy as jnp
import jax
from jax.typing import ArrayLike

from chex import dataclass

from envs.flappy_bird import FlappyBirdEnv, Action, State
from utils import ReplayBuffer, LinearlyInterpolatedTable

DISCOUNT_RATE = 0.95
EPSILON = 0.1

LEARNING_RATE = 0.1

REPLAY_BUFFER_SIZE = 1000
BATCH_SIZE = 128 #32
TRAIN_FREQ = 1
N_ENVS = 16

TARGET_UPDATE_INTERVAL = 100

STEPS = 100000
LOG_INTERVAL = 1000

SEED = 2
key = jax.random.key(SEED)

DT = 0.1
env= FlappyBirdEnv(DT)

def get_observation(state):
    return jnp.array((state.bird_pos_y, state.bird_vel_y, state.pipe1_pos_x, 
        state.pipe1_pos_y, state.pipe2_pos_x, state.pipe2_pos_y))

@dataclass(frozen=True)
class Transition:
    cur_obs: ArrayLike
    action: Action
    reward: ArrayLike
    new_obs: ArrayLike

key, subkey = jax.random.split(key, 2)
dummy_state = env.reset(subkey)
dummy_transition = Transition(
    state = dummy_state,
    action = Action(flap=jnp.array(0)),
    reward = jnp.array(0),
    new_state = dummy_state
)

replay_buffer = ReplayBuffer(dummy_transition, REPLAY_BUFFER_SIZE)
replay_buffer_state = replay_buffer.init()

q_table = LinearlyInterpolatedTable(
    min=( 0,  -600, -60,  175, 300, 175 ), 
    max=( 800, 1500, 300, 625, 660, 625 ), 
    step=( 5,  50,   15,  5,   15,  5 )
) # bird_pos_y, bird_vel_y, pipe1_pos_x, pipe1_pos_y, pipe2_pos_x, pipe2_pos_y

policy_q_vals = jnp.array((q_table.init(0), q_table.init(0)))
target_q_vals = policy_q_vals

def get_greedy_action(observation):
    q_vals = jax.vmap(q_table.get, in_axes=[0, None])(policy_q_vals, observation)
    return jnp.argmax(q_vals)

def get_action(key, observation):
    do_greedy_key, random_action_key = jax.random.split(key)

    random_action = jax.random.randint(random_action_key, shape=(), minval=0, maxval=2)
    greedy_action = get_greedy_action(observation)

    return jnp.where(jax.random.uniform(do_greedy_key) < EPSILON, random_action, greedy_action)

def env_step(carry, _):
    key, env_state = carry

    key, action_key, step_key, reset_key = jax.random.split(key, 3)

    cur_obs = get_observation(env_state)
    action = get_action(action_key, cur_obs)
    new_state, reward, terminated = env.step(env_state, action, step_key)
    new_obs = get_observation(new_state)

    # reset env if terminated, don't otherwise
    next_state = jax.lax.cond(terminated, lambda: env.reset(reset_key), lambda: new_state)

    return (key, next_state), Transition(cur_obs=cur_obs, action=action, reward=reward, new_obs=new_obs)

## initialize ##

key, reset_key = jax.random.split(key, 2)
reset_keys = jax.random.split(reset_key, N_ENVS)

env_states = jax.vmap(env.reset)(reset_keys)

## sample transitions from environment ##

key, step_key = jax.random.split(key, 2)
step_keys = jax.random.split(step_key, N_ENVS)

carry, transitions = jax.lax.scan(jax.vmap(env_step), (step_keys, env_states), None, length=TRAIN_FREQ)
_, env_states = carry

transitions = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), transitions)
    # flatten to remove axis 0
replay_buffer_state = replay_buffer.insert(replay_buffer_state, transitions)

## train -- update policy q-table ##
key, sample_key = jax.random.split(key, 2)
sampled_transitions = replay_buffer.sample(replay_buffer_state, BATCH_SIZE, sample_key)

def update_q
next_q_vals = jax.vmap(q_table.get, in_axes=[0, None])(policy_q_vals, new_obs)
new_q = reward + DISCOUNT_RATE*jnp.max(next_q_vals)

old_q = q_table.get(policy_q_vals[action], cur_obs)
policy_q_vals = q_table.adjust(policy_q_vals[action], cur_obs, LEARNING_RATE * (new_q - old_q))


for step in range(STEPS):

    if (step + 1) % TARGET_UPDATE_INTERVAL == 0:
        target_q_table = np.array(policy_q_table) # hard update; copy policy to target


    if (step + 1) % LOG_INTERVAL == 0: # log
        print("Completed Steps: " + str(step + 1))
