import jax.numpy as jnp
from jax import random

import numpy as np

from envs.maze import MazeEnv

from utils import CircularBuffer

DISCOUNT_RATE = 0.95

EPSILON = 0.1
BATCH_SIZE = 32
TRAIN_FREQ = 4
TARGET_UPDATE_INTERVAL = 100
REPLAY_BUFFER_SIZE = 500
LEARNING_RATE = 0.1

STEPS = 100000
LOG_INTERVAL = 10000

SEED = 2
key = random.key(SEED)

key, subkey = random.split(key)
env = MazeEnv(key=subkey)

cur_state_replay_buffer = CircularBuffer(np.zeros((REPLAY_BUFFER_SIZE, *env.observation_space.shape), dtype="int32"))
next_state_replay_buffer = CircularBuffer(np.zeros((REPLAY_BUFFER_SIZE, *env.observation_space.shape), dtype="int32"))
action_replay_buffer = CircularBuffer(np.zeros((REPLAY_BUFFER_SIZE, *env.action_space.shape), dtype="int32"))
reward_replay_buffer = CircularBuffer(np.zeros((REPLAY_BUFFER_SIZE), dtype="int32"))

policy_q_table = np.zeros((*MazeEnv.MAP_SHAPE, 4))
target_q_table = np.array(policy_q_table)

truncated = True

for step in range(STEPS):
    if truncated or terminated: #reset
        obs, info = env.reset()
        terminated = False
        truncated = True

    # step thru env

    key, subkey = random.split(key)
    if random.uniform(subkey) < EPSILON: # random explore
        key, subkey = random.split(key)
        action = random.randint(subkey, shape=(), minval=0, maxval=4)
    else: # greedy
        action = np.argmax(policy_q_table[tuple(obs)])

    new_obs, reward, terminated, truncated, info = env.step(action)

    cur_state_replay_buffer.add(obs)
    action_replay_buffer.add(action)
    reward_replay_buffer.add(reward)
    next_state_replay_buffer.add(new_obs)

    obs = new_obs

    if (step + 1) % TARGET_UPDATE_INTERVAL == 0:
        target_q_table = np.array(policy_q_table) # hard update; copy policy to target

    if (step + 1) % TRAIN_FREQ == 0: # update
        key, subkey = random.split(key)
        sampled_indices = random.choice(subkey, cur_state_replay_buffer.cur_size, 
            shape=(BATCH_SIZE,), replace=cur_state_replay_buffer.cur_size < BATCH_SIZE)

        for i in sampled_indices:
            cur_state = cur_state_replay_buffer.get_arr()[i]
            next_state = next_state_replay_buffer.get_arr()[i]
            action = action_replay_buffer.get_arr()[i]
            reward = reward_replay_buffer.get_arr()[i]

            q = reward + DISCOUNT_RATE*np.max(target_q_table[tuple(next_state)])
            policy_q_table[(*cur_state, action)] += \
                LEARNING_RATE * (q - policy_q_table[(*cur_state, action)])

    if (step + 1) % LOG_INTERVAL == 0: # log
        print("Completed Steps: " + str(step + 1))

result = ""

for y, row in enumerate(policy_q_table):

    for x, tile in enumerate(row):
        if env.TILE_IS_END[y,x] or not env.TILE_IS_PASSABLE[y,x]:
            result += "@"
        else:
            best_action = np.argmax(tile)
            result += ( "^", "v", "<", ">" )[best_action]

        result += " "

    result += "\n"

print(result)
print(policy_q_table)