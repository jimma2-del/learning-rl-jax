import jax.numpy as jnp
from jax import random

import numpy as np

from core.envs.gridworld import GridworldEnv

from circular_buffer_np import CircularBufferNP

DISCOUNT_RATE = 0.95

EPSILON = 0.1
BATCH_SIZE = 32
TRAIN_FREQ = 4
TARGET_UPDATE_INTERVAL = 100
REPLAY_BUFFER_SIZE = 500
LEARNING_RATE = 0.1

STEPS = 10000
LOG_INTERVAL = 1000

SEED = 2
key = random.key(SEED)

env = GridworldEnv.built_in_map("general")

cur_state_replay_buffer = CircularBufferNP(np.zeros((REPLAY_BUFFER_SIZE, *env.observation_space.low.shape), dtype="int32"))
next_state_replay_buffer = CircularBufferNP(np.zeros((REPLAY_BUFFER_SIZE, *env.observation_space.low.shape), dtype="int32"))
action_replay_buffer = CircularBufferNP(np.zeros((REPLAY_BUFFER_SIZE, *env.action_space.low.shape), dtype="int32"))
reward_replay_buffer = CircularBufferNP(np.zeros((REPLAY_BUFFER_SIZE), dtype="int32"))

policy_q_table = np.zeros((*tuple(env.observation_space.high + 1), env.action_space.high + 1))
target_q_table = np.array(policy_q_table)

truncated = True

for step in range(STEPS):
    if truncated or terminated: #reset
        key, reset_key = random.split(key)
        state, info = env.reset(reset_key)
        terminated = False
        truncated = True

    key, obs_key = random.split(key)
    obs = env.get_obs(obs_key, state)

    # step thru env

    key, subkey = random.split(key)
    if random.uniform(subkey) < EPSILON: # random explore
        key, subkey = random.split(key)
        action = random.randint(subkey, shape=(), minval=0, maxval=4)
    else: # greedy
        action = np.argmax(policy_q_table[tuple(obs)])

    key, subkey = random.split(key)
    new_state, reward, terminated, truncated, info = env.step(subkey, state, action)
    key, obs_key = random.split(key)
    new_obs = env.get_obs(obs_key, new_state)

    cur_state_replay_buffer.add(obs)
    action_replay_buffer.add(action)
    reward_replay_buffer.add(reward)
    next_state_replay_buffer.add(new_obs)

    state = new_state

    if (step + 1) % TARGET_UPDATE_INTERVAL == 0:
        target_q_table = np.array(policy_q_table) # hard update; copy policy to target

    if (step + 1) % TRAIN_FREQ == 0: # update
        key, subkey = random.split(key)
        sampled_indices = random.choice(subkey, cur_state_replay_buffer.cur_size, 
            shape=(BATCH_SIZE,), replace=cur_state_replay_buffer.cur_size < BATCH_SIZE)

        for i in sampled_indices:
            cur_obs = cur_state_replay_buffer.get_arr()[i]
            next_obs = next_state_replay_buffer.get_arr()[i]
            action = action_replay_buffer.get_arr()[i]
            reward = reward_replay_buffer.get_arr()[i]

            q = reward + DISCOUNT_RATE*np.max(target_q_table[tuple(next_obs)])
            policy_q_table[(*cur_obs, action)] += \
                LEARNING_RATE * (q - policy_q_table[(*cur_obs, action)])

    if (step + 1) % LOG_INTERVAL == 0: # log
        print("Completed Steps: " + str(step + 1))

print(env.visualize_q_table(policy_q_table))
print(policy_q_table)