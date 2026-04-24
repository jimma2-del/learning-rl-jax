import jax.numpy as jnp
from jax import random

import numpy as np

from envs.maze import MazeEnv

from utils import CircularBufferNP

DISCOUNT_RATE = 0.95

EPSILON = 0.1
BATCH_SIZE = 32
TRAIN_FREQ = 4
TARGET_UPDATE_INTERVAL = 100
REPLAY_BUFFER_SIZE = 500
LEARNING_RATE = 0.1

STEPS = 1000000
LOG_INTERVAL = 10000

SEED = 2
key = random.key(SEED)

key, subkey = random.split(key)
env = MazeEnv(key=subkey)

input_shape = (env.observation_space.shape[0] + env.observation_space.shape[0], )

cur_state_replay_buffer = CircularBufferNP(np.zeros((REPLAY_BUFFER_SIZE, *input_shape), dtype="int32")) # *goal, *obs
next_state_replay_buffer = CircularBufferNP(np.zeros((REPLAY_BUFFER_SIZE, *input_shape), dtype="int32")) # *goal, *obs
action_replay_buffer = CircularBufferNP(np.zeros((REPLAY_BUFFER_SIZE, *env.action_space.shape), dtype="int32"))
reward_replay_buffer = CircularBufferNP(np.zeros((REPLAY_BUFFER_SIZE), dtype="int32"))

policy_q_table = np.zeros((*env.observation_space.nvec, *env.observation_space.nvec, env.action_space.n))
target_q_table = np.array(policy_q_table)

truncated = True
episode_transitions = [] # [ (obs, action, reward, new_obs) ]

for step in range(STEPS):
    if truncated or terminated: # episode ended; reset
        # do HER; relabel transitions and add to replay buffer

        if len(episode_transitions) != 0: # avoid first env reset
            hindsight_goal = episode_transitions[-1][3]

            for obs, action, reward, new_obs in episode_transitions:
                cur_state_replay_buffer.add(jnp.concatenate((hindsight_goal, obs)))
                action_replay_buffer.add(action)
                next_state_replay_buffer.add(jnp.concatenate((hindsight_goal, new_obs)))

                # re-compute reward with goal
                if jnp.all(hindsight_goal != env.GOAL_POS) and jnp.all(new_obs == hindsight_goal):
                    reward += env.GOAL_REWARD

                reward_replay_buffer.add(reward)

            episode_transitions = [] 

        # reset env and flags
        obs, info = env.reset()
        terminated = False
        truncated = True

    # step thru env

    key, subkey = random.split(key)
    if random.uniform(subkey) < EPSILON: # random explore
        key, subkey = random.split(key)
        action = random.randint(subkey, shape=(), minval=0, maxval=4)
    else: # greedy
        action = np.argmax(policy_q_table[(*env.GOAL_POS, *obs)])

    new_obs, reward, terminated, truncated, info = env.step(action)
    episode_transitions.append((obs, action, reward, new_obs))

    obs = new_obs

    if (step + 1) % TARGET_UPDATE_INTERVAL == 0:
        target_q_table = np.array(policy_q_table) # hard update; copy policy to target

    if (step + 1) % TRAIN_FREQ == 0 and cur_state_replay_buffer.cur_size != 0: # update
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

for y, row in enumerate(policy_q_table[tuple(env.GOAL_POS)]):

    for x, tile in enumerate(row):
        if env.TILE_IS_END[y,x] or not env.TILE_IS_PASSABLE[y,x]:
            result += "@"
        else:
            best_action = np.argmax(tile)
            result += ( "^", "v", "<", ">" )[best_action]

        result += " "

    result += "\n"

print(result)
#print(policy_q_table[tuple(env.GOAL_POS)])
print(policy_q_table)