from ..envs.maze import MazeEnv
from jax import random

env = MazeEnv(random.key(10))

for i in range(5):
    obs, info = env.reset()
    print(env.render())

    eps_reward = 0

    done = False

    while not done:
        action = { "u": 0, "d": 1, "l": 2, "r": 3 }[input("> ")]

        obs, reward, terminated, truncated, info = env.step(action)
        eps_reward += reward
        done = truncated or terminated

        print()
        print("Episode Reward: " + str(eps_reward))
        print(env.render())

    print("=================================")
    