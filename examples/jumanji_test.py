import jax.numpy as jnp
import jax
import numpy as np

from flax import nnx

from jumanji.environments.routing.snake import Snake

from core.envs.jumanji_wrapper import JumanjiWrapper

NUM_EPISODES = 1
ENV_NAME = "snake"

rngs = nnx.Rngs(0, params=1, env=2, actions=3, transitions=4)

jumanji_env = Snake()
env = JumanjiWrapper(jumanji_env)

def policy(rngs, obs):
    return env.action_space.sample(rngs.actions())

states = []

for _ in range(NUM_EPISODES):
    eps_return = 0
    steps = 0

    terminated = False
    truncated = False

    state, info = env.reset(rngs.env())
    states.append(state)

    while not (terminated or truncated):
        obs = env.get_obs(rngs.env(), state)
        action = policy(rngs, obs)

        state, reward, terminated, truncated, info = env.step(rngs.env(), state, action)

        states.append(state)

        eps_return += reward
        steps += 1

    print(f"{'Terminated' if terminated else 'Truncated'} at steps={steps}, return={eps_return}.")

jumanji_env.animate(states, 100, f"./examples/{ENV_NAME}_animation.gif")