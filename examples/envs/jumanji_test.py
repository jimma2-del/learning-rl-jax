import jax.numpy as jnp
import jax
import numpy as np

from flax import nnx

from jumanji.environments.routing.snake import Snake

from core.envs.jumanji import JumanjiWrapper
from core.envs.utils import rollout_episode

from core.envs.wrappers import EpisodeStepCountWrapper

NUM_EPISODES = 1
ENV_NAME = "snake"
STEPS_LIMIT = 600

rngs = nnx.Rngs(0, params=1, env=5, actions=3, transitions=4)

jumanji_env = Snake()
env = JumanjiWrapper(jumanji_env)

#@nnx.jit
def actor(obs, rngs):
    return env.action_space.sample(rngs.actions()), {}

comb_states = []

for _ in range(NUM_EPISODES):
    timesteps, state, info = rollout_episode(rngs, EpisodeStepCountWrapper(env, STEPS_LIMIT), actor)
    eps_return = sum(timesteps.reward)
    steps = len(timesteps.reward)

    print(f"{'Truncated' if timesteps.truncated[-1] else 'Terminated'} at steps={steps}, return={eps_return}.")

    comb_states += [ jax.tree.map(lambda x: x[i], timesteps.state.state) for i in range(steps + 1) ]

jumanji_env.animate(comb_states, 100, f"./examples/envs/{ENV_NAME}_animation.gif")