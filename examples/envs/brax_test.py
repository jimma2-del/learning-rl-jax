import jax.numpy as jnp
import jax
import numpy as np

from flax import nnx

from brax.envs import create
from core.envs.brax import BraxWrapper

from core.envs.utils import rollout_episode, visualize_pygame
from core.envs.wrappers import JitWrapper

from brax.io import html

NUM_EPISODES = 1
ENV_NAME = "ant"
STEPS_LIMIT = 1000

rngs = nnx.Rngs(0, params=1, env=5, actions=3)

brax_env = create(ENV_NAME, auto_reset=False, batch_size=None, episode_length=STEPS_LIMIT)

env = BraxWrapper(brax_env)

#@nnx.jit
def policy(obs, rngs):
    return env.action_space.sample(rngs.actions())

VISUALIZE_METHOD = "pygame"

if VISUALIZE_METHOD == 'html':
    states = []

    for _ in range(NUM_EPISODES):
        timesteps, state, info = rollout_episode(rngs, JitWrapper(env), policy)

        eps_return = sum(timesteps.reward)
        steps = len(timesteps.reward)

        print(f"{'Truncated' if timesteps.truncated[-1] else 'Terminated'} at steps={steps}, return={eps_return}.")

        states += [ jax.tree.map(lambda x: x[i], timesteps.state.pipeline_state) for i in range(steps + 1) ]

    html_content = html.render(brax_env.sys, states)
    with open(f"./examples/envs/{ENV_NAME}_render.html", "w") as f:
        f.write(html_content)

elif VISUALIZE_METHOD == 'pygame':
    visualize_pygame(
        rngs, JitWrapper(env), policy, 
        fps=1.0 / brax_env.dt, 
        verbose=False
    )