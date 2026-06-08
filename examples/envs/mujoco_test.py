import jax.numpy as jnp
import jax
import numpy as np

from flax import nnx

from mujoco_playground import registry

from core.envs.mujoco_playground import MuJoCoPlaygroundWrapper
from core.envs.utils import rollout_episode, visualize_pygame
from core.envs.wrappers import JitWrapper, EpisodeStepCountWrapper

import mediapy

NUM_EPISODES = 1
ENV_NAME = "CheetahRun"
STEPS_LIMIT = 1000

rngs = nnx.Rngs(0, params=1, env=5, actions=3)

config = registry.get_default_config(ENV_NAME)

#config.impl = 'jax'

# for the 'warp' backend (impl): maximum num of contacts in ALL (parallel) worlds
    # very big by default because it is for large batches
    # here, we don't parallelize, so we should reduce them drastically
    # to avoid being out of memory in rollout_episode
#config.naconmax = 240 

mjx_env = registry.load(ENV_NAME, config)

env = MuJoCoPlaygroundWrapper(mjx_env)

#@nnx.jit
def policy(obs, rngs):
    return env.action_space.sample(rngs.actions())

VISUALIZE_METHOD = "video"

FPS = 1.0 / mjx_env.dt

if VISUALIZE_METHOD == 'video':
    frames = []

    for _ in range(NUM_EPISODES):
        timesteps, state, info = rollout_episode(rngs, 
            JitWrapper(EpisodeStepCountWrapper(env, STEPS_LIMIT)), policy,

            # remove unnecessary warp `_impl` property, which takes up a lot of memory
            take_func = lambda ts: ts.replace(state=ts.state.replace(
                state=ts.state.state.tree_replace({ 'data._impl': None })))
        )

        eps_return = sum(timesteps.reward)
        steps = len(timesteps.reward)

        print(f"{'Truncated' if timesteps.truncated[-1] else 'Terminated'} at steps={steps}, return={eps_return}.")

        states = [ jax.tree.map(lambda x: x[i], timesteps.state.state) for i in range(steps + 1) ]
        frames += mjx_env.render(states, camera='side')

    mediapy.write_video(f"./examples/envs/{ENV_NAME}_render.mp4", frames, fps=FPS)

elif VISUALIZE_METHOD == 'pygame':
    visualize_pygame(
        rngs, JitWrapper(env), policy, 
        fps=FPS, 
        verbose=False
    )