import time
from os import path

import jax
import jax.numpy as jnp

from jax.typing import ArrayLike

from flax import nnx
from optax import schedules

from mujoco_playground import registry
from core.envs.mujoco_playground import MuJoCoPlaygroundWrapper

from core.envs.wrappers import ObsRangeNormalizeWrapper, EpisodeStepCountWrapper, JitWrapper, VmapWrapper

from core.envs.utils import rollout_episode, visualize_pygame, evaluate_episodes

from core.algos import a2c

#jax.config.update("jax_log_compiles", True)

rngs = nnx.Rngs(0, params=1, env=2, actions=3)

ENV_NAME = "CheetahRun"
N_ENVS = 256
EVAL_EPS = 256
MAX_STEPS = 500

CAMERA = None#'side'

config = registry.get_default_config(ENV_NAME)
config.impl = 'jax' # 'warp' backend currently does not work

mjx_env = registry.load(ENV_NAME, config)

env = MuJoCoPlaygroundWrapper(mjx_env, { 'camera': CAMERA })

algo = a2c.A2C(VmapWrapper(env))

import orbax.checkpoint as ocp

SAVE_PATH = path.abspath(f'examples/a2c/_tmp/{ENV_NAME}')

# test load
abstract_model = nnx.eval_shape(lambda: algo.create_default_policy(rngs=nnx.Rngs(0)))
graphdef, abstract_state = nnx.split(abstract_model)
checkpointer_load = ocp.StandardCheckpointer()
state_restored = checkpointer_load.restore(SAVE_PATH, abstract_state)

policy_state = nnx.merge(graphdef, state_restored)

### ENJOY ###
import mediapy

rngs = nnx.Rngs(0, params=1, env=5, actions=3)

def policy(obs, rngs):
    print(policy_state(obs, rngs=rngs))
    return algo.get_action(rngs, policy_state, obs, deterministic=True)

returns, lengths = nnx.jit(evaluate_episodes, static_argnums=(1, 2, 3, 4, 5))(
    rngs, EpisodeStepCountWrapper(VmapWrapper(env), max_eps_len=MAX_STEPS), nnx.vmap(policy), EVAL_EPS, N_ENVS)
print(returns)
print(lengths)
print(f"Episode Return: mean={jnp.mean(returns)} std={jnp.std(returns, ddof=1)}")
print(f"Episode Length: mean={jnp.mean(lengths)} std={jnp.std(lengths, ddof=1)}")

VISUALIZE_METHOD = "pygame"
NUM_EPISODES = 1
rngs = nnx.Rngs(0, params=1, env=5, actions=3)

FPS = 1.0 / mjx_env.dt

if VISUALIZE_METHOD == 'video':
    frames = []

    for _ in range(NUM_EPISODES):
        timesteps, state, info = rollout_episode(rngs, 
            JitWrapper(EpisodeStepCountWrapper(env, MAX_STEPS)), policy,

            # remove unnecessary warp `_impl` property, which takes up a lot of memory
            take_func = lambda ts: ts.replace(state=ts.state.replace(
                state=ts.state.state.tree_replace({ 'data._impl': None })))
        )

        eps_return = sum(timesteps.reward)
        steps = len(timesteps.reward)

        print(f"{'Truncated' if timesteps.truncated[-1] else 'Terminated'} at steps={steps}, return={eps_return}.")

        states = [ jax.tree.map(lambda x: x[i], timesteps.state.state) for i in range(steps + 1) ]
        frames += mjx_env.render(states, camera=CAMERA)

    mediapy.write_video(f"./examples/a2c/visualizations/a2c_{ENV_NAME}.mp4", frames, fps=FPS)

elif VISUALIZE_METHOD == 'pygame':
    visualize_pygame(
        rngs, JitWrapper(env), policy, 
        fps=FPS, 
        verbose=False
    )