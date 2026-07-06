import time
from os import path

import jax
import jax.numpy as jnp

from jax.typing import ArrayLike

from flax import nnx
from optax import schedules

from mujoco_playground import registry
from core.envs.mujoco_playground import MuJoCoPlaygroundWrapper

from core.envs.wrappers import EpisodeStepCountWrapper, JitWrapper, VmapWrapper, PrecomputedResetsPoolWrapper

from core.envs.utils import rollout_episode, visualize_pygame, evaluate_episodes

from core.algos import ppo

#jax.config.update("jax_log_compiles", True)

ENV_NAME = "WalkerRun"
MAX_STEPS = 500#100#1000
CAMERA = 'side'

rngs = nnx.Rngs(1, env=2, actions=3)

config = registry.get_default_config(ENV_NAME)

config.impl = 'jax' # compatibility with 'warp' backend is experimental
#config.naconmax = 50_000

config.ctrl_dt = 0.05
# config.sim_dt = 0.005

mjx_env = registry.load(ENV_NAME, config)

env = MuJoCoPlaygroundWrapper(mjx_env, {'camera': CAMERA})

RESETS_POOL_SIZE = 32768
resets_pool_states_infos = jax.vmap(env.reset)(jax.random.split(rngs.env(), RESETS_POOL_SIZE))
env = PrecomputedResetsPoolWrapper(env, resets_pool_states_infos)

algo = ppo.PPO(VmapWrapper(env))

import orbax.checkpoint as ocp

SAVE_PATH = path.abspath(f'examples/ppo/_tmp/{ENV_NAME}')

# test load
#abstract_model = nnx.eval_shape(lambda: algo.make_actor(rngs=nnx.Rngs(0), deterministic_sampling=True))

def make_actor():
    rngs = nnx.Rngs(0)

    obs_trunk = ppo.Networks.make_default_obs_trunk(env.observation_space)
    policy_head = ppo.Networks.make_default_policy_head(rngs, env.observation_space.flattened_dim, env.action_space,
        hidden_dims=(512, 256, 128), activation_func=nnx.swish)
    value_head = ppo.Networks.make_default_value_head(rngs, env.observation_space.flattened_dim,
        hidden_dims=(512, 256, 128), activation_func=nnx.swish)
    networks = ppo.Networks(obs_trunk=obs_trunk, policy_head=policy_head, value_head=value_head)

    return algo.make_actor(networks=networks, deterministic_sampling=True)

abstract_model = nnx.eval_shape(make_actor)

graphdef, abstract_state = nnx.split(abstract_model)
checkpointer_load = ocp.StandardCheckpointer()
state_restored = checkpointer_load.restore(SAVE_PATH, abstract_state)

actor = nnx.merge(graphdef, state_restored)

### ENJOY ###
import mediapy

rngs = nnx.Rngs(0, params=1, env=5, actions=3)

EVAL_EPS = 256
returns, lengths = nnx.jit(evaluate_episodes, static_argnums=(1, 3, 4))(
    rngs, EpisodeStepCountWrapper(VmapWrapper(env), max_eps_len=MAX_STEPS), actor, EVAL_EPS, EVAL_EPS)

print("Episode Returns:")
print(returns)
print()

print("Episode Lengths:")
print(lengths)
print()

print(f"Episode Return: mean={jnp.mean(returns)} std={jnp.std(returns, ddof=1)}")
print(f"Episode Length: mean={jnp.mean(lengths)} std={jnp.std(lengths, ddof=1)}")

VISUALIZE_METHOD = "video"
NUM_EPISODES = 1
rngs = nnx.Rngs(0, params=1, env=5, actions=3)

FPS = 1.0 / mjx_env.dt

if VISUALIZE_METHOD == 'video':
    frames = []

    for _ in range(NUM_EPISODES):
        timesteps, state, info = rollout_episode(rngs, 
            JitWrapper(EpisodeStepCountWrapper(env, MAX_STEPS)), actor,

            # remove unnecessary warp `_impl` property, which takes up a lot of memory
            take_func = lambda ts: ts.replace(state=ts.state.replace(
                state=ts.state.state.tree_replace({ 'data._impl': None })))
        )

        eps_return = sum(timesteps.reward)
        steps = len(timesteps.reward)

        print(f"{'Truncated' if timesteps.truncated[-1] else 'Terminated'} at steps={steps}, return={eps_return}.")

        states = [ jax.tree.map(lambda x: x[i], timesteps.state.state) for i in range(steps + 1) ]
        frames += mjx_env.render(states, camera=CAMERA)

    mediapy.write_video(f"./examples/ppo/visualizations/ppo_{ENV_NAME}.mp4", frames, fps=FPS)

elif VISUALIZE_METHOD == 'pygame':
    visualize_pygame(
        rngs, JitWrapper(env), actor, 
        fps=FPS, 
        verbose=False
    )