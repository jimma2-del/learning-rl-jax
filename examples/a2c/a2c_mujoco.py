import time
from os import path

import jax
import jax.numpy as jnp

from jax.typing import ArrayLike

from flax import nnx
from optax import schedules

from mujoco_playground import registry
from core.envs.mujoco_playground import MuJoCoPlaygroundWrapper

from core.envs.wrappers import ObsRangeNormalizeWrapper, EpisodeStepCountWrapper, JitWrapper

from core.envs.utils import rollout_episode, visualize_pygame, evaluate_episodes

from core.algos import a2c

#jax.config.update("jax_log_compiles", True)

rngs = nnx.Rngs(0, params=1, env=2, actions=3)

ENV_NAME = "WalkerWalk"

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

### TRAIN ###

STEPS = 1_000_000
LOG_INTERVAL_STEPS = 100_000
EVAL_EPS = 32

hyperparameters = a2c.Hyperparameters(
    learning_rate = 2.5e-4,#10e-4,
    n_envs = 32,
    n_steps = 5,
    ent_coef = 0.001
)

algo = a2c.A2C(env, hyperparameters)

training_state = algo.init_training_state(rngs)

while training_state.steps < STEPS:
    start_time = time.perf_counter()

    training_state, metrics = algo.train_epoch(rngs, training_state, LOG_INTERVAL_STEPS)

    elasped_time = time.perf_counter() - start_time
    sps = LOG_INTERVAL_STEPS / elasped_time
    print(f"Completed steps={training_state.steps}; sps={sps:,.1f}")
    print("Metrics: " + " ".join([ f"{key}={val}" for key, val in metrics.items() ]))

    # eval
    returns, lengths = nnx.jit(evaluate_episodes, static_argnums=(1, 2, 3, 4, 5))(
        rngs, env, 
        lambda obs, rngs: algo.get_action(rngs, training_state.policy, obs, deterministic=True), 
        EVAL_EPS, hyperparameters.n_envs
    )

    print(f"Episode Return: mean={jnp.mean(returns)} std={jnp.std(returns, ddof=1)}")
    print(f"Episode Length: mean={jnp.mean(lengths)} std={jnp.std(lengths, ddof=1)}")

    print()

# test save
#import orbax.checkpoint as ocp

# SAVE_PATH = path.abspath('examples/dqn/_tmp/flappybird')

# _, state = nnx.split(q_net)
# checkpointer_save = ocp.StandardCheckpointer()
# checkpointer_save.save(SAVE_PATH, state)


### ENJOY ###
import mediapy

rngs = nnx.Rngs(0, params=1, env=5, actions=3)

def policy(obs, rngs):
    print(training_state.policy(obs, rngs=rngs))
    return algo.get_action(rngs, training_state.policy, obs, deterministic=True)

MAX_STEPS = 500

EVAL_EPS = 256
returns, lengths = nnx.jit(evaluate_episodes, static_argnums=(1, 2, 3, 4, 5))(
    rngs, env, policy, EVAL_EPS, hyperparameters.n_envs)
print(returns)
print(lengths)
print(f"Episode Return: mean={jnp.mean(returns)} std={jnp.std(returns, ddof=1)}")
print(f"Episode Length: mean={jnp.mean(lengths)} std={jnp.std(lengths, ddof=1)}")

VISUALIZE_METHOD = "gif"
NUM_EPISODES = 1
STEPS_LIMIT = 1000
rngs = nnx.Rngs(0, params=1, env=5, actions=3)

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
        frames += mjx_env.render(states)

    mediapy.write_video(f"./examples/envs/{ENV_NAME}_render.mp4", frames, fps=FPS)

elif VISUALIZE_METHOD == 'pygame':
    visualize_pygame(
        rngs, JitWrapper(env), policy, 
        fps=FPS, 
        verbose=False
    )