import time
from os import path

import jax
import jax.numpy as jnp

from jax.typing import ArrayLike

from flax import nnx
from optax import schedules

from mujoco_playground import registry
from core.envs.mujoco_playground import MuJoCoPlaygroundWrapper

from core.envs.wrappers import ObsRangeNormalizeWrapper, EpisodeStepCountWrapper, \
    JitWrapper, VmapWrapper, PrecomputedResetsPoolWrapper, ClipActionsToBoundsWrapper

from core.envs.utils import rollout_episode, visualize_pygame, evaluate_episodes

from core.algos import ppo

#jax.config.update("jax_log_compiles", True)

ENV_NAME = "WalkerRun"
N_ENVS = 2048
CAMERA = 'side' # None

rngs = nnx.Rngs(0, params=100, env=200, actions=300)

config = registry.get_default_config(ENV_NAME)
config.impl = 'jax' # 'warp' backend currently does not work
config.ctrl_dt = 0.05
config.sim_dt = 0.005

mjx_env = registry.load(ENV_NAME, config)

env = MuJoCoPlaygroundWrapper(mjx_env, {'camera': CAMERA})

RESETS_POOL_SIZE = 32768
resets_pool_states_infos = jax.vmap(env.reset)(jax.random.split(rngs.env(), RESETS_POOL_SIZE))
env = PrecomputedResetsPoolWrapper(env, resets_pool_states_infos)

#env = ClipActionsToBoundsWrapper(env)

### TRAIN ###

STEPS = 100_000_000 #1_000_000
LOG_INTERVAL_STEPS = 5_000_000 #100_000

MAX_STEPS = 200

EVAL_EPS = 256#2048 # 
EVAL_N_ENVS = 256#2048

hyperparameters = ppo.Hyperparameters(
    learning_rate = 2.5e-4,
    n_envs = N_ENVS,
    gae_lambda = 0.95,

    rollout_length = 32,
    n_minibatches = 8,#32, 
    n_epochs = 4,

    clip_epsilon = 0.2,

    vf_coef = 0.5, 
    ent_coef = 0.01,

    normalize_advantages = True,

    recompute_advantages = True,
    target_kl = 0.02
)

algo = ppo.PPO(EpisodeStepCountWrapper(VmapWrapper(env), max_eps_len=MAX_STEPS), hyperparameters)

training_state = algo.init_training_state(rngs)
train = nnx.jit(algo.train, static_argnames=('steps',))

@nnx.jit
def evaluate(rngs, policy):
    return evaluate_episodes(
        rngs, EpisodeStepCountWrapper(VmapWrapper(env), max_eps_len=MAX_STEPS), 
        nnx.vmap(lambda obs, rngs: algo.get_action(rngs, policy, obs, deterministic=True)), 
        EVAL_EPS, EVAL_N_ENVS
    )

while training_state.steps < STEPS:
    start_time = time.perf_counter()

    training_state, metrics = train(rngs, training_state, LOG_INTERVAL_STEPS)

    elasped_time = time.perf_counter() - start_time
    sps = LOG_INTERVAL_STEPS / elasped_time
    print(f"Completed steps={training_state.steps}; sps={sps:,.1f}")
    print("Metrics: " + " ".join([ f"{key}={val:.5g}" for key, val in metrics.items() ]))

    # eval
    returns, lengths = evaluate(rngs, training_state.policy)

    print(f"Episode Return: mean={jnp.mean(returns)} std={jnp.std(returns, ddof=1)}")
    print(f"Episode Length: mean={jnp.mean(lengths)} std={jnp.std(lengths, ddof=1)}")

    print()

# test save
import orbax.checkpoint as ocp

SAVE_PATH = path.abspath(f'examples/ppo/_tmp/{ENV_NAME}')

_, state = nnx.split(training_state.policy)
checkpointer_save = ocp.StandardCheckpointer()
checkpointer_save.save(SAVE_PATH, state)

### ENJOY ###
import mediapy

rngs = nnx.Rngs(0, params=1, env=5, actions=3)

def actor(obs, rngs):
    #print(training_state.policy(obs, rngs=rngs))
    return algo.get_action(rngs, training_state.policy, obs, deterministic=True)

EVAL_EPS = 256
returns, lengths = nnx.jit(evaluate_episodes, static_argnums=(1, 2, 3, 4, 5))(
    rngs, EpisodeStepCountWrapper(VmapWrapper(env), max_eps_len=MAX_STEPS), nnx.vmap(actor), EVAL_EPS, EVAL_N_ENVS)
print(returns)
print(lengths)
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