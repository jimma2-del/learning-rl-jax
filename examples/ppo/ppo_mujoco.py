from typing import Mapping

import time
from os import path

import jax
import jax.numpy as jnp

from jax.typing import ArrayLike

from flax import nnx
from optax import schedules

from mujoco_playground import registry
from core.envs.mujoco_playground import MuJoCoPlaygroundWrapper

from core.envs.wrappers import EpisodeStepCountWrapper, JitWrapper, VmapWrapper, \
    PrecomputedResetsPoolWrapper, ClipActionsToBoundsWrapper, AutoResetWrapper

from core.envs.utils import rollout_episode, visualize_pygame, evaluate_episodes, stagger_env_states

from core.algos import ppo

from core.utils.nnx_modules import MLP, RunningMeanVarNorm, ActionDistributionHead, Pipe
from core.utils.batch_utils import flatten_batched_tree, get_tree_flattened_dim

#jax.config.update("jax_log_compiles", True)

ENV_NAME = "G1JoystickRoughTerrain"
MAX_STEPS = 1000
N_ENVS = 2048
CAMERA = 'track'#'side' # None

rngs = nnx.Rngs(0, params=1, env=2, actions=3, optimize_samples=4)

config = registry.get_default_config(ENV_NAME)

config.impl = 'warp' #'jax' # compatibility with 'warp' backend is experimental
#config.naconmax = 50_000
#config.njmax = 32 # for SpotFlatTerrainJoystick

# config.ctrl_dt = 0.05
# config.sim_dt = 0.005

mjx_env = registry.load(ENV_NAME, config)

env = MuJoCoPlaygroundWrapper(mjx_env, {'camera': CAMERA})

RESETS_POOL_SIZE = 2048 - 1#32768 - 1 # subtract one to avoid being the same as nacomax
resets_pool_states_infos = jax.vmap(env.reset)(jax.random.split(rngs.env(), RESETS_POOL_SIZE))
env = PrecomputedResetsPoolWrapper(env, resets_pool_states_infos)

#env = ClipActionsToBoundsWrapper(env)

### TRAIN ###
STEPS = 200_000_000 #1_000_000

EVAL_EPS = 256
EVAL_INTERVAL = 100_000
N_LOGS_PER_EVAL = 10

hyperparameters = ppo.Hyperparameters(
    discount_rate = 0.97,

    learning_rate = schedules.cosine_decay_schedule(3e-4, STEPS), #2.5e-4,
    n_envs = N_ENVS,
    gae_lambda = 0.95,

    rollout_length = 32,
    n_minibatches = 32, 
    n_epochs = 8,

    clip_epsilon = 0.2,

    vf_coef = 0.5, 
    ent_coef = 0.01,

    normalize_advantages = True,

    recompute_advantages = True,
    target_kl = 0.02,

    truncated_frac = 1.1 / MAX_STEPS
)

wrapped_env = EpisodeStepCountWrapper(VmapWrapper(env), max_eps_len=MAX_STEPS)
algo = ppo.PPO(wrapped_env, hyperparameters)

# custom networks are needed to allow for asymmetric actor-critic
    # policy net only gets sensor data (partial observation); critic gets full (privileged) state
obs_trunk = RunningMeanVarNorm(env.observation_space.shapes_dtypes)

try_enter = lambda x, key: x[key] if isinstance(x, Mapping) and key in x else x

policy_obs_sdt = try_enter(env.observation_space.shapes_dtypes, 'state')
policy_head = ActionDistributionHead(env.action_space)
policy_head = Pipe(
    lambda x: try_enter(x, 'state'),
    lambda x: flatten_batched_tree(policy_obs_sdt, x),
    MLP(rngs, (get_tree_flattened_dim(policy_obs_sdt), 512, 256, 128, policy_head.input_dim), 
        activation_func=nnx.swish), 
    policy_head
)

value_func_obs_sdt = try_enter(env.observation_space.shapes_dtypes, 'privileged_state')
value_head = Pipe(
    lambda x: try_enter(x, 'privileged_state'),
    lambda x: flatten_batched_tree(value_func_obs_sdt, x),
    MLP(rngs, (get_tree_flattened_dim(value_func_obs_sdt), 512, 256, 128, 1), 
        activation_func=nnx.swish),
    lambda x: jnp.squeeze(x, axis=-1)
)

networks = ppo.Networks(obs_trunk=obs_trunk, policy_head=policy_head, value_head=value_head)

training_state = algo.init_training_state(rngs, networks=networks)
train = nnx.jit(algo.train, static_argnames=('steps',), donate_argnames=('training_state'))

# jitted_stagger = nnx.jit(stagger_env_states, 
#     static_argnames=('env', 'n_envs', 'stagger_step_size'), donate_argnames=('initial_env_states'))
# training_state.env_states = jitted_stagger(rngs, AutoResetWrapper(wrapped_env), hyperparameters.n_envs, 
#     stagger_step_size=1, initial_env_states=training_state.env_states)

@nnx.jit
def evaluate(rngs, actor):
    return evaluate_episodes(
        rngs, wrapped_env, actor, 
        EVAL_EPS, EVAL_EPS
    )

while training_state.steps < STEPS:
    start_time = time.perf_counter()
    training_state, metrics = train(rngs, training_state, EVAL_INTERVAL)
    elasped_time = time.perf_counter() - start_time

    print()

    avg_metrics = jax.tree.map(lambda x: list(map(jnp.mean, jnp.array_split(x, N_LOGS_PER_EVAL))), metrics)
    steps = avg_metrics.pop('steps')
    for i in range(N_LOGS_PER_EVAL):
        print(f"Step {steps[i]:.0f}: " + " ".join([ f"{key}={val[i]:.5g}" for key, val in avg_metrics.items() ]))

    print()

    sps = EVAL_INTERVAL / elasped_time
    print(f"COMPLETED steps={training_state.steps}; sps={sps:,.1f}")

    # eval
    actor = algo.make_actor(training_state.networks, deterministic_sampling=True)
    returns, lengths = evaluate(rngs, actor)

    print(f"Episode Return: mean={jnp.mean(returns)} std={jnp.std(returns, ddof=1)}")
    print(f"Episode Length: mean={jnp.mean(lengths)} std={jnp.std(lengths, ddof=1)}")

print()

# test save
import orbax.checkpoint as ocp

SAVE_PATH = path.abspath(f'examples/ppo/_tmp/{ENV_NAME}')

_, state = nnx.split(actor)
checkpointer_save = ocp.StandardCheckpointer()
checkpointer_save.save(SAVE_PATH, state)

### ENJOY ###
import mediapy

rngs = nnx.Rngs(0, params=1, env=5, actions=3)

EVAL_EPS = 256
returns, lengths = nnx.jit(evaluate_episodes, static_argnums=(1, 3, 4))(
    rngs, wrapped_env, actor, EVAL_EPS, EVAL_EPS)

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