import time
from os import path

import jax
import jax.numpy as jnp

from jax.typing import ArrayLike

from flax import nnx
from optax import schedules

from gymnax.environments import Acrobot, CartPole, MinBreakout, Reacher, Swimmer
from core.envs.gymnax import GymnaxWrapper

from core.envs.flappy_bird import FlappyBirdEnv

from core.envs.wrappers import EpisodeStepCountWrapper, VmapWrapper

from core.envs.utils import rollout_episode, visualize_pygame, evaluate_episodes

from core.algos import a2c

#jax.config.update("jax_log_compiles", True)

rngs = nnx.Rngs(0, params=1, env=2, actions=3)

## Gymnax

gymnax_env = Acrobot()
gymnax_env_params = gymnax_env.default_params

env = GymnaxWrapper(gymnax_env)

## Flappy Bird

# DT = 0.1
# env = FlappyBirdEnv(DT)

### TRAIN ###

STEPS = 1_000_000

EVAL_EPS = 256
EVAL_INTERVAL = 100_000
N_LOGS_PER_EVAL = 4

hyperparameters = a2c.Hyperparameters(
    learning_rate = 2.5e-4,#10e-4,
    n_envs = 256,
    rollout_length = 5,
    ent_coef = 0.01,
    truncated_frac = 0.0,
)

algo = a2c.A2C(VmapWrapper(env), hyperparameters)

training_state = algo.init_training_state(rngs)
train = nnx.jit(algo.train, static_argnames=('steps',), donate_argnames=('training_state'))

@nnx.jit
def evaluate(rngs, actor):
    return evaluate_episodes(
        rngs, VmapWrapper(env), actor,
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
# import orbax.checkpoint as ocp

# SAVE_PATH = path.abspath('examples/a2c/_tmp/acrobot')

# _, state = nnx.split(actor)
# checkpointer_save = ocp.StandardCheckpointer()
# checkpointer_save.save(SAVE_PATH, state)


### ENJOY ###

# # NOTE: visualizer is broken in gymnax main branch; use PR https://github.com/RobertTLange/gymnax/pull/84
#     # edit `_render_and_close()` to remove `with env:` statement to avoid closing pygame early
from gymnax.visualize import Visualizer
from gymnax.visualize.vis_gym import render_acrobot

rngs = nnx.Rngs(0, params=1, env=5, actions=3)

EVAL_EPS = 256
returns, lengths = nnx.jit(evaluate_episodes, static_argnums=(1, 3, 4))(
    rngs, VmapWrapper(env), actor, EVAL_EPS, EVAL_EPS)

print("Episode Returns:")
print(returns)
print()

print("Episode Lengths:")
print(lengths)
print()

print(f"Episode Return: mean={jnp.mean(returns)} std={jnp.std(returns, ddof=1)}")
print(f"Episode Length: mean={jnp.mean(lengths)} std={jnp.std(lengths, ddof=1)}")

VISUALIZE_METHOD = "gif"
rngs = nnx.Rngs(0, params=1, env=5, actions=3)

if VISUALIZE_METHOD == 'gif':
    MAX_STEPS = 500
    NUM_EPISODES = 1

    comb_states = []
    comb_cum_rewards = jnp.array((0,))

    for _ in range(NUM_EPISODES):
        timesteps, state, info = rollout_episode(rngs, EpisodeStepCountWrapper(env, MAX_STEPS), actor)
        cum_rewards = jnp.cumsum(timesteps.reward)
        steps = len(timesteps.reward)

        print(f"{'Truncated' if timesteps.truncated[-1] else 'Terminated'} at steps={steps}, return={cum_rewards[-1]}.")

        comb_states += [ jax.tree.map(lambda x: x[i], timesteps.state.state) for i in range(steps + 1) ]
        comb_cum_rewards = jnp.concatenate((comb_cum_rewards, jnp.array((0,)), cum_rewards), axis=0)

    vis = Visualizer(gymnax_env, gymnax_env_params, comb_states, comb_cum_rewards)
    vis.animate("./examples/a2c/visualizations/a2c_test_anim.gif")
    #vis.animate(save_fname=None, view=True)

elif VISUALIZE_METHOD == 'pygame':
    MAX_STEPS = 500

    # Acrobot
    FPS = 10

    visualize_pygame(
        rngs, env, actor, 
        fps=FPS, 
        render_func=lambda state, action: render_acrobot(None, gymnax_env_params, state),
        episode_steps_limit=MAX_STEPS,
        verbose=False
    )

    ## Flappy Bird
    # visualize_pygame(
    #     rngs, env, actor, 
    #     fps=round(1/DT), 
    #     verbose=False
    # )