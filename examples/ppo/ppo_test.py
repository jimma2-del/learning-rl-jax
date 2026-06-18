import time
from os import path

import jax
import jax.numpy as jnp

from jax.typing import ArrayLike

from flax import nnx
from optax import schedules

from gymnax.environments import Acrobot, CartPole, MinBreakout, MinAsterix
from core.envs.gymnax import GymnaxWrapper

from core.envs.flappy_bird import FlappyBirdEnv

from core.envs.wrappers import ObsRangeNormalizeWrapper, EpisodeStepCountWrapper, VmapWrapper

from core.envs.utils import rollout_episode, visualize_pygame, evaluate_episodes

from core.algos import ppo

#jax.config.update("jax_log_compiles", True)

rngs = nnx.Rngs(0, params=10, env=20, actions=30)

## Gymnax

gymnax_env = Acrobot()
gymnax_env_params = gymnax_env.default_params

env = GymnaxWrapper(gymnax_env)

## Flappy Bird

# DT = 0.1
# env = FlappyBirdEnv(DT)

# env = ObsRangeNormalizeWrapper(env)

### TRAIN ###

STEPS = 1_000_000
LOG_INTERVAL_STEPS = 100_000
EVAL_EPS = 32

hyperparameters = ppo.Hyperparameters(
    learning_rate = 2.5e-4,#10e-4,
    n_envs = 256,
    gae_lambda = 0.95,

    rollout_length = 32,
    n_minibatches = 8, 
    n_epochs = 4, 

    clip_epsilon = 0.2,

    vf_coef = 0.5, 
    ent_coef = 0.01,

    normalize_advantages = True,

    recompute_advantages = True,
    target_kl = 0.02
)

algo = ppo.PPO(VmapWrapper(env), hyperparameters)

training_state = algo.init_training_state(rngs)
train = nnx.jit(algo.train, static_argnames=('steps',))

@nnx.jit
def evaluate(rngs, policy):
    return evaluate_episodes(
        rngs, VmapWrapper(env), 
        nnx.vmap(lambda obs, rngs: algo.get_action(rngs, policy, obs, deterministic=True)), 
        EVAL_EPS, 256
    )

while training_state.steps < STEPS:
    start_time = time.perf_counter()

    training_state, metrics = train(rngs, training_state, LOG_INTERVAL_STEPS)

    elasped_time = time.perf_counter() - start_time
    sps = LOG_INTERVAL_STEPS / elasped_time
    print(f"Completed steps={training_state.steps}; sps={sps:,.1f}")
    print("Metrics: " + " ".join([ f"{key}={val}" for key, val in metrics.items() ]))

    # eval
    returns, lengths = evaluate(rngs, training_state.policy)

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

# # NOTE: visualizer is broken in gymnax main branch; use PR https://github.com/RobertTLange/gymnax/pull/84
#     # edit  _render_and_close() to remove 'with env:' statement to avoid closing pygame early
from gymnax.visualize import Visualizer
from gymnax.visualize.vis_gym import render_acrobot

rngs = nnx.Rngs(0, params=1, env=5, actions=3)

def actor(obs, rngs):
    print(training_state.policy(obs, rngs=rngs))
    return algo.get_action(rngs, training_state.policy, obs, deterministic=True)

MAX_STEPS = 500

EVAL_EPS = 256
returns, lengths = nnx.jit(evaluate_episodes, static_argnums=(1, 2, 3, 4, 5))(
    rngs, VmapWrapper(env), nnx.vmap(actor), EVAL_EPS, hyperparameters.n_envs)
print(returns)
print(lengths)
print(f"Episode Return: mean={jnp.mean(returns)} std={jnp.std(returns, ddof=1)}")
print(f"Episode Length: mean={jnp.mean(lengths)} std={jnp.std(lengths, ddof=1)}")

VISUALIZE_METHOD = "gif"
rngs = nnx.Rngs(0, params=1, env=5, actions=3)

if VISUALIZE_METHOD == 'gif':
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
    vis.animate("./examples/ppo/visualizations/ppo_test_anim.gif")
    #vis.animate(save_fname=None, view=True)

elif VISUALIZE_METHOD == 'pygame':
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