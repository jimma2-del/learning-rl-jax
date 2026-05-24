import time
from os import path

import jax
import jax.numpy as jnp

from jax.typing import ArrayLike

from flax import nnx
from optax import schedules

from gymnax.environments import Acrobot, CartPole, MinBreakout
from core.envs.gymnax import GymnaxWrapper

from core.envs.flappy_bird import FlappyBirdEnv, State as FlappyBirdState

from core.envs.wrappers import ObsRangeNormalizeWrapper

from core.envs.utils import rollout_episode, visualize_pygame, evaluate_episodes

from core.algos import dqn
from core.utils import LinearlyInterpolatedTable

#jax.config.update("jax_log_compiles", True)

rngs = nnx.Rngs(0, params=1, env=2, actions=3, transitions=4)

## Gymnax

# gymnax_env = Acrobot()#MinBreakout()#CartPole()
# gymnax_env_params = gymnax_env.default_params

#env = GymnaxWrapper(gymnax_env)

## Flappy Bird

DT = 0.1
env = FlappyBirdEnv(DT)

#env = ObsRangeNormalizeWrapper(env)

### TRAIN ###

STEPS = 1_000_000
LOG_INTERVAL_STEPS = 100_000
EVAL_EPS = 32

hyperparameters = dqn.Hyperparameters(
    learning_rate = 2.5e-4,
    train_freq = 4,
    n_envs = 32,
    epsilon = schedules.linear_schedule(1, 0.05, 0.1*STEPS),
    replay_buffer_size = 100_000
)

algo = dqn.DQN(env, hyperparameters)

training_state = algo.init_training_state(rngs)

while training_state.steps < STEPS:
    start_time = time.perf_counter()

    training_state, metrics = algo.train_epoch(rngs, training_state, LOG_INTERVAL_STEPS)

    elasped_time = time.perf_counter() - start_time
    sps = LOG_INTERVAL_STEPS / elasped_time
    print(f"Completed steps={training_state.steps}; sps={sps:,.1f}")
    print("Metrics: " + " ".join([ f"{key}={val}" for key, val in metrics.items() ]))

    # eval
    # returns, lengths = nnx.jit(evaluate_episodes, static_argnums=(1, 2, 3, 4, 5))(
    #     rngs, env, 
    #     lambda rngs, obs: algo.get_action(rngs, training_state.policy, obs), 
    #     EVAL_EPS, hyperparameters.n_envs
    # )

    # print(f"Episode Return: mean={jnp.mean(returns)} std={jnp.std(returns, ddof=1)}")
    # print(f"Episode Length: mean={jnp.mean(lengths)} std={jnp.std(lengths, ddof=1)}")

    print()

import orbax.checkpoint as ocp

q_net = training_state.policy

SAVE_PATH = path.abspath('examples/dqn/_tmp/flappybird')

# test save
_, state = nnx.split(q_net)
checkpointer_save = ocp.StandardCheckpointer()
checkpointer_save.save(SAVE_PATH, state)


### ENJOY ###

# # NOTE: visualizer is broken in gymnax main branch; use PR https://github.com/RobertTLange/gymnax/pull/84
#     # edit  _render_and_close() to remove 'with env:' statement to avoid closing pygame early
from gymnax.visualize import Visualizer
from gymnax.visualize.vis_gym import render_acrobot

rngs = nnx.Rngs(0, params=1, env=5, actions=3, transitions=4)

def policy(rngs, obs):
    return algo.get_action(rngs, q_net, obs)

MAX_STEPS = 500

EVAL_EPS = 256
returns, lengths = nnx.jit(evaluate_episodes, static_argnums=(1, 2, 3, 4, 5))(
    rngs, env, policy, EVAL_EPS, hyperparameters.n_envs)
print(returns)
print(lengths)
print(f"Episode Return: mean={jnp.mean(returns)} std={jnp.std(returns, ddof=1)}")
print(f"Episode Length: mean={jnp.mean(lengths)} std={jnp.std(lengths, ddof=1)}")

VISUALIZE_METHOD = "pygame"
rngs = nnx.Rngs(0, params=1, env=5, actions=3, transitions=4)

if VISUALIZE_METHOD == 'gif':
    NUM_EPISODES = 1

    comb_states = []
    comb_cum_rewards = jnp.array((0,))

    for _ in range(NUM_EPISODES):
        timesteps, truncated = rollout_episode(rngs, env, policy, MAX_STEPS)
        cum_rewards = jnp.cumsum(timesteps.reward)
        steps = len(timesteps.reward)

        print(f"{'Truncated' if truncated else 'Terminated'} at steps={steps}, return={cum_rewards[-1]}.")

        comb_states += [ jax.tree.map(lambda x: x[i], timesteps.state) for i in range(steps + 1) ]
        comb_cum_rewards = jnp.concatenate((comb_cum_rewards, jnp.array((0,)), cum_rewards), axis=0)

    vis = Visualizer(gymnax_env, gymnax_env_params, comb_states, comb_cum_rewards)
    vis.animate("./examples/dqn/dqn_test_anim.gif")
    #vis.animate(save_fname=None, view=True)

elif VISUALIZE_METHOD == 'pygame':
    # Acrobot
    # FPS = 10
    # window_size = render_acrobot(None, gymnax_env_params, env.reset(rngs.env())[0]).swapaxes(0,1).shape[:2]

    # visualize_pygame(
    #     rngs, env, policy, 
    #     window_size, FPS, 
    #     lambda state, action: render_acrobot(None, gymnax_env_params, state),
    #     episode_steps_limit=MAX_STEPS,
    #     verbose=False
    # )

    ## Flappy Bird
    FPS = round(1/DT)
    window_size = (env.settings.window_size[1], env.settings.window_size[0])

    visualize_pygame(
        rngs, env, policy, 
        window_size, FPS, 
        verbose=False
    )