import time
from os import path

import jax
import jax.numpy as jnp

from jax.typing import ArrayLike

from flax import nnx

from gymnax.environments import Acrobot, CartPole, MinBreakout
from core.envs.gymnax import GymnaxWrapper

from core.envs.flappy_bird import FlappyBirdEnv, State as FlappyBirdState

from core.envs.wrappers import ObsRangeNormalizeWrapper, EpisodeStepCountWrapper

from core.envs.utils import rollout_episode, visualize_pygame, evaluate_episodes

from core.algos import dqn

## Gymnax

# gymnax_env = Acrobot()#MinBreakout()#CartPole()
# gymnax_env_params = gymnax_env.default_params

#env = GymnaxWrapper(gymnax_env)

## Flappy Bird

DT = 0.1
env = FlappyBirdEnv(DT)

#env = ObsRangeNormalizeWrapper(env)

algo = dqn.DQN(env)

import orbax.checkpoint as ocp


SAVE_PATH = path.abspath('examples/dqn/_tmp/flappybird')

# test load
abstract_model = nnx.eval_shape(lambda: algo.create_default_policy(rngs=nnx.Rngs(0)))
graphdef, abstract_state = nnx.split(abstract_model)
checkpointer_load = ocp.StandardCheckpointer()
state_restored = checkpointer_load.restore(SAVE_PATH, abstract_state)

q_net = nnx.merge(graphdef, state_restored)

### ENJOY ###

# # NOTE: visualizer is broken in gymnax main branch; use PR https://github.com/RobertTLange/gymnax/pull/84
#     # edit `_render_and_close()` to remove `with env:` statement to avoid closing pygame early
from gymnax.visualize import Visualizer
from gymnax.visualize.vis_gym import render_acrobot

rngs = nnx.Rngs(0, params=1, env=5, actions=3, transitions=4)

def policy(obs, rngs):
    return algo.get_action(rngs, q_net, obs)

N_ENVS = 32

MAX_STEPS = 500

EVAL_EPS = 256
returns, lengths = nnx.jit(evaluate_episodes, static_argnums=(1, 2, 3, 4, 5))(
    rngs, env, policy, EVAL_EPS, N_ENVS)
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
        timesteps, state, info = rollout_episode(rngs, EpisodeStepCountWrapper(env, MAX_STEPS), policy)
        cum_rewards = jnp.cumsum(timesteps.reward)
        steps = len(timesteps.reward)

        print(f"{'Truncated' if timesteps.truncated[-1] else 'Terminated'} at steps={steps}, return={cum_rewards[-1]}.")

        comb_states += [ jax.tree.map(lambda x: x[i], timesteps.state.state) for i in range(steps + 1) ]
        comb_cum_rewards = jnp.concatenate((comb_cum_rewards, jnp.array((0,)), cum_rewards), axis=0)

    vis = Visualizer(gymnax_env, gymnax_env_params, comb_states, comb_cum_rewards)
    vis.animate("./examples/dqn/dqn_test_anim.gif")
    #vis.animate(save_fname=None, view=True)

elif VISUALIZE_METHOD == 'pygame':
    # Acrobot
    # FPS = 10

    # visualize_pygame(
    #     rngs, env, policy, 
    #     fps=FPS, 
    #     render_func=lambda state, action: render_acrobot(None, gymnax_env_params, state),
    #     episode_steps_limit=MAX_STEPS,
    #     verbose=False
    # )

    ## Flappy Bird
    FPS = round(1/DT)

    visualize_pygame(
        rngs, env, policy, 
        fps=FPS, 
        verbose=False
    )