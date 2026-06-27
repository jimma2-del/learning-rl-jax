import time

import numpy as np

import jax
import jax.numpy as jnp

from optax import schedules
from flax import nnx

from gymnax.environments import Acrobot, CartPole
from core.envs.gymnax import GymnaxWrapper, Space

from core.envs.wrappers import Wrapper, VmapWrapper

from core.envs.utils import evaluate_episodes

from core.algos import tabular_q_learning

#jax.config.update("jax_log_compiles", True)

#gymnax_env = CartPole()
gymnax_env = Acrobot()
gymnax_env_params = gymnax_env.default_params

env = GymnaxWrapper(gymnax_env)

# wrapper to compress obs by converting (sin, cos) -> angle

class AcrobotWrapper(Wrapper):
    def get_obs(self, key, state):
        obs = super().get_obs(key, state)

        return jnp.array((
            jnp.atan2(obs[1], obs[0]), 
            jnp.atan2(obs[3], obs[2]), 
            obs[4], 
            obs[5]
        ), dtype=jnp.float32)
        
    @property
    def observation_space(self):
        return Space(
            low=np.array((-np.pi, -np.pi, -13, -29), dtype=np.float32),
            high=np.array((np.pi, np.pi, 13, 29), dtype=np.float32)
        )

env = AcrobotWrapper(env)

### TRAIN ###
# Q_TABLE_GRIDPOINTS_PER_AXIS = 10

# low  = env.observation_space.low
# high = env.observation_space.high
# res  = (env.observation_space.high - env.observation_space.low) / Q_TABLE_GRIDPOINTS_PER_AXIS

# ACROBOT
# low  = ( -1,  -1,  -1,  -1, -13, -29)
# high = (  1,   1,   1,   1,  13,  29)
# res  = (0.2, 0.2, 0.2, 0.2,   2,   2)

# low  = (-3.2, -3.2, -13, -29)
# high = ( 3.2,  3.2,  13,  29)
# res  = ( 0.4,  0.4,   2,   4)

low  = (-3.2, -3.2,   -6,  -15)
high = ( 3.2,  3.2,    6,   15)
res  = ( 0.2,  0.4, 0.25, 0.25)

# # testing performance on a smaller table
# low  = (-3.2, -3.2, -6, -15)
# high = ( 3.2,  3.2,  6,  15)
# res  = ( 1.6,  1.6,  2,   2)

# # CARTPOLE
# low  = (-2.4, -2.4, -0.2095, -2.4)
# high = ( 2.4,  2.4,  0.2095,  2.4)
# #res  = ( 0.1,  0.1,   0.005, 0.05)
# res  = ( 0.2,  0.2,    0.02,  0.2)

rngs = nnx.Rngs(0, params=1, env=2, actions=3, transitions=4)

STEPS = 70_000_000

EVAL_EPS = 256
EVAL_INTERVAL = 7_000_000
N_LOGS_PER_EVAL = 7

hyperparameters = tabular_q_learning.Hyperparameters(
    discount_rate = 0.98,
    learning_rate = 0.1,
    epsilon = schedules.linear_schedule(1, 0.05, 0.1*STEPS),
    n_envs = 256, #64,
)

algo = tabular_q_learning.TabularQLearning(VmapWrapper(env), hyperparameters)
train = nnx.jit(algo.train, static_argnames=('steps',))

q_func = tabular_q_learning.LinInterpTabularQFunc(
    int(env.action_space.high + 1), Space(np.array(low), np.array(high)), np.array(res))
training_state = algo.init_training_state(rngs, q_func)

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
    actor = algo.make_actor(training_state.q_func, epsilon=0)
    returns, lengths = evaluate(rngs, actor)

    print(f"Episode Return: mean={jnp.mean(returns)} std={jnp.std(returns, ddof=1)}")
    print(f"Episode Length: mean={jnp.mean(lengths)} std={jnp.std(lengths, ddof=1)}")

print()

### ENJOY ###

# NOTE: visualizer is broken in gymnax main branch; use PR https://github.com/RobertTLange/gymnax/pull/84
    # edit `_render_and_close()` to remove `with env:` statement to avoid closing pygame early
from gymnax.visualize import Visualizer
from gymnax.visualize.vis_gym import render_acrobot

from flax import nnx
from core.envs.utils import rollout_episode, visualize_pygame

rngs = nnx.Rngs(0, params=1, env=5, actions=3, transitions=4)

VISUALIZE_METHOD = 'gif'

if VISUALIZE_METHOD == 'gif':
    NUM_EPISODES = 1

    comb_states = []
    comb_cum_rewards = jnp.array((0,))

    for _ in range(NUM_EPISODES):
        timesteps, state, info = rollout_episode(rngs, env, actor)
        cum_rewards = jnp.cumsum(timesteps.reward)
        steps = len(timesteps.reward)

        print(f"{'Truncated' if timesteps.truncated[-1] else 'Terminated'} at steps={steps}, return={cum_rewards[-1]}.")

        comb_states += [ jax.tree.map(lambda x: x[i], timesteps.state) for i in range(steps + 1) ]
        comb_cum_rewards = jnp.concatenate((comb_cum_rewards, jnp.array((0,)), cum_rewards), axis=0)

    vis = Visualizer(gymnax_env, gymnax_env_params, comb_states, comb_cum_rewards)
    vis.animate("./examples/tabular_q_learning/visualizations/lin_interp_tabular_q_acrobot_anim2.gif")
    #vis.animate("./examples/tabular_q_learning/visualizations/lin_interp_tabular_q_cartpole_anim.gif")
    #vis.animate(save_fname=None, view=True)

elif VISUALIZE_METHOD == 'pygame':
    FPS = 10

    visualize_pygame(
        rngs, env, actor, 
        fps=FPS, 
        render_func=lambda state, action: render_acrobot(None, gymnax_env_params, state),
        verbose=False
    )