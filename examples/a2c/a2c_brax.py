import time
from os import path

import jax.numpy as jnp
import jax

from flax import nnx

from brax.envs import create
from core.envs.brax import BraxWrapper

from core.envs.utils import rollout_episode, visualize_pygame, evaluate_episodes
from core.envs.wrappers import JitWrapper, VmapWrapper

from brax.io import html

from optax import schedules
from core.algos import a2c

NUM_EPISODES = 1
ENV_NAME = "halfcheetah"
STEPS_LIMIT = 1000

rngs = nnx.Rngs(0, params=1, env=5, actions=3)

brax_env = create(ENV_NAME, auto_reset=False, batch_size=None, episode_length=STEPS_LIMIT)

env = BraxWrapper(brax_env)

### TRAIN ###

STEPS = 60_000_000#1_000_000
LOG_INTERVAL_STEPS = 10_000_000#100_000

MAX_STEPS = 500

EVAL_EPS = 2048#256
N_ENVS = 2048#256
EVAL_N_ENVS = 2048#256

hyperparameters = a2c.Hyperparameters(
    learning_rate = 10e-4,#2.5e-4,#schedules.linear_schedule(4e-4, 1e-4, STEPS),
    n_envs = N_ENVS,
    n_steps = 5,
    ent_coef = 0.001#schedules.linear_schedule(0.0015, 0.0001, STEPS)
)

algo = a2c.A2C(VmapWrapper(env), hyperparameters)

training_state = algo.init_training_state(rngs)
train = nnx.jit(algo.train, static_argnames=('steps',))

@nnx.jit
def evaluate(rngs, policy):
    return evaluate_episodes(
        rngs, VmapWrapper(env), 
        nnx.vmap(lambda obs, rngs: algo.get_action(rngs, policy, obs, deterministic=True)), 
        EVAL_EPS, EVAL_N_ENVS
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
# import orbax.checkpoint as ocp

# SAVE_PATH = path.abspath(f'examples/a2c/_tmp/{ENV_NAME}')

# _, state = nnx.split(training_state.policy)
# checkpointer_save = ocp.StandardCheckpointer()
# checkpointer_save.save(SAVE_PATH, state)

## enjoy ##
rngs = nnx.Rngs(0, params=1, env=5, actions=3)

#@nnx.jit
def policy(obs, rngs):
    return env.action_space.sample(rngs.actions())
    
VISUALIZE_METHOD = "html"

if VISUALIZE_METHOD == 'html':
    states = []

    for _ in range(NUM_EPISODES):
        timesteps, state, info = rollout_episode(rngs, JitWrapper(env), policy)

        eps_return = sum(timesteps.reward)
        steps = len(timesteps.reward)

        print(f"{'Truncated' if timesteps.truncated[-1] else 'Terminated'} at steps={steps}, return={eps_return}.")

        states += [ jax.tree.map(lambda x: x[i], timesteps.state.pipeline_state) for i in range(steps + 1) ]

    html_content = html.render(brax_env.sys, states)
    with open(f"./examples/a2c/visualizations/a2c_{ENV_NAME}.html", "w") as f:
        f.write(html_content)

elif VISUALIZE_METHOD == 'pygame':
    visualize_pygame(
        rngs, JitWrapper(env), policy, 
        fps=1.0 / brax_env.dt, 
        verbose=False
    )