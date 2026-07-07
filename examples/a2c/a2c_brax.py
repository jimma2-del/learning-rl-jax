import time
from os import path

import jax.numpy as jnp
import jax

from flax import nnx

from brax.envs import create
from core.envs.brax import BraxWrapper

from core.envs.utils import rollout_episode, visualize_pygame, evaluate_episodes
from core.envs.wrappers import JitWrapper, VmapWrapper, PrecomputedResetsPoolWrapper

from brax.io import html

from optax import schedules
from core.algos import a2c

ENV_NAME = "halfcheetah"
MAX_STEPS = 1000
N_ENVS = 2048#256

rngs = nnx.Rngs(0, params=1, env=5, actions=3)

brax_env = create(ENV_NAME, auto_reset=False, batch_size=None, episode_length=MAX_STEPS)

env = BraxWrapper(brax_env)

RESETS_POOL_SIZE = 32768
resets_pool_states_infos = jax.vmap(env.reset)(jax.random.split(rngs.env(), RESETS_POOL_SIZE))
env = PrecomputedResetsPoolWrapper(env, resets_pool_states_infos)

### TRAIN ###

STEPS = 10_000_000

EVAL_EPS = 256
EVAL_INTERVAL = 1_000_000
N_LOGS_PER_EVAL = 4

MAX_STEPS = 500

hyperparameters = a2c.Hyperparameters(
    learning_rate = 2.5e-4,#schedules.linear_schedule(4e-4, 1e-4, STEPS),
    n_envs = N_ENVS,
    rollout_length = 5,
    ent_coef = 0.01,#schedules.linear_schedule(0.0015, 0.0001, STEPS)
    truncated_frac = 1.0 / MAX_STEPS,
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

# SAVE_PATH = path.abspath(f'examples/a2c/_tmp/{ENV_NAME}')

# _, state = nnx.split(actor)
# checkpointer_save = ocp.StandardCheckpointer()
# checkpointer_save.save(SAVE_PATH, state)

## enjoy ##
rngs = nnx.Rngs(0, params=1, env=5, actions=3)

VISUALIZE_METHOD = "html"

if VISUALIZE_METHOD == 'html':
    NUM_EPISODES = 1

    states = []

    for _ in range(NUM_EPISODES):
        timesteps, state, info = rollout_episode(rngs, JitWrapper(env), actor)

        eps_return = sum(timesteps.reward)
        steps = len(timesteps.reward)

        print(f"{'Truncated' if timesteps.truncated[-1] else 'Terminated'} at steps={steps}, return={eps_return}.")

        states += [ jax.tree.map(lambda x: x[i], timesteps.state.pipeline_state) for i in range(steps + 1) ]

    html_content = html.render(brax_env.sys, states)
    with open(f"./examples/a2c/visualizations/a2c_{ENV_NAME}.html", "w") as f:
        f.write(html_content)

elif VISUALIZE_METHOD == 'pygame':
    visualize_pygame(
        rngs, JitWrapper(env), actor, 
        fps=1.0 / brax_env.dt, 
        verbose=False
    )