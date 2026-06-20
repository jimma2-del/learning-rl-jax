import time

import jax
import jax.numpy as jnp

from jax.typing import ArrayLike
import chex

from flax import nnx
from optax import schedules

from jumanji.environments.routing.snake import Snake, State
from jumanji.environments.logic.game_2048 import Game2048, State, Observation
from core.envs.jumanji import JumanjiWrapper

from core.envs.wrappers import ObsRangeNormalizeWrapper, Wrapper, VmapWrapper
from core.envs.utils import rollout_episode, evaluate_episodes
from core.envs.base import Space

from core.algos import dqn

#jax.config.update("jax_log_compiles", True)

rngs = nnx.Rngs(0, params=1, env=2, actions=3, transitions=4)

jumanji_env = Snake()#Game2048()#Snake()
env = JumanjiWrapper(jumanji_env,
    #lambda state: Observation(state.board, jumanji_env._get_action_mask(state.board))
)

# remove steps and mask from observations to simplify

class CustomWrapper(Wrapper):

    def get_obs(self, key: chex.PRNGKey, state: State):
        obs = super().get_obs(key, state)
        return obs.grid#obs.board#obs.grid

    @property
    def observation_space(self) -> Space:
        space = super().observation_space
        return Space(low=space.low.grid, high=space.high.grid)
        #return Space(low=space.low.board, high=space.high.board)

env = CustomWrapper(env)

# #env = ObsRangeNormalizeWrapper(env)

### TRAIN ###

STEPS = 1_000_000
LOG_INTERVAL_STEPS = 100_000
EVAL_EPS = 32

hyperparameters = dqn.Hyperparameters(
    learning_rate = 2.5e-4,
    train_freq = 4,
    n_envs = 32,
    batch_size = 32,
    epsilon = schedules.linear_schedule(1, 0.05, 0.1*STEPS),
    replay_buffer_size = 100_000
)

algo = dqn.DQN(VmapWrapper(env), hyperparameters)

training_state = algo.init_training_state(rngs)
train = nnx.jit(algo.train, static_argnames=('steps',))

@nnx.jit
def evaluate(rngs, actor):
    return evaluate_episodes(
        rngs, VmapWrapper(env), actor, 
        EVAL_EPS, hyperparameters.n_envs
    )

while training_state.steps < STEPS:
    start_time = time.perf_counter()

    training_state, metrics = train(rngs, training_state, LOG_INTERVAL_STEPS)

    elasped_time = time.perf_counter() - start_time
    sps = LOG_INTERVAL_STEPS / elasped_time
    print(f"Completed steps={training_state.steps}; sps={sps:,.1f}")
    print("Metrics: " + " ".join([ f"{key}={val}" for key, val in metrics.items() ]))

    #eval
    training_state.actor.eval() # make greedy instead of epsilon-greedy
    returns, lengths = evaluate(rngs, training_state.actor)

    print(f"Episode Return: mean={jnp.mean(returns)} std={jnp.std(returns, ddof=1)}")
    print(f"Episode Length: mean={jnp.mean(lengths)} std={jnp.std(lengths, ddof=1)}")

    print()

### ENJOY ###

NUM_EPISODES = 1
ENV_NAME = "snake_test"#"game_2048"
#STEPS_LIMIT = 300#600 
ANIMATE_LIMIT = 300

rngs = nnx.Rngs(0, params=1, env=2, actions=3, transitions=4)

comb_states = []

for _ in range(NUM_EPISODES):
    timesteps, state, info = rollout_episode(rngs, env, training_state.actor)
    eps_return = sum(timesteps.reward)
    steps = len(timesteps.reward)

    print(f"{'Truncated' if timesteps.truncated[-1] else 'Terminated'} at steps={steps}, return={eps_return}.")

    comb_states += [ jax.tree.map(lambda x: x[i], timesteps.state) for i in range(steps + 1) ]

jumanji_env.animate(comb_states[:ANIMATE_LIMIT], 100, f"./examples/dqn/visualizations/dqn_{ENV_NAME}_animation.gif")