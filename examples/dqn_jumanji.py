import jax
import jax.numpy as jnp

from jax.typing import ArrayLike
import chex

from flax import nnx
from optax import schedules

#from jumanji.environments.routing.snake import Snake, State
from jumanji.environments.logic.game_2048 import Game2048, State, Observation
from core.envs.jumanji_wrapper import JumanjiWrapper

from core.envs.wrappers import ObsRangeNormalizeWrapper, Wrapper
from core.envs.base import Space

from core.algos.dqn import DQN, DQNHyperparameters

#jax.config.update("jax_log_compiles", True)

rngs = nnx.Rngs(0, params=1, env=2, actions=3, transitions=4)

jumanji_env = Game2048()#Snake()
env = JumanjiWrapper(jumanji_env,
    lambda state: Observation(state.board, jumanji_env._get_action_mask(state.board))
)

# remove steps and mask from observations to simplify

class CustomWrapper(Wrapper):

    def get_obs(self, key: chex.PRNGKey, state: State):
        obs = super().get_obs(key, state)
        return obs.board#obs.grid

    @property
    def observation_space(self) -> Space:
        space = super().observation_space
        #return Space(low=space.low.grid, high=space.high.grid)
        return Space(low=space.low.board, high=space.high.board)

env = CustomWrapper(env)

# #env = ObsRangeNormalizeWrapper(env)

### TRAIN ###

STEPS = 10_000_000
LOG_INTERVAL_STEPS = 100_000

hyperparameters = DQNHyperparameters(
    learning_rate = 2.5e-4,
    train_freq = 4,
    n_envs = 64,#32,
    batch_size = 32,
    epsilon = schedules.linear_schedule(1, 0.05, 0.1*STEPS),
    replay_buffer_size = 100_000
)

algo = DQN(env, hyperparameters)

q_net = algo.train(rngs, STEPS, log_interval_steps=LOG_INTERVAL_STEPS)

### ENJOY ###

NUM_EPISODES = 1
ENV_NAME = "game_2048"
MAX_STEPS = 600 

rngs = nnx.Rngs(0, params=1, env=2, actions=3, transitions=4)

@nnx.jit
def policy(rngs, obs):
    return algo.get_greedy_action(rngs, q_net, obs)

reset = jax.jit(env.reset)
step = jax.jit(env.step)
get_obs = jax.jit(env.get_obs)

states = []

for _ in range(NUM_EPISODES):
    eps_return = 0
    steps = 0

    terminated = False
    truncated = False

    state, info = reset(rngs.env())
    states.append(state)

    while steps < MAX_STEPS and not (terminated or truncated):
        obs = get_obs(rngs.env(), state)
        action = policy(rngs, obs)

        state, reward, terminated, truncated, info = step(rngs.env(), state, action)

        states.append(state)

        eps_return += reward
        steps += 1

    print(f"{'Terminated' if terminated else 'Truncated'} at steps={steps}, return={eps_return}.")

jumanji_env.animate(states, 100, f"./examples/dqn_{ENV_NAME}_animation.gif")