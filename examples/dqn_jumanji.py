import jax
import jax.numpy as jnp

from jax.typing import ArrayLike
import chex

from flax import nnx
from optax import schedules

from jumanji.environments.routing.snake import Snake, State
from jumanji.environments.logic.game_2048 import Game2048, State, Observation
from core.envs.jumanji import JumanjiWrapper

from core.envs.wrappers import ObsRangeNormalizeWrapper, Wrapper
from core.envs.utils import rollout_episode
from core.envs.base import Space

from core.algos.dqn import DQN, DQNHyperparameters

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
ENV_NAME = "snake_test"#"game_2048"
#STEPS_LIMIT = 300#600 
ANIMATE_LIMIT = 300

rngs = nnx.Rngs(0, params=1, env=2, actions=3, transitions=4)

#@nnx.jit
def policy(rngs, obs):
    return algo.get_greedy_action(rngs, q_net, obs)

comb_states = []

for _ in range(NUM_EPISODES):
    timesteps, truncated = rollout_episode(rngs, env, policy)
    eps_return = sum(timesteps.reward)
    steps = len(timesteps.reward)

    print(f"{'Truncated' if truncated else 'Terminated'} at steps={steps}, return={eps_return}.")

    comb_states += [ jax.tree.map(lambda x: x[i], timesteps.state) for i in range(steps + 1) ]

jumanji_env.animate(comb_states[:ANIMATE_LIMIT], 100, f"./examples/dqn_{ENV_NAME}_animation.gif")