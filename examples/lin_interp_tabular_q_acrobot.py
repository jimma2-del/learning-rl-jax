import jax
import jax.numpy as jnp

from gymnax.environments import Acrobot, CartPole
from core.envs.gymnax import GymnaxWrapper, Space

from core.envs.wrappers import Wrapper

from core.algos.linearly_interpolated_tabular_q import LinearlyInterpolatedTabularQ, TabularQHyperparameters
from core.utils import LinearlyInterpolatedTable

#jax.config.update("jax_log_compiles", True)

SEED = 2
key = jax.random.key(SEED)

gymnax_env = Acrobot()
gymnax_env_params = gymnax_env.default_params

env = GymnaxWrapper(gymnax_env)

# add wrapper for custom rewards and observations

class AcrobotWrapper(Wrapper):
    def step(self, key, state, action):
        step_key, obs_key = jax.random.split(key)

        state, reward, terminated, truncated, info = super().step(step_key, state, action)
        obs = super().get_obs(obs_key, state)

        height = (-obs[0] - (obs[0] * obs[2] - obs[1] * obs[3])) / 2 # [-1, 0.5]
        h_d = ((obs[1] * (1 + obs[2]) + obs[0] * obs[3]) * obs[4] 
            + (obs[1] * obs[2] + obs[0] * obs[3]) * obs[5]) / 2

        reward = reward + (height - 1)/2 + jnp.abs(h_d)/5

        #jax.debug.print("reward={r} height={h} h_d={h_d}", r=reward, h=height, h_d=h_d)
        #print(f"reward={reward} height={height} h_d={h_d}")

        return state, reward, terminated, False, info

    def get_obs(self, key, state):
        obs = super().get_obs(key, state)

        return jnp.array((
            jnp.atan2(obs[1], obs[0]), 
            jnp.atan2(obs[3], obs[2]), 
            obs[4], obs[5]
        ), dtype=jnp.float32)
        
    @property
    def observation_space(self):
        return Space(
            low=jnp.array((-jnp.pi, -jnp.pi, -13, -29), dtype=jnp.float32),
            high=jnp.array((jnp.pi, jnp.pi, 13, 29), dtype=jnp.float32)
        )

env = AcrobotWrapper(env)

### TRAIN ###
Q_TABLE_GRIDPOINTS_PER_AXIS = 10

# q_table = LinearlyInterpolatedTable(
#     min=env.observation_space.low, 
#     max=env.observation_space.high, 
#     step=(env.observation_space.high - env.observation_space.low) / Q_TABLE_GRIDPOINTS_PER_AXIS
# )

# ACROBOT
# q_table = LinearlyInterpolatedTable(
#     min=(-1, -1, -1, -1, -13, -29), 
#     max=(1, 1, 1, 1, 13, 29), 
#     step=(0.2, 0.2, 0.2, 0.2, 2, 2)
# )
# q_table = LinearlyInterpolatedTable(
#     min=(-3.2, -3.2, -13, -29), 
#     max=(3.2, 3.2, 13, 29), 
#     step=(0.4, 0.4, 2, 4)
# )

q_table = LinearlyInterpolatedTable(
    min=(-3.2, -3.2, -6, -15), 
    max=(3.2, 3.2, 6, 15), 
    step=(0.2, 0.4, 0.25, 0.25)
)

# # testing performance on a smaller table
# q_table = LinearlyInterpolatedTable(
#     min=(-3.2, -3.2, -6, -15), 
#     max=(3.2, 3.2, 6, 15), 
#     step=(1.6, 1.6, 2, 2)
# )

# # CARTPOLE
# q_table = LinearlyInterpolatedTable(
#     min=(-2.4, -2.4, -0.2095, -2.4), 
#     max=(2.4, 2.4, 0.2095, 2.4), 
#     #step=(0.1, 0.1, 0.005, 0.05)
#     step=(0.2, 0.2, 0.02, 0.2)
# )

hyperparameters = TabularQHyperparameters(
    discount_rate = 0.98,
    learning_rate = 0.01,

    epsilon_final = 0.01,

    replay_buffer_size = 4096, #1024,
    batch_size = 256, #64,
    train_freq = 1,
    n_envs = 256, #64,

    target_update_interval = 4096, #512,
)

algo = LinearlyInterpolatedTabularQ(env, q_table, hyperparameters)

#q_vals = algo.init_q_table_vals()

STEPS = 10_000_000#1_000_000_000
LOG_INTERVAL_STEPS = 1_000_000#10_000_000

key, train_key = jax.random.split(key, 2)
q_vals = algo.train(train_key, STEPS, log_interval_steps=LOG_INTERVAL_STEPS)

### ENJOY ###

# NOTE: visualizer is broken in gymnax main branch; use PR https://github.com/RobertTLange/gymnax/pull/84
    # edit  _render_and_close() to remove 'with env:' statement to avoid closing pygame early
from gymnax.visualize import Visualizer
from gymnax.visualize.vis_gym import render_acrobot

from flax import nnx
from core.envs.utils import rollout_episode, visualize_pygame

rngs = nnx.Rngs(0, params=1, env=5, actions=3, transitions=4)

def policy(rngs, obs):
    return algo.get_greedy_action(q_vals, obs)

VISUALIZE_METHOD = "pygame"

if VISUALIZE_METHOD == 'gif':
    NUM_EPISODES = 1

    comb_states = []
    comb_cum_rewards = jnp.array((0,))

    for _ in range(NUM_EPISODES):
        timesteps, truncated = rollout_episode(rngs, env, policy)
        cum_rewards = jnp.cumsum(timesteps.reward)
        steps = len(timesteps.reward)

        print(f"{'Truncated' if truncated else 'Terminated'} at steps={steps}, return={cum_rewards[-1]}.")

        comb_states += [ jax.tree.map(lambda x: x[i], timesteps.state) for i in range(steps + 1) ]
        comb_cum_rewards = jnp.concatenate((comb_cum_rewards, jnp.array((0,)), cum_rewards), axis=0)

    vis = Visualizer(gymnax_env, gymnax_env_params, comb_states, comb_cum_rewards)
    #vis.animate("./examples/lin_interp_tabular_q_acrobot_anim.gif")
    vis.animate("./examples/lin_interp_tabular_q_cartpole_anim.gif")
    #vis.animate(save_fname=None, view=True)

elif VISUALIZE_METHOD == 'pygame':
    FPS = 10
    window_size = render_acrobot(None, gymnax_env_params, env.reset(rngs.env())[0]).swapaxes(0,1).shape[:2]

    visualize_pygame(
        rngs, env, policy, 
        window_size, FPS, 
        lambda state, action: render_acrobot(None, gymnax_env_params, state),
        verbose=False
    )