![Demo](examples/tutorial/snake_ppo.gif)

# Learning RL JAX
Repository of deep reinforcement learning (DRL) algorithms and environments, written completely in JAX + Flax NNX for end-to-end GPU-accelerated training. Standardized environment and algorithm API provides a clean interface, with wrappers allowing it to be used with popular JAX RL environment libraries including Gymnax, Jumanji, Brax, Mujoco Playground, etc. 

Algorithms include: Tabular Q-Learning (w/ linear interpolation for continuous observations), Deep Q-Learning (DQN), Advantage Actor-Critic (A2C), Proximal Policy Optimization (PPO), Twin Delayed Deep Deterministic Policy Gradient (TD3), Soft Actor-Critic (SAC).

Custom environments include: Gridworld, Flappy Bird. More custom environments to come!

Tutorial Notebook: <a href="https://colab.research.google.com/github/jimma2-del/learning-rl-jax/blob/main/examples/tutorial/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" width="140" align="center"/></a>