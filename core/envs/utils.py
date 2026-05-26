import math

from typing import Any, Generic, Callable
from typing_extensions import TypeVar

import chex

import jax
from jax.typing import ArrayLike

import jax.numpy as jnp

from flax import nnx

from core.envs.base import Environment
from core.envs.wrappers import AutoResetWrapper

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")
TEnvAction = TypeVar("TEnvAction")
TRenderFrame = TypeVar("TRenderFrame", default=None)

def evaluate_episodes(rngs: nnx.Rngs, 
    env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame], 
    policy: Callable[[nnx.Rngs, TEnvObs], TEnvAction], 
    episodes: int, n_envs = 32,
    eps_steps_limit: int = None
) -> tuple[jax.Array, jax.Array]:
    """Runs the policy in `n_envs` environments in parallel until completing `episodes` episodes, 
    returning an array of trajectory returns and and an array of trajectory lengths.

    Runs at LEAST `episodes` episodes. If `episodes` is not a multiple of `n_envs`, will run the next multiple.
    """

    eps_per_env = math.ceil(episodes / n_envs)
    env = AutoResetWrapper(env)

    def env_step(carry):
        eps_done, rngs, env_state, cur_return, eps_returns, cur_len, eps_lens = carry

        # step env
        obs = env.get_obs(rngs.env(), env_state)
        action = policy(rngs, obs)
        env_state, reward, terminated, truncated, info = env.step(rngs.env(), env_state, action)

        if eps_steps_limit is not None:
            truncated = jnp.logical_or(truncated, cur_len >= eps_steps_limit)

        cur_return = cur_return + reward
        cur_len = cur_len + 1
        done = jnp.logical_or(terminated, truncated)

        # add metrics to aggregate array of eps metrics if done
        eps_returns = eps_returns.at[eps_done].set(cur_return)
        eps_lens = eps_lens.at[eps_done].set(cur_len)

        eps_done = jnp.where(done, eps_done + 1, eps_done)

        # reset cur metrics if done
        not_done = jnp.logical_not(done)
        cur_return = cur_return * not_done
        cur_len = cur_len * not_done

        return eps_done, rngs, env_state, cur_return, eps_returns, cur_len, eps_lens

    def run_env(rngs):
        env_state, info = env.reset(rngs.env())

        eps_done, rngs, env_state, cur_return, eps_returns, cur_len, eps_lens = nnx.while_loop(
            lambda input: input[0] < eps_per_env, 
            env_step, 
            (0, rngs, env_state, 0, jnp.empty(eps_per_env), 0, jnp.empty(eps_per_env))
        )

        return eps_returns, eps_lens

    eps_returns, eps_lens = nnx.vmap(run_env)(rngs.fork(split=n_envs))

    return eps_returns.flatten(), eps_lens.flatten()

@chex.dataclass
class Timestep(Generic[TEnvState, TEnvObs, TEnvAction]):
    state: TEnvState
    obs: TEnvObs
    action: TEnvAction
    reward: ArrayLike
    info: dict[Any, Any]

def rollout_episode(rngs: nnx.Rngs, 
    env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame], 
    policy: Callable[[nnx.Rngs, TEnvObs], TEnvAction],
    steps_limit: int | None = None,
    chunk_size: int = 100,
) -> tuple[Timestep, bool]:
    """Rollout one episode. Intended for visualization, rather than training. Do NOT JIT.

    Returns: timesteps, truncated
        length of states, infos, and observations will be one greater than actions and rewards
        truncated is a single bool, True if the episode was truncated or `steps_limit` reached, False if terminated
    """

    if steps_limit is not None:
        chunk_size = steps_limit

    @nnx.jit
    @nnx.scan
    def rollout(carry, rngs):
        state, obs = carry

        action = policy(rngs, obs)
        state, reward, terminated, truncated, info = env.step(rngs.env(), state, action)
        obs = env.get_obs(rngs.env(), state)

        return (state, obs), (
            Timestep(state=state, obs=obs, action=action, reward=reward, info=info), 
            terminated, truncated
        )

    state, info = env.reset(rngs.env())
    obs = env.get_obs(rngs.env(), state)

    # info from env.reset and env.step should have the same structure
    #   however, some libraries do not implement this correctly
    #   replace reset info with step info if not
    _, _, _, _, dummy_step_info = env.step(rngs.env(), state, env.action_space.sample(rngs.env()))
    if jax.tree.structure(info) != jax.tree.structure(dummy_step_info):
        info = dummy_step_info

    comb_timesteps = Timestep(state=state, obs=obs, info=info, reward=0, action=env.action_space.low)
    comb_timesteps = jax.tree.map(lambda x: jnp.array((x,)), comb_timesteps)

    done = False
    
    while not done:
        (state, obs), (timesteps, terminateds, truncateds) = rollout(
            (state, obs), rngs.fork(split=chunk_size))

        dones = jnp.logical_or(terminateds, truncateds)
        done_idx = jnp.argmax(dones)
        done = dones[done_idx]
        terminated = terminateds[done_idx]

        if done:
            timesteps = jax.tree.map(lambda x: x[:done_idx+1], timesteps)

        comb_timesteps = jax.tree.map(lambda comb, new: jnp.concatenate((comb, new), axis=0), 
            comb_timesteps, timesteps)

        if steps_limit is not None:
            break

    comb_timesteps.reward = comb_timesteps.reward[1:]
    comb_timesteps.action = comb_timesteps.action[1:]

    return comb_timesteps, jnp.logical_not(terminated)

def visualize_pygame(rngs: nnx.Rngs, 
    env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame], 
    policy: Callable[[nnx.Rngs, TEnvObs], TEnvAction],
    window_size: tuple[int, int], fps: int = 10,
    render_func: Callable[[TEnvState, TEnvAction], ArrayLike] | None = None,
    verbose: bool = False,
    episode_steps_limit: int | None = None,
) -> tuple[Timestep, bool]:
    """Visualizes episodes, rendering in pygame, until closed.
    `render_func` should output an array of pixels (y, x, r, g, b). Defaults to `env.render` if not given.
    Does not JIT wrap the environment or policy. Try JIT wrapping before passing in if slow,
        though this may incur penalties due to the slow transfer of data between devices.
    """
    import pygame
    import numpy as np

    render_func = render_func if render_func is not None else env.render

    state, info = env.reset(rngs.env())
    steps = 0
    eps_return = 0
    terminated = False
    truncated = False

    pygame.init()
    clock = pygame.time.Clock()
    pygame.display.set_caption("Environment Visualization")
    screen = pygame.display.set_mode(window_size)

    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        if terminated or truncated:
            print(f"{'Terminated' if terminated else 'Truncated'} at steps={steps}, return={eps_return}\n")

            state, info = env.reset(rngs.env())
            steps = 0
            eps_return = 0
            terminated = False
            truncated = False
        
        else:
            obs = env.get_obs(rngs.env(), state)
            action = policy(rngs, obs)

            state, reward, terminated, truncated, info = env.step(rngs.env(), state, action)

            steps += 1
        
            if reward != 0 or verbose:
                eps_return += reward
                print(f"steps={steps}, reward={reward}, return={eps_return}", end="")

                if verbose:
                    print(f" obs={obs}, action={action}", end="")
                
                print()

        image_array = np.asarray(render_func(state, action))
        pygame_surface = pygame.surfarray.make_surface(image_array.swapaxes(0,1))
        screen.blit(pygame_surface, (0,0))
        pygame.display.flip()

        clock.tick(fps)

        truncated = truncated or (episode_steps_limit is not None and steps >= episode_steps_limit)

    pygame.quit()