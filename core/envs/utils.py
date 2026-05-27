import math

from typing import Any, Generic, Callable
from typing_extensions import TypeVar

import chex

import jax
from jax.typing import ArrayLike

import jax.numpy as jnp

from flax import nnx

from core.utils.batch_utils import get_tree_vmap_dim

from core.envs.base import Environment
from core.envs.wrappers import AutoResetWrapper, VmapWrapper, VmapConditionallyResetWrapper

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

    `env` does not need to already be wrapped with `VmapWrapper`/`AutoResetWrapper`/`VmapConditionallyResetWrapper`.
        If `env` is already wrapped with these, ensure they are placed at the top level
        so that they can be detected, avoiding duplicate wrapping.

    Runs at LEAST `episodes` episodes. If `episodes` is not a multiple of `n_envs`, will run the next multiple.
    """

    eps_per_env = math.ceil(episodes / n_envs)

    if isinstance(env, VmapWrapper) or isinstance(env, VmapConditionallyResetWrapper):
        env = env.env

    if not isinstance(env, AutoResetWrapper):
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
    terminated: bool
    truncated: bool

def rollout_episode(rngs: nnx.Rngs, 
    env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame], 
    policy: Callable[[nnx.Rngs, TEnvObs], TEnvAction],
    steps_limit: int | None = None,
    chunk_size: int = 100,
) -> tuple[Timestep[TEnvState, TEnvObs, TEnvAction], bool]:
    """Rollout one episode. Intended for visualization, rather than training. Do NOT JIT.

    Performs rollout in chunks of `chunk_size` steps for speed, combining chunks at the end.

    Returns: timesteps
        length of states, infos, and observations will be one greater than actions and rewards
        last timestep will have either terminated=True or truncated=True
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

        return (state, obs), Timestep(state=state, obs=obs, action=action, reward=reward, info=info,
            terminated=terminated, truncated=truncated)

    state, info = env.reset(rngs.env())
    obs = env.get_obs(rngs.env(), state)

    # info from env.reset and env.step should have the same structure
    #   however, some libraries do not implement this correctly
    #   replace reset info with step info if not
    _, _, _, _, dummy_step_info = env.step(rngs.env(), state, env.action_space.sample(rngs.env()))
    if jax.tree.structure(info) != jax.tree.structure(dummy_step_info):
        info = dummy_step_info

    comb_timesteps = Timestep(
        state=state, obs=obs, info=info, 
        reward=0, action=env.action_space.low, # dummy value for first reward/action; removed later
        terminated=False, truncated=False
    )

    # add batch dimension
    comb_timesteps = jax.tree.map(lambda x: jnp.array((x,)), comb_timesteps)

    done = False
    
    while not done:
        (state, obs), timesteps = rollout(
            (state, obs), rngs.fork(split=chunk_size))

        dones = jnp.logical_or(timesteps.terminated, timesteps.truncated)
        done_idx = jnp.argmax(dones)
        done = dones[done_idx]

        if done:
            timesteps = jax.tree.map(lambda x: x[:done_idx+1], timesteps)

        comb_timesteps = jax.tree.map(lambda comb, new: jnp.concatenate((comb, new), axis=0), 
            comb_timesteps, timesteps)

        if steps_limit is not None:
            break
    
    # remove first dummy value; reward and action will have 1 fewer element
    comb_timesteps.reward = comb_timesteps.reward[1:]
    comb_timesteps.action = comb_timesteps.action[1:]

    return comb_timesteps

def parallel_rollout(rngs: nnx.Rngs,
    env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
    policy: Callable[[nnx.Rngs, TEnvObs], TEnvAction],
    iter: int,
    n_envs: int | None = 32,
    initial_env_states: TEnvState | None = None,
) -> tuple[Timestep[TEnvState, TEnvObs, TEnvAction], TEnvState]:
    """Runs `n_envs` environments in parallel for `iter` steps each,
        for a total of `iter * n_envs` steps.

    Automatically resets environments that finish.
        Places original, unresetted terminal states into `info[AutoResetWrapper.NEXT_STATE_INFO_KEY]` (useful eg. for truncation)

    `policy` should accept a batched input of `rngs` and `obs`. Apply `jax.vmap` before passing in if not already batched.

    `env` does not need to already be wrapped with `VmapWrapper`/`AutoResetWrapper`/`VmapConditionallyResetWrapper`.
        If `env` is already wrapped with these, ensure they are placed at the top level
        so that they can be detected, avoiding duplicate wrapping.

    Initializes `n_envs` initial environment states if none given.
    `n_envs` can be infered from the length of the axis if `initial_env_states` is given.

    Returns: Timestep's, final environment states
        timesteps will have two extra leading axes: `shape = (iter, n_envs, ...)`
    """

    if not isinstance(env, VmapConditionallyResetWrapper):
        if not isinstance(env, VmapWrapper):
            if not isinstance(env, AutoResetWrapper):
                env = AutoResetWrapper(env)

            env = VmapWrapper(env)

    if initial_env_states is None:
        assert n_envs is not None, "Must specify `n_envs` if `initial_env_states` not given."
        initial_env_states, info = env.reset(rngs.env(), num=n_envs)
    elif n_envs is None:
        n_envs = get_tree_vmap_dim(initial_env_states)

    def batched_env_step(states: TEnvState, rngs: nnx.Rngs) -> tuple[TEnvState, Timestep[TEnvState, TEnvObs, TEnvAction]]:
        obs = env.get_obs(rngs.env(), states)

        actions = policy(rngs.fork(split=n_envs), obs)

        new_states, rewards, terminateds, truncateds, infos = env.step(rngs.env(), states, actions)

        return new_states, Timestep(state=states, obs=obs, action=actions, reward=rewards, 
            info=infos, terminated=terminateds, truncated=truncateds)

    env_states, timesteps = nnx.scan(batched_env_step)(initial_env_states, rngs.fork(split=iter))

    return timesteps, env_states

def visualize_pygame(rngs: nnx.Rngs, 
    env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame], 
    policy: Callable[[nnx.Rngs, TEnvObs], TEnvAction],
    window_size: tuple[int, int] | None = None, 
    fps: int = 10,
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

    if render_func is None:
        render_func = env.render

    state, info = env.reset(rngs.env())

    if window_size is None: # infer window_size from render_func
        dummy_img = render_func(state, env.action_space.sample(jax.random.key(0)))
        window_size = dummy_img.shape[-2::-1] # first two dims, swapped

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