import math

from typing import Any, Generic, Callable, Protocol, TypeAlias
from typing_extensions import TypeVar

import chex

import jax
from jax.typing import ArrayLike

import jax.numpy as jnp

from flax import nnx

from core.utils.batch_utils import get_tree_vmap_dim
from core.utils.func_utils import optionally_pass

from core.envs.base import Environment
from core.envs.wrappers import AutoResetWrapper, VmapWrapper, VmapConditionallyResetWrapper

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")
TEnvAction = TypeVar("TEnvAction")
TRenderFrame = TypeVar("TRenderFrame", default=None)

class PolicyWithRngs(Generic[TEnvObs, TEnvAction], Protocol):
    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs) -> TEnvAction: ...

Policy: TypeAlias = Callable[[TEnvObs], TEnvAction] | PolicyWithRngs[TEnvObs, TEnvAction]

def evaluate_episodes(rngs: nnx.Rngs, 
    env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame], 
    policy: Policy[TEnvObs, TEnvAction], 
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
        action = optionally_pass(policy, rngs=rngs)(obs)
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

TTakeObj = TypeVar('TTakeObj', default=Timestep[TEnvState, TEnvObs, TEnvAction])

def rollout_episode(rngs: nnx.Rngs, 
    env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame], 
    policy: Policy[TEnvObs, TEnvAction],
    chunk_size: int = 100,
    take_func: Callable[[Timestep[TEnvState, TEnvObs, TEnvAction]], TTakeObj] | None = None
) -> tuple[TTakeObj, TEnvState, dict[Any, Any]]:
    """Rollout one episode. Intended for visualization, rather than training. Do NOT JIT.
    Performs rollout in chunks of `chunk_size` steps for speed, combining chunks at the end.

    `take_func`: Optional function to specify which values to take from Timestep
        A full `Timestep` object is returned by default.

    Returns: 
        timesteps, or user defined take objs for each timestep
        final state and final info (may be useful eg. if truncated)
    """

    if take_func is None:
        take_func = lambda x: x

    @nnx.jit
    @nnx.scan
    def rollout(carry, rngs):
        state, info = carry

        obs = env.get_obs(rngs.env(), state)
        action = optionally_pass(policy, rngs=rngs)(obs)

        next_state, reward, terminated, truncated, next_info = env.step(rngs.env(), state, action)

        timestep = Timestep(state=state, obs=obs, action=action, reward=reward, info=info,
            terminated=terminated, truncated=truncated)

        return (next_state, next_info), (take_func(timestep), terminated, truncated)

    state, info = env.reset(rngs.env())

    comb_take_vals = None
    done = False
    
    while not done:
        (state, info), (take_vals, terminated, truncated) = rollout(
            (state, info), rngs.fork(split=chunk_size))

        dones = jnp.logical_or(terminated, truncated)
        done_idx = jnp.argmax(dones)
        done = dones[done_idx]

        if done:
            take_vals = jax.tree.map(lambda x: x[:done_idx+1], take_vals)

        if comb_take_vals is None:
            comb_take_vals = take_vals
        else: # append to existing
            comb_take_vals = jax.tree.map(lambda comb, new: jnp.concatenate((comb, new), axis=0), 
                comb_take_vals, take_vals)
    
    return comb_take_vals, state, info

def parallel_rollout(rngs: nnx.Rngs,
    env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
    policy: Policy[TEnvObs, TEnvAction],
    iter: int,
    n_envs: int | None = 32,
    initial_env_states: TEnvState | None = None,
    initial_env_infos: dict[Any, Any] | None = None,
    take_func: Callable[[Timestep[TEnvState, TEnvObs, TEnvAction]], TTakeObj] | None = None
) -> tuple[TTakeObj, TEnvState, dict[Any, Any]]:
    """Runs `n_envs` environments in parallel for `iter` steps each,
        for a total of `iter * n_envs` steps.

    Automatically resets environments that finish.
        Places the original, unresetted state into `info[UNRESET_STATE_INFO_KEY]` (useful eg. for truncation)
        This will be the same as the returned new_state if not terminated and not truncated.

    `policy` should accept a batched input of `rngs` and `obs`. Apply `jax.vmap` before passing in if not already batched.

    `env` does not need to already be wrapped with `VmapWrapper`/`AutoResetWrapper`/`VmapConditionallyResetWrapper`.
        If `env` is already wrapped with these, ensure they are placed at the top level
        so that they can be detected, avoiding duplicate wrapping.

    Initializes `n_envs` initial environment states if none given.
    `n_envs` can be infered from the length of the axis if `initial_env_states` is given.

    If `initial_env_states` is given but `initial_env_infos` is not given,
        the first infos will be a dummy value.

    `take_func`: Optional function to specify which values to take from Timestep
        A full `Timestep` object is returned by default.

    Returns: timesteps, or user defined take objs for each timestep; final states; final infos
        timesteps/take objs will have two extra leading axes: `shape = (iter, n_envs, ...)`
    """

    if not isinstance(env, VmapConditionallyResetWrapper):
        if not isinstance(env, VmapWrapper):
            if not isinstance(env, AutoResetWrapper):
                env = AutoResetWrapper(env)

            env = VmapWrapper(env)

    if initial_env_states is None:
        assert n_envs is not None, "Must specify `n_envs` if `initial_env_states` not given."
        initial_env_states, initial_env_infos = env.reset(rngs.env(), num=n_envs)
    elif n_envs is None:
        n_envs = get_tree_vmap_dim(initial_env_states)

    if initial_env_infos is None:
        _, initial_env_infos = env.reset(jax.random.key(0), num=n_envs) # dummy value

    if take_func is None:
        take_func = lambda x: x

    def batched_env_step(carry: tuple[TEnvState, dict[Any,Any]], rngs: nnx.Rngs) -> tuple[TEnvState, TTakeObj]:
        states, infos = carry

        obs = env.get_obs(rngs.env(), states)
        actions = optionally_pass(policy, rngs=rngs.fork(split=n_envs))(obs)

        new_states, rewards, terminateds, truncateds, new_infos = env.step(rngs.env(), states, actions)

        return (new_states, new_infos), take_func(Timestep(state=states, obs=obs, action=actions, reward=rewards, 
            info=infos, terminated=terminateds, truncated=truncateds))

    (env_states, infos), take_values = nnx.scan(batched_env_step)(
        (initial_env_states, initial_env_infos), rngs.fork(split=iter))

    return take_values, env_states, infos

def visualize_pygame(rngs: nnx.Rngs, 
    env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame], 
    policy: Policy[TEnvObs, TEnvAction],
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
            action = optionally_pass(policy, rngs=rngs)(obs)

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