"""
Vector-env contract tests for JumpKingEnv_Multi.

These verify the gym vector convention without running a long rollout:
 - single_observation_space / single_action_space exist with the right types
 - observation_space / action_space have leading dim N
 - reset() returns (obs, info) with obs shape (N, 9), float32
 - step(actions) returns shapes/dtypes per the contract

If any of these fail, plugging into ppo_rnd_envpool.py will silently break
storage tensor sizes or policy head construction — caught here in 1 second
instead of after a long training crash.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces


pytestmark = pytest.mark.browser   # whole file requires a real driver


@pytest.mark.parametrize("num_agents", [1, 4, 8])
def test_spaces_are_declared(env_multi, num_agents):
    """Both single-agent and batched spaces exist with the right shapes."""
    env = env_multi(num_agents=num_agents)

    # Single-agent spaces — used by PPO to size network heads
    assert isinstance(env.single_observation_space, spaces.Box)
    assert env.single_observation_space.shape == (9,)
    assert env.single_observation_space.dtype == np.float32

    assert isinstance(env.single_action_space, spaces.Discrete)
    assert env.single_action_space.n == 6

    # Batched spaces — used by the rollout buffer
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (num_agents, 9)

    assert isinstance(env.action_space, spaces.MultiDiscrete)
    assert env.action_space.nvec.tolist() == [6] * num_agents


def test_reset_shape_and_dtype(env_multi):
    """reset() returns (obs, info) with the contracted shape and dtype."""
    env = env_multi(num_agents=4)
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4, 9)
    assert obs.dtype == np.float32

    assert isinstance(info, dict)


def test_step_shapes_and_dtypes(env_multi):
    """One step from a sampled action vector returns contract-compliant values."""
    env = env_multi(num_agents=4)
    env.reset()

    actions = env.action_space.sample()
    assert actions.shape == (4,)

    obs, rewards, terminated, truncated, info = env.step(actions)

    # obs
    assert obs.shape == (4, 9)
    assert obs.dtype == np.float32

    # rewards
    assert rewards.shape == (4,)
    assert rewards.dtype == np.float32

    # done flags must be bool arrays of shape (N,)
    assert terminated.shape == (4,)
    assert terminated.dtype == bool
    assert truncated.shape == (4,)
    assert truncated.dtype == bool

    # info must carry the keys PPO and diagnostics depend on
    assert "final_observation" in info
    assert info["final_observation"].shape == (4, 9)
    assert "success" in info
    assert info["success"].shape == (4,)


def test_step_count_increments_per_step(env_multi):
    """Internal _step_count must tick by 1 each step, per-agent."""
    env = env_multi(num_agents=2)
    env.reset()
    assert (env._step_count == 0).all()

    actions = np.zeros(2, dtype=np.int64)   # all idle
    env.step(actions)
    assert (env._step_count == 1).all()

    env.step(actions)
    assert (env._step_count == 2).all()


def test_prev_jump_held_is_per_agent_array(env_multi):
    """Regression guard: _prev_jump_held must stay shape (N,), not a scalar."""
    env = env_multi(num_agents=4)
    env.reset()

    assert env._prev_jump_held.shape == (4,)
    assert env._prev_jump_held.dtype == bool

    # Mix jump and non-jump actions; the array must still have shape (N,) after
    actions = np.array([0, 3, 2, 5], dtype=np.int64)
    env.step(actions)
    assert env._prev_jump_held.shape == (4,)
