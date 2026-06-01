"""
End-to-end smoke test for JumpKingEnv_Multi.

Runs 50 random steps through a real browser and asserts the vector-env contract
holds *every* step — not just the first. Catches:

  - Auto-reset misbehaviour (e.g. obs shape collapsing to (N-k, 9) when k agents die)
  - Reward/dtype drift across steps
  - Selenium round-trip silently failing mid-rollout
  - PBRS exploding (rewards >> theoretical bound)

Slow (~30 s for N=4) — run via `pytest -m browser` or `python tests/main.py --browser`.
"""

from __future__ import annotations

import numpy as np
import pytest


pytestmark = pytest.mark.browser


def test_50_random_steps_contract_holds(env_multi):
    """The full vector-env contract must hold every step of a 50-step rollout."""
    env = env_multi(num_agents=4)
    obs, _ = env.reset()
    assert obs.shape == (4, 9)

    for t in range(50):
        actions = env.action_space.sample()
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Every return value keeps its shape / dtype, every step
        assert obs.shape == (4, 9),   f"obs shape drifted at step {t}: {obs.shape}"
        assert obs.dtype == np.float32, f"obs dtype drifted at step {t}: {obs.dtype}"

        assert rewards.shape == (4,), f"rewards shape drifted at step {t}"
        assert rewards.dtype == np.float32

        assert terminated.shape == (4,) and terminated.dtype == bool
        assert truncated.shape == (4,) and truncated.dtype == bool

        # Per-step rewards must be finite — catches PBRS explosion / nan propagation
        assert np.all(np.isfinite(rewards)), f"non-finite reward at step {t}: {rewards}"

        # No reward should exceed any sane bound this early in a random rollout.
        # Loose bound: STEP_PENALTY + max plausible PBRS jump + SUCCESS_REWARD
        assert np.all(np.abs(rewards) < 1000), (
            f"reward magnitude exceeded sanity bound at step {t}: {rewards}"
        )

        # final_observation always shape (N, 9) — even on no-done steps
        assert info["final_observation"].shape == (4, 9)


def test_done_agents_get_fresh_obs(env_multi, monkeypatch):
    """
    When an agent's _is_at_princess returns True (we force it via a monkey-patch),
    the returned obs at that slot must be the *fresh* start state, not the terminal one.

    This is the auto-reset contract — the bit ppo_rnd_envpool.py depends on.
    """
    env = env_multi(num_agents=4)
    env.reset()

    # Force agent 1 to "succeed" on the next step
    original = env._is_at_princess_all
    def forced_success(states):
        out = original(states)
        out[1] = True   # mark agent 1 as at-princess
        return out
    monkeypatch.setattr(env, "_is_at_princess_all", forced_success)

    actions = np.zeros(4, dtype=np.int64)
    pre_obs, _ = env.reset()
    obs, rewards, terminated, truncated, info = env.step(actions)

    # Agent 1 must be flagged as terminated
    assert terminated[1] == True, "forced success should mark agent 1 terminated"

    # Agent 1 must have received the success bonus (≥ SUCCESS_REWARD - small step penalty)
    from environment.jump_king_env import SUCCESS_REWARD, STEP_PENALTY
    assert rewards[1] >= SUCCESS_REWARD + STEP_PENALTY - 1.0, (
        f"agent 1 expected success bonus, got reward={rewards[1]}"
    )

    # Agent 1's slot in returned obs must be the FRESH start state (post-reset),
    # while info["final_observation"][1] must be the TERMINAL state (pre-reset).
    # We don't know the exact pre-reset value but we *can* assert they're different.
    assert not np.array_equal(obs[1], info["final_observation"][1]), (
        "obs[1] should be post-reset; info['final_observation'][1] should be pre-reset"
    )

    # Agents 0, 2, 3 not flagged → obs and final_observation may coincide for them
    assert not terminated[0]
    assert not terminated[2]
    assert not terminated[3]


def test_per_agent_step_count_resets_on_done(env_multi, monkeypatch):
    """
    After a forced termination, the resurrected agent's _step_count must be back to 0,
    while the alive agents' counts continue ticking.
    """
    env = env_multi(num_agents=4)
    env.reset()

    # Run a few steps to bump everyone's count
    for _ in range(5):
        env.step(np.zeros(4, dtype=np.int64))
    assert (env._step_count == 5).all()

    # Force agent 2 to terminate next step
    original = env._is_at_princess_all
    monkeypatch.setattr(
        env, "_is_at_princess_all",
        lambda states: (lambda out: (out.__setitem__(2, True), out)[1])(original(states))
    )

    env.step(np.zeros(4, dtype=np.int64))

    # Agent 2 reset → 0; others increment to 6
    assert env._step_count[2] == 0, f"agent 2 count not reset: {env._step_count}"
    assert env._step_count[0] == 6
    assert env._step_count[1] == 6
    assert env._step_count[3] == 6
