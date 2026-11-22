"""
Pure-Python tests for the PBRS reward shaping inside JumpKingEnv_Multi.step().

Uses the MockDriver fixture so these tests run in ~1 ms each — no browser, no
network. The point is to lock down the math: any silent regression in the
shaping formula (e.g. someone re-factors `γ * h_curr - h_prev` to
`γ * (h_curr - h_prev)`) will fail the telescoping check immediately.

What's covered:
 - per-step PBRS formula: F = γ·h_curr − h_prev
 - telescoping over a climb-fall cycle (~0 net reward → exploit closed)
 - SUCCESS_REWARD bonus applied to successful agents
 - terminated/truncated flag separation
 - per-agent independence: agent k's reward depends only on agent k's height
"""

from __future__ import annotations

import numpy as np
import pytest

from environment.jump_king_env import (
    STEP_PENALTY,
    SUCCESS_REWARD,
    FINAL_LEVEL,
    PRINCESS_X_MIN,
    PRINCESS_Y_MAX,
    MAX_EPISODE_STEPS,
    POTENTIAL_SCALE,
)


def _phi(heights):
    """
    The env's potential Φ(s) for a raw pixel height.

    The env divides `global_height` by POTENTIAL_SCALE so shaping rewards land
    at O(1) instead of O(100s of pixels). Tests express their expectations
    through this helper so they keep locking down the *formula* and stay valid
    if the scale constant is ever retuned.
    """
    return np.asarray(heights, dtype=np.float64) / POTENTIAL_SCALE


# Convenience: column indices into the (N, 10) raw-state array
COL_X, COL_Y, COL_VX, COL_VY        = 0, 1, 2, 3
COL_GROUNDED, COL_JUMP_TIMER         = 4, 5
COL_FACING_RIGHT, COL_LEVEL          = 6, 7
COL_GLOBAL_HEIGHT, COL_IS_DEAD       = 8, 9


def _state_batch(heights, levels=None, xs=None, ys=None, is_dead=None,
                 num_agents=4):
    """Build a (N, 10) raw-state array with the fields the tests care about."""
    s = np.zeros((num_agents, 10), dtype=np.float32)
    s[:, COL_GLOBAL_HEIGHT] = heights
    if levels  is not None: s[:, COL_LEVEL]   = levels
    if xs      is not None: s[:, COL_X]       = xs
    if ys      is not None: s[:, COL_Y]       = ys
    if is_dead is not None: s[:, COL_IS_DEAD] = is_dead
    return s


# ────────────────────────────────────────────────────────────────────────────
# 1. The per-step PBRS formula  F = γ·h_curr − h_prev  (the regression locker)
# ────────────────────────────────────────────────────────────────────────────

def test_pbrs_formula_matches_gamma_times_h_curr_minus_h_prev(env_multi_mocked, mock_driver):
    """
    Lock down the exact shaping formula. If anyone refactors it to the
    `γ·(h_curr − h_prev)` form (the exploit-reopening bug) this test fires.
    """
    env = env_multi_mocked
    gamma = env.gamma

    # Reset establishes _prev_height = Φ(s) at reset time (in scaled units)
    mock_driver.set_states(_state_batch(heights=[100.0, 200.0, 300.0, 400.0]))
    env.reset()
    prev_h = env._prev_height.copy()
    assert np.allclose(prev_h, _phi([100.0, 200.0, 300.0, 400.0]))

    # Next step: heights move by +10, −5, 0, +20
    new_heights = np.array([110.0, 195.0, 300.0, 420.0], dtype=np.float32)
    mock_driver.set_states(_state_batch(heights=new_heights))

    actions = np.zeros(4, dtype=np.int64)
    obs, rewards, terminated, truncated, info = env.step(actions)

    # Expected per-agent reward = STEP_PENALTY + (γ·Φ(s') − Φ(s))
    expected_shaping = gamma * _phi(new_heights) - prev_h
    expected_reward  = STEP_PENALTY + expected_shaping

    np.testing.assert_allclose(rewards, expected_reward, rtol=1e-5, atol=1e-5)


def test_pbrs_uses_correct_asymmetric_form_not_the_exploit_form(env_multi_mocked, mock_driver):
    """
    The bug we *don't* want: F = γ·(h_curr − h_prev) = γ·height_gain.
    Asserting the correct asymmetric form via a single explicit case.
    """
    env = env_multi_mocked
    gamma = env.gamma

    # Start at height 1000 everywhere
    mock_driver.set_states(_state_batch(heights=[1000.0]*4))
    env.reset()

    # Climb +50 px — expected shaping = γ·Φ(1050) − Φ(1000)
    mock_driver.set_states(_state_batch(heights=[1050.0]*4))
    _, rewards, *_ = env.step(np.zeros(4, dtype=np.int64))

    phi_prev, phi_curr = _phi(1000.0), _phi(1050.0)
    correct_shaping = gamma * phi_curr - phi_prev
    # the broken formula's value (γ·height_gain) — must NOT match
    exploit_shaping = gamma * (phi_curr - phi_prev)

    expected = STEP_PENALTY + correct_shaping
    assert np.allclose(rewards, expected, rtol=1e-5)

    # Hard fail if the exploit form ever sneaks back in
    exploit_reward = STEP_PENALTY + exploit_shaping
    assert not np.allclose(rewards, exploit_reward, rtol=1e-5), (
        "REGRESSION: shaping looks like γ·height_gain — climb-fall exploit is back."
    )


# ────────────────────────────────────────────────────────────────────────────
# 2. Telescoping  — the climb-fall exploit must NOT pay out
# ────────────────────────────────────────────────────────────────────────────

def test_climb_fall_cycle_telescopes_to_near_zero(env_multi_mocked, mock_driver):
    """
    PBRS's whole point: a closed loop (climb to h, then fall back to h₀) must
    pay roughly zero total shaping. Specifically: γ^T·h_T − h_0 ≈ 0 if h_T = h_0.
    """
    env = env_multi_mocked
    gamma = env.gamma

    mock_driver.set_states(_state_batch(heights=[0.0]*4))
    env.reset()

    # Climb: 0 → 100 → 200 → 300
    # Fall:  300 → 200 → 100 → 0
    trajectory = [100.0, 200.0, 300.0, 200.0, 100.0, 0.0]

    total_shaping = np.zeros(4, dtype=np.float64)
    for h in trajectory:
        mock_driver.set_states(_state_batch(heights=[h]*4))
        _, rewards, *_ = env.step(np.zeros(4, dtype=np.int64))
        # Strip the STEP_PENALTY to isolate shaping contribution
        shaping = rewards.astype(np.float64) - STEP_PENALTY
        total_shaping += shaping

    # Telescoping bound: total = γ^T·Φ_T − Φ_0 = γ·0 − 0 = 0  (per agent)
    # Slack covers the (1 − γ) corrections accumulated mid-trajectory:
    #   Σ (1−γ)·Φ_t = 0.001 × Φ(100+200+300+200+100+0) ≈ 0.001 × 1.0 ≈ 1e-3
    # Bound is derived from the scale so it stays tight if POTENTIAL_SCALE moves.
    bound = 10.0 * (1.0 - gamma) * _phi(300.0) * len(trajectory)
    assert np.all(np.abs(total_shaping) < max(bound, 1e-6)), (
        f"climb-fall cycle did NOT telescope; total shaping = {total_shaping}. "
        f"PBRS may be broken — check the formula in step()."
    )


# ────────────────────────────────────────────────────────────────────────────
# 3. Terminal bonus and termination separation
# ────────────────────────────────────────────────────────────────────────────

def test_success_reward_applied_only_to_successful_agents(env_multi_mocked, mock_driver):
    """
    SUCCESS_REWARD must be added to base_reward for the agents that
    reached the princess zone — and nobody else.
    """
    env = env_multi_mocked

    mock_driver.set_states(_state_batch(heights=[100.0]*4))
    env.reset()

    # Agent 2 reaches the princess zone (level=FINAL, x ≥ X_MIN, y ≤ Y_MAX);
    # others stay at level 0.
    states = _state_batch(
        heights=[110.0]*4,
        levels =[0, 0, FINAL_LEVEL, 0],
        xs     =[0, 0, PRINCESS_X_MIN + 50, 0],
        ys     =[400, 400, PRINCESS_Y_MAX - 10, 400],
    )
    mock_driver.set_states(states)

    _, rewards, terminated, _, info = env.step(np.zeros(4, dtype=np.int64))

    assert terminated.tolist() == [False, False, True, False]
    assert info["success"][2] == True
    assert info["success"][0] == False

    # Agent 2's reward = shaping + STEP_PENALTY + SUCCESS_REWARD
    # Agents 0/1/3:    = shaping + STEP_PENALTY   (no bonus)
    diff = rewards[2] - rewards[0]   # difference is exactly SUCCESS_REWARD (shaping equal)
    assert np.isclose(diff, SUCCESS_REWARD, atol=1e-5), (
        f"agent 2's reward delta vs others should be SUCCESS_REWARD={SUCCESS_REWARD}, got {diff}"
    )


def test_fell_out_terminates_without_success_bonus(env_multi_mocked, mock_driver):
    """
    is_dead=True must terminate the agent but NOT award SUCCESS_REWARD.
    """
    env = env_multi_mocked

    mock_driver.set_states(_state_batch(heights=[500.0]*4))
    env.reset()

    # Agent 1 falls out (is_dead=True), height drops to 0
    states = _state_batch(
        heights=[510.0, 0.0, 500.0, 500.0],
        is_dead=[0,    1,   0,    0],
    )
    mock_driver.set_states(states)

    _, rewards, terminated, _, info = env.step(np.zeros(4, dtype=np.int64))

    assert terminated[1] == True
    assert info["success"][1] == False
    # No SUCCESS_REWARD added to agent 1
    assert rewards[1] < SUCCESS_REWARD, (
        f"fell-out agent should not receive SUCCESS_REWARD; got reward={rewards[1]}"
    )


def test_truncation_at_max_episode_steps(env_multi_mocked, mock_driver):
    """
    Once _step_count reaches MAX_EPISODE_STEPS, truncated must fire.
    Verified with one agent only — the array logic is identical for all N.
    """
    env = env_multi_mocked
    mock_driver.set_states(_state_batch(heights=[0.0]*4))
    env.reset()

    # Fast-forward _step_count just below the threshold
    env._step_count[:] = MAX_EPISODE_STEPS - 1

    mock_driver.set_states(_state_batch(heights=[0.0]*4))
    _, _, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.int64))

    assert truncated.all(), f"all agents should be truncated; got {truncated}"
    assert not terminated.any(), "no terminations (no princess, no death)"


# ────────────────────────────────────────────────────────────────────────────
# 4. Per-agent independence
# ────────────────────────────────────────────────────────────────────────────

def test_one_agents_height_does_not_affect_another_agents_reward(env_multi_mocked, mock_driver):
    """
    Agent k's PBRS reward must depend on agent k's height alone. If a refactor
    accidentally uses a shared max-height somewhere, this test fires.
    """
    env = env_multi_mocked
    gamma = env.gamma

    mock_driver.set_states(_state_batch(heights=[100.0, 100.0, 100.0, 100.0]))
    env.reset()

    # Only agent 0 climbs; everyone else holds height
    mock_driver.set_states(_state_batch(heights=[200.0, 100.0, 100.0, 100.0]))
    _, rewards_a, *_ = env.step(np.zeros(4, dtype=np.int64))

    # Reset and try the symmetric case: only agent 3 climbs
    mock_driver.set_states(_state_batch(heights=[100.0]*4))
    env.reset()
    mock_driver.set_states(_state_batch(heights=[100.0, 100.0, 100.0, 200.0]))
    _, rewards_b, *_ = env.step(np.zeros(4, dtype=np.int64))

    # The climbing agent's reward in each case should be identical (positions swapped)
    assert np.isclose(rewards_a[0], rewards_b[3], rtol=1e-5), (
        "PBRS appears to be coupling agents — agent 0 climbing should yield the "
        "same reward as agent 3 climbing by the same amount."
    )
    # Non-climbing agents should get the *same* reward in both cases too
    assert np.isclose(rewards_a[1], rewards_b[1], rtol=1e-5)
    assert np.isclose(rewards_a[2], rewards_b[2], rtol=1e-5)
