"""
Shared pytest fixtures for the Jump King test suite.

Two fixture categories:

1.  Browser-backed (session-scoped): launches the local game server and a
    headless Chromium driver *once* per pytest session. Tests that need a real
    env interact via the `env_multi` factory below. Guarded by the
    `@pytest.mark.browser` marker so a plain `pytest` invocation skips them.

2.  Mock-driver (function-scoped): a stand-in for the Selenium driver that
    returns synthesized JS state. Lets us exercise PBRS math, termination
    logic, and shape contracts without paying the ~30 ms / call browser tax.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

# Make project-root imports work regardless of where pytest is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ────────────────────────────────────────────────────────────────────────────
# Browser-backed fixtures  (only used by @pytest.mark.browser tests)
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def driver():
    """
    Session-scoped Selenium driver pointed at the locally-served game.

    Launches once, torn down at the end of the pytest session. Tests that
    need a fresh game state should call `driver.execute_script("...ResetPlayer...")`
    or just re-create the env (which calls reset() in __init__).

    Skips the whole test (rather than failing) if Chromium isn't available —
    keeps CI from blocking on environment issues.
    """
    try:
        from server.main import launch_server, launch_browser, wait_for_game, setup_game
    except ImportError as e:
        pytest.skip(f"server.main unavailable: {e}")

    launch_server()
    try:
        d = launch_browser(headless=True)
    except Exception as e:
        pytest.skip(f"browser launch failed: {e}")

    d.get("http://localhost:8000")
    wait_for_game(d)
    setup_game(d)

    yield d

    try:
        d.quit()
    except Exception:
        pass  # best-effort teardown


@pytest.fixture
def env_multi(driver) -> Callable[[int], Any]:
    """
    Factory for fresh `JumpKingEnv_Multi` instances.

    Returns a callable: `make_env = env_multi`, then `env = make_env(num_agents=4)`.
    Each call gives a freshly-reset env wired to the session driver.

    Tracks instances so we can close them at function-scope teardown.
    """
    from environment.jump_king_env import JumpKingEnv_Multi

    created: list = []

    def _factory(num_agents: int = 4, gamma: float = 0.999, frame_skip: int = 4):
        env = JumpKingEnv_Multi(
            num_agents=num_agents,
            driver=driver,
            frame_skip=frame_skip,
            gamma=gamma,
        )
        created.append(env)
        return env

    yield _factory

    # Envs share the session driver, so we don't quit it — just drop refs.
    created.clear()


# ────────────────────────────────────────────────────────────────────────────
# Mock-driver fixtures  (used by fast unit tests, no browser)
# ────────────────────────────────────────────────────────────────────────────

class MockDriver:
    """
    Minimal Selenium-driver lookalike. Captures every `execute_script` call
    and returns scripted responses based on the script's content.

    Tests configure `set_states(states)` to control what _get_raw_state_all
    sees. Apply-action and step-frames scripts are accepted as no-ops.

    Why not unittest.mock.MagicMock?
      MagicMock would let any call succeed without checking what JS was
      actually invoked. This class makes it easy to assert on the exchange
      ("the env tried to apply these actions; here is the state I'll fake back")
      without becoming a full JS interpreter.
    """

    def __init__(self, num_agents: int = 4):
        self.num_agents = num_agents
        # Default state: all agents at origin, alive, level 0
        self._next_states = np.zeros((num_agents, 10), dtype=np.float32)
        # log of (script, args) for assertion in tests
        self.calls: list[tuple[str, tuple]] = []

    def set_states(self, states: np.ndarray) -> None:
        """Inject the next _get_raw_state_all return value. Shape: (N, 10)."""
        assert states.shape == (self.num_agents, 10), (
            f"states shape must be ({self.num_agents}, 10); got {states.shape}"
        )
        self._next_states = states.astype(np.float32, copy=True)

    def execute_script(self, script: str, *args):
        self.calls.append((script, args))

        # Route by content. The env's scripts are small and string-distinctive.
        if "Float32Array" in script and "GetGlobalHeight" in script:
            # _get_raw_state_all → flat list of N*10 floats
            return self._next_states.flatten().tolist()
        if "ResetPlayer" in script:
            return None
        if "new Population" in script:
            return None
        if "leftHeld" in script:
            return None  # _apply_action_all
        if "Update()" in script:
            return None  # _step_frames_all
        if "toDataURL" in script:
            # Return a 1x1 transparent PNG so render() doesn't crash if called
            return ("data:image/png;base64,"
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
                    "AAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==")
        if "redraw" in script:
            return None

        # Unknown script — fail loudly so the test author sees the surprise
        raise AssertionError(
            f"MockDriver got an unexpected script:\n{script[:200]}..."
        )


@pytest.fixture
def mock_driver():
    """Fresh MockDriver per test (no shared state between tests)."""
    return MockDriver(num_agents=4)


@pytest.fixture
def env_multi_mocked(mock_driver):
    """
    `JumpKingEnv_Multi` wired to a MockDriver. Lets us test reward / termination
    logic without a real browser. Constructor is monkey-patched so it skips
    the Population-creation JS round-trip (mock_driver accepts it anyway).
    """
    from environment.jump_king_env import JumpKingEnv_Multi

    env = JumpKingEnv_Multi(
        num_agents=mock_driver.num_agents,
        driver=mock_driver,
        frame_skip=4,
        gamma=0.999,
    )
    return env
