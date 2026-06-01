# This script lives at environment/testing/testing.py (two levels deep), so
# the project root isn't on sys.path automatically. Insert it before the
# `environment.*` / `server.*` imports so those packages can resolve no
# matter where the script is invoked from.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from environment.jump_king_env import JumpKingEnv_Multi
from server.main import launch_server, launch_browser, wait_for_game, setup_game
PORT = 8000

# -- Setup Server --
# 1. Start server first — driver.get() needs it running
launch_server()

# 2. Open browser and navigate to the game
driver = launch_browser(headless=True)   # no rendering during training — saves ~30% wall time
driver.get(f"http://localhost:{PORT}")

# 3. Wait for p5.js setup() to finish and player to be created
wait_for_game(driver)

# 4. Pause the draw loop — we will step physics manually via execute_script
setup_game(driver)

env = JumpKingEnv_Multi(num_agents=4, driver=driver, gamma=0.999)

try:
    # Spaces are wired
    assert env.single_action_space.n == 6
    assert env.single_observation_space.shape == (9,)
    assert env.action_space.nvec.tolist() == [6, 6, 6, 6]
    assert env.observation_space.shape == (4, 9)

    # reset() returns the right shape and dtype
    obs, info = env.reset()
    assert obs.shape == (4, 9) and obs.dtype == np.float32

    # step() preserves the contract for 50 steps including any auto-reset events
    for t in range(50):
        actions = env.action_space.sample()
        obs, r, term, trunc, info = env.step(actions)
        assert obs.shape == (4, 9)
        assert r.shape == (4,) and r.dtype == np.float32
        assert term.shape == (4,) and term.dtype == bool
        assert trunc.shape == (4,) and trunc.dtype == bool
        assert info["final_observation"].shape == (4, 9)
        assert info["state"].shape == (4, 10)
        assert info["success"].shape == (4,)

    print("✓ vector env contract holds")
finally:
    driver.quit()
