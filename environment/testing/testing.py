import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from environment.jump_king_env import JumpKingEnv_Multi
from server.main import launch_server, launch_browser, wait_for_game, setup_game
PORT = 8000

# -- Setup Server --
launch_server()
driver = launch_browser(headless=True)  
driver.get(f"http://localhost:{PORT}")
wait_for_game(driver)
setup_game(driver)

env = JumpKingEnv_Multi(num_agents=4, driver=driver, gamma=0.999)

try:
    # Spaces are wired
    assert env.single_action_space.n == 6
    assert env.single_observation_space.shape == (9,)
    assert env.action_space.nvec.tolist() == [6, 6, 6, 6]
    assert env.observation_space.shape == (4, 9)

    # TODO!
    # ValueError: cannot reshape array of size 6000 into shape (4,10)
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
