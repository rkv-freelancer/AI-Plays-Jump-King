import http.server
import socketserver
import threading
import time
import os

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from environment.jump_king_env import JumpKingEnv
from server.main import launch_server, launch_browser, wait_for_game, setup_game

GAME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'game')
PORT = 8000

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    # 1. Start server first — driver.get() needs it running
    launch_server()

    # 2. Open browser and navigate to the game
    driver = launch_browser(headless=False)
    driver.get(f"http://localhost:{PORT}")

    # 3. Wait for p5.js setup() to finish and player to be created
    wait_for_game(driver)

    # 4. Pause the draw loop — we will step physics manually via execute_script
    setup_game(driver)

    # 5. Create the Gymnasium environment
    env = JumpKingEnv(driver, frame_skip=4)

    # ------------------------------------------------------------------
    # Sanity check: run one random-action episode and print state info
    # ------------------------------------------------------------------
    obs, info = env.reset()
    print(f"Initial obs: {obs}")

    total_reward = 0.0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 50 == 0:
            s = info["state"]
            print(f"  step={step:3d}  height={s['global_height']:.1f}  "
                  f"level={s['level']}  reward={reward:.3f}")

        if terminated or truncated:
            print(f"Episode ended at step {step}. Total reward: {total_reward:.2f}")
            break

    print(f"Done. Total reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
