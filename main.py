import http.server
import socketserver
import threading
import time
import os

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from environment.jump_king_env import JumpKingEnv

GAME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'game')
PORT = 8000

# ------------------------------------------------------------------
# Server
# ------------------------------------------------------------------

def start_server():
    os.chdir(GAME_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    # allow_reuse_address prevents "address already in use" on restart
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Game server running at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.shutdown()


def launch_server() -> threading.Thread:
    thread = threading.Thread(target=start_server, daemon=True)
    thread.start()
    time.sleep(1)  # give the socket time to bind
    if not thread.is_alive():
        raise RuntimeError("Server thread failed to start.")
    return thread


# ------------------------------------------------------------------
# Browser / Selenium
# ------------------------------------------------------------------

def launch_browser(headless: bool = False) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--mute-audio")
    driver = webdriver.Chrome(options=options)
    return driver


def wait_for_game(driver: webdriver.Chrome, timeout: int = 5):
    """Poll until the `p5.js` setup() has run and `player` exists in JS."""
    print("Waiting for game to load...")
    
    deadline = time.time() + timeout
    while time.time() < deadline:
        ready = driver.execute_script(
            "return typeof player !== 'undefined' && player !== null"
        )
        if ready:
            print("Game loaded.")
            return
        time.sleep(0.5)
    raise TimeoutError("Game did not load within the timeout period.")


def setup_game(driver: webdriver.Chrome):
    """
    Pause the p5.js draw loop so we control every physics step manually.
    Also ensure the game is in single-player mode.
    """
    driver.execute_script("""
        testingSinglePlayer = true;
        noLoop();
    """)


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
