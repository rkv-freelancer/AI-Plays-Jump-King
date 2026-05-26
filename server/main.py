import http.server
import socketserver
import threading
import time
import os
import functools

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

GAME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'game')
PORT = 8000

# ------------------------------------------------------------------
# Server
# ------------------------------------------------------------------

def start_server():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=GAME_DIR)
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
    Used for training — no rendering needed, maximum speed.
    """
    driver.execute_script("""
        testingSinglePlayer = true;
        noLoop();
    """)


def setup_eval_game(driver: webdriver.Chrome):
    """
    Set single-player mode and leave the p5.js draw loop running.
    Used for eval — the browser renders at its natural frame rate.
    noLoop() is NOT called, so redraw() is not needed.
    """
    driver.execute_script("testingSinglePlayer = true;")

    