import http.server
import socketserver
import threading
import time
import os

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

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

    