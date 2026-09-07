import http.server
import socketserver
import threading
import time
import os
import functools
from typing import Any

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

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
        options.add_argument("--headless=new")
        # Required when running inside a container (root user, no sandbox, limited /dev/shm)
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        # ── Renderer-crash hardening for long video captures ─────────────────
        # Capturing thousands of canvas frames back-to-back (toDataURL) is the
        # heaviest sustained load Chrome sees in this app and was OOM-killing the
        # renderer ("tab crashed") in constrained containers. These flags trim
        # memory/process overhead and disable subsystems we never use:
        options.add_argument("--disable-software-rasterizer")
        options.add_argument("--disable-gpu-compositing")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-features=Translate")
        # Allow overriding the Chrome binary (e.g. /usr/bin/chromium on Debian/Modal)
        chrome_bin = os.environ.get("CHROME_BIN")
        if chrome_bin:
            options.binary_location = chrome_bin
    options.add_argument("--mute-audio")

    # Allow overriding the chromedriver path (e.g. /usr/bin/chromedriver on Debian/Modal)
    chromedriver_bin = os.environ.get("CHROMEDRIVER_BIN")
    service = Service(chromedriver_bin) if chromedriver_bin else None

    kwargs: dict[str, Any] = {"options": options}
    if service is not None:
        kwargs["service"] = service
    return webdriver.Chrome(**kwargs)


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

    