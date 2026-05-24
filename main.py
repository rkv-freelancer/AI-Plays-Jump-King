import http.server 
import socketserver

# psutil (process and system utilities) is a cross-platform library for retrieving information on running processes and system utilization (CPU, memory, disks, network) in Python. It is useful mainly for system monitoring, profiling and limiting process resources and management of running processes.
import psutil

import threading 
import webbrowser
import time
import random 
import os 
from control.movement import MovementController

# Path to the Game Directory
GAME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'game')
PORT = 8000
 
# TODO! FIX THIS 
def monitor_browser_tab(browser_name="chrome"):
    """Monitor if the browser tab is closed (Command+W)"""
    print(f"Monitoring browser tab... (will exit when tab is closed)")
    
    # Get initial browser processes
    initial_browser_count = len([p for p in psutil.process_iter(['name']) 
                              if browser_name.lower() in p.info['name'].lower()])
    
    while True:
        time.sleep(2)  # Check every 2 seconds
        
        # Count current browser processes
        current_browser_count = len([p for p in psutil.process_iter(['name']) 
                                  if browser_name.lower() in p.info['name'].lower()])
        
        # If browser count decreased, assume tab was closed
        if current_browser_count < initial_browser_count:
            print("Browser tab closed. Exiting program...")
            os._exit(0)  # Force exit the program
    

def start_server(): 
    """Start a simple HTTP server to serve the game files."""
    os.chdir(GAME_DIR)
    
    # Create a handler for the HTTP server. 
    # This will serve files from the current directory.
    handler = http.server.SimpleHTTPRequestHandler
    
    # Create the server 
    # The server will listen on the specified port and serve files from the current directory
    httpd = socketserver.TCPServer(("", PORT), handler)
    
    # Print the server address
    print(f"Serving game at http://localhost:{PORT}")
    
    # Print instructions for stopping the server
    # The server will run indefinitely until interrupted
    # by the user (e.g., by pressing Ctrl+C)
    print("Press Ctrl+C to stop the server.")
    
    try: 
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server.")
        httpd.shutdown()
        
def random_movement(controller, duration=60): 
    start_time = time.time()    
    actions = ['left', 'right', 'jump', 'charged_jump']
    
    print("Starting random movements. Press Ctrl+C to stop.")
    
    try: 
        while time.time() - start_time < duration:
            # Select a random action
            action = random.choice(actions)
            
            # Execute the action with appropriate parameters
            if action == 'charged_jump':
                # Random charge time between 0.5 and 1.5 seconds
                charge_time = random.uniform(0.5, 1.5)
                print(f"Executing charged jump with charge time {charge_time:.2f}")
                controller.execute_action(action, charged_time=charge_time)
            else:
                print(f"Executing {action}")
                controller.execute_action(action)
            
            # Wait for a bit before the next action
            time.sleep(random.uniform(0.5, 1.5))  # Random delay between actions
            
    except KeyboardInterrupt:
        print("Stopping random movements.")
        return
    
def main():
    # Start the server in a separate thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Add a condition to check if the server is running
    time.sleep(1)  # Allow some time for the thread to start
    
    # Check if the server thread is alive
    if not server_thread.is_alive():
        print("Server failed to start.")
        return
    
    # Give the server some time to start
    time.sleep(2)

    # Open the game in the default web browser
    webbrowser.open(f"http://localhost:{PORT}")
    
    # Give the game time to load
    print("Waiting for game to load...")
    time.sleep(5)
    
    try:
        # Initialize the movement controller
        controller = MovementController(key_hold_time=0.1)
        
        # Start random movements
        print("Press Ctrl+C at any time to stop the program")
        random_movement(controller, duration=120)
        
    except KeyboardInterrupt:
        print("Program terminated by user.")
    finally:
        print("Shutting down...")
        # No need to explicitly stop the server thread since we made it a daemon thread
    
if __name__ == "__main__":
    main()