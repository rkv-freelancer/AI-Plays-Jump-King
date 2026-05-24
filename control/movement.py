import pyautogui as pg
import time
import random

class MovementController: 
    def __init__(self, key_hold_time=0.1, time_delay=0.1):
        """
        Initialize the movement controller
        
        Args:
            key_hold_time: How long to hold down a key in seconds
        """
        self.time_delay = time_delay  # Default speed for movement actions
        self.key_hold_time = key_hold_time # Default time to hold down a key
        
        # Define the actions and their corresponding methods
        self.actions = {
            'left': self._move_left,
            'right': self._move_right,
            'jump': self._jump,
            'charged_jump': self._charged_jump
        }

    def _move_left(self):
        """Press the left arrow key."""
        pg.keyDown('left')
        time.sleep(self.time_delay)
        pg.keyUp('left')
        
    def _move_right(self):
        """Press the right arrow key."""
        pg.keyDown('right')
        time.sleep(self.time_delay)
        pg.keyUp('right')
        
    def _jump(self):
        """Press the space bar to jump."""
        pg.keyDown('space')
        time.sleep(self.key_hold_time)
        pg.keyUp('space')
        
    def _charged_jump(self, charged_time = 0.5):
        """Press the space bar for a charged jump."""
        pg.keyDown('space')
        time.sleep(charged_time)
        pg.keyUp('space')
        
    def execute_action(self, action, **kwargs):
        """
        Execute a movement action
        
        Args:
            action: The action to perform (left, right, jump, charged_jump)
        """
        if action in self.actions:
            if kwargs:
                self.key_hold_time = kwargs.get('key_hold_time', self.key_hold_time)
                self.time_delay = kwargs.get('speed', self.key_hold_time)
                self.actions[action](**kwargs)
            else: 
                self.actions[action]()
        else:
            raise ValueError(f"Action '{action}' is not recognized.")