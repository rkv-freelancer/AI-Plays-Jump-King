import pyautogui 
import time
import random

class MovementController: 
    def __init__(self, key_hold_time=0.1):
        """
        Initialize the movement controller
        
        Args:
            key_hold_time: How long to hold down a key in seconds
        """
        self.key_hold_time = key_hold_time
        self.actions = {
            'left': self._move_left,
            'right': self._move_right,
            'jump': self._jump,
            'charged_jump': self._charged_jump
        }

    def _move_left(self):
        """Press the left arrow key."""
        pyautogui.keyDown('left')
        time.sleep(self.speed)
        pyautogui.keyUp('left')
        
    def _move_right(self):
        """Press the right arrow key."""
        pyautogui.keyDown('right')
        time.sleep(self.speed)
        pyautogui.keyUp('right')
        
    def _jump(self):
        """Press the space bar to jump."""
        pyautogui.keyDown('space')
        time.sleep(self.key_hold_time)
        pyautogui.keyUp('space')
        
    def _charged_jump(self, charged_time = 0.5):
        """Press the space bar for a charged jump."""
        pyautogui.keyDown('space')
        time.sleep(charged_time)
        pyautogui.keyUp('space')
        
    def execute_action(self, action, **kwargs):
        """
        Execute a movement action
        
        Args:
            action: The action to perform (left, right, jump, charged_jump)
        """
        if action in self.actions:
            if kwargs:
                self.key_hold_time = kwargs.get('key_hold_time', self.key_hold_time)
                self.speed = kwargs.get('speed', self.key_hold_time)
                self.actions[action](**kwargs)
            else: 
                self.actions[action]()
        else:
            raise ValueError(f"Action '{action}' is not recognized.")