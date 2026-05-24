import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Maps discrete action index → which input flags to hold for that step
ACTION_MAP = {
    0: {"left": False, "right": False, "jump": False},  # idle
    1: {"left": True,  "right": False, "jump": False},  # walk left
    2: {"left": False, "right": True,  "jump": False},  # walk right
    3: {"left": False, "right": False, "jump": True},   # charge jump (neutral)
    4: {"left": True,  "right": False, "jump": True},   # charge jump (left)
    5: {"left": False, "right": True,  "jump": True},   # charge jump (right)
}


class JumpKingEnv(gym.Env):
    """
    Gymnasium environment for Jump King.

    One step = `frame_skip` physics frames.

    Observation (9 floats):
        x, y, vx, vy, is_grounded, jump_timer, facing_right, level, global_height

    Action (Discrete 6):
        0: idle
        1: walk left
        2: walk right
        3: charge jump neutral
        4: charge jump left
        5: charge jump right

    Jump fires automatically when the agent switches away from a jump action —
    mirroring how the real game works (hold space to charge, release to jump).
    """

    metadata = {"render_modes": []}

    def __init__(self, driver, frame_skip=4):
        super().__init__()
        self.driver = driver
        self.frame_skip = frame_skip

        self._prev_height = 0.0
        self._prev_jump_held = False

        # 9-dimensional continuous observation
        low  = np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0, 0, 0, 0,  0],      dtype=np.float32)
        high = np.array([ np.inf,  np.inf,  np.inf,  np.inf, 1, 30, 1, 43, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = spaces.Discrete(len(ACTION_MAP))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_raw_state(self) -> dict:
        return self.driver.execute_script("""
            return {
                x:             player.currentPos.x,
                y:             player.currentPos.y,
                vx:            player.currentSpeed.x,
                vy:            player.currentSpeed.y,
                is_grounded:   player.isOnGround,
                jump_timer:    player.jumpTimer,
                facing_right:  player.facingRight,
                level:         player.currentLevelNo,
                global_height: player.GetGlobalHeight(),
                is_dead:       player.playersDead
            }
        """)

    def _to_obs(self, state: dict) -> np.ndarray:
        return np.array([
            state["x"],
            state["y"],
            state["vx"],
            state["vy"],
            float(state["is_grounded"]),
            float(state["jump_timer"]),
            float(state["facing_right"]),
            float(state["level"]),
            float(state["global_height"]),
        ], dtype=np.float32)

    def _apply_action(self, action_idx: int):
        action = ACTION_MAP[action_idx]
        curr_jump = action["jump"]

        # Jump fires when jump key transitions from held → released
        release_jump = self._prev_jump_held and not curr_jump
        self._prev_jump_held = curr_jump

        self.driver.execute_script("""
            var left  = arguments[0];
            var right = arguments[1];
            var jump  = arguments[2];
            var fire  = arguments[3];
            player.leftHeld  = left;
            player.rightHeld = right;
            player.jumpHeld  = jump;
            if (fire) { player.Jump(); }
        """, action["left"], action["right"], curr_jump, release_jump)

    def _step_frames(self, n: int):
        # Advance physics directly — p5.js draw loop is paused via noLoop()
        self.driver.execute_script("""
            var n = arguments[0];
            for (var i = 0; i < n; i++) {
                player.Update();
            }
        """, n)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=42, options=None):
        super().reset(seed=seed)
        self.driver.execute_script("player.ResetPlayer()")
        self._prev_jump_held = False
        state = self._get_raw_state()
        self._prev_height = state["global_height"]
        return self._to_obs(state), {}

    def step(self, action: int):
        self._apply_action(action)
        self._step_frames(self.frame_skip)

        state = self._get_raw_state()
        obs = self._to_obs(state)

        height_gain = state["global_height"] - self._prev_height
        self._prev_height = state["global_height"]

        # Positive reward for gaining height, small penalty per step to encourage speed
        reward = float(height_gain) - 0.01

        terminated = bool(state["is_dead"])
        truncated  = False

        return obs, reward, terminated, truncated, {"state": state}

    def close(self):
        self.driver.quit()
