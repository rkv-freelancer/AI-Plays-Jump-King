import base64
import io

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image

# Maps discrete action index → which input flags to hold for that step
ACTION_MAP = {
    0: {"left": False, "right": False, "jump": False},  # idle
    1: {"left": True,  "right": False, "jump": False},  # walk left
    2: {"left": False, "right": True,  "jump": False},  # walk right
    3: {"left": False, "right": False, "jump": True},   # charge jump (neutral)
    4: {"left": True,  "right": False, "jump": True},   # charge jump (left)
    5: {"left": False, "right": True,  "jump": True},   # charge jump (right)
}

# ── MDP termination / reward constants ───────────────────────────────────────
# Final level is index 42 (0-indexed; "Frame 43" in-game).
# Main platform: y=360, player height=65 → currentPos.y ≈ 295 when standing.
# Coin at (986, 306) confirms the right-side princess zone on that platform.
# SUCCESS_ZONE is generous to ensure the agent gets credit for reaching the babe.
FINAL_LEVEL       = 42      # 0-indexed last level
PRINCESS_X_MIN    = 900     # princess is on the right portion of the platform
PRINCESS_Y_MAX    = 340     # player y ≤ this when standing on (or near) the platform
SUCCESS_REWARD    = 10.0    # one-time bonus when episode ends with success
STEP_PENALTY      = -0.01   # constant per-step base-MDP cost → encourages speed
MAX_EPISODE_STEPS = 2000    # hard time-limit; episode is truncated after this

# Default discount factor for PBRS shaping. MUST match PPO's args.gamma — if
# they disagree, the Ng-Harada-Russell invariance theorem no longer applies
# and a new exploit can re-emerge. JumpKingEnv accepts gamma as a constructor
# argument; pass args.gamma explicitly from the training loop.
DEFAULT_GAMMA     = 0.999


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

    Termination / truncation:
        terminated  – agent reaches the princess zone on the final level (level 42),
                      OR the player falls off the bottom of level 0 (playersDead).
        truncated   – episode reaches MAX_EPISODE_STEPS without termination.

    Reward (Potential-Based Reward Shaping; Ng, Harada, Russell 1999):

        Per-step reward = F(s, a, s') + base R(s, a, s')

        Shaping term  F = γ·Φ(s') − Φ(s)   with Φ(s) = global_height(s)
                        = height_gain − (1 − γ)·h(s')
                      → telescopes over any trajectory to γ^T·h(s_T) − h(s_0),
                        so the climb-fall oscillation exploit cancels out and
                        no asymmetric clip (MAX_FALL_PENALTY) is needed.

        Base term  R   = STEP_PENALTY           every step (time pressure)
                       + SUCCESS_REWARD          on princess-zone termination

        Critical: do NOT also add height_gain, a princess-distance bonus, or
        any other non-PBRS dense reward on top — combining two shaping sources
        breaks the invariance theorem and re-opens reward hacking.

    Jump fires automatically when the agent switches away from a jump action —
    mirroring how the real game works (hold space to charge, release to jump).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, driver, frame_skip=4, gamma: float = DEFAULT_GAMMA):
        super().__init__()
        self.driver     = driver
        self.frame_skip = frame_skip
        # γ for the PBRS shaping term. MUST match the discount factor used by
        # the learner (PPO's args.gamma) — otherwise invariance is broken.
        self.gamma      = float(gamma)

        self._prev_height     = 0.0
        self._prev_jump_held  = False
        self._step_count      = 0

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
        action    = ACTION_MAP[action_idx]
        curr_jump = action["jump"]

        # Jump fires when jump key transitions from held → released
        release_jump         = self._prev_jump_held and not curr_jump
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

    def _is_at_princess(self, state: dict) -> bool:
        """
        True when the agent has entered the success zone on the final level.

        The princess is a background sprite on level 42 (0-indexed).
        There is no JS princess object; the zone is derived from level geometry:
          - Main platform at y=360; player y ≈ 295 when standing on it.
          - Coin at (986, 306) confirms the right-side princess area.
          - PRINCESS_X_MIN=900, PRINCESS_Y_MAX=340 gives a generous landing box.

        To calibrate: run the game manually, reach the princess, and print
        state["x"] / state["y"] to tighten these bounds if needed.
        """
        return (
            int(state["level"]) == FINAL_LEVEL
            and float(state["x"]) >= PRINCESS_X_MIN
            and float(state["y"]) <= PRINCESS_Y_MAX
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=42, options=None):
        super().reset(seed=seed)
        self.driver.execute_script("player.ResetPlayer()")
        self._prev_jump_held = False
        self._step_count     = 0
        state                = self._get_raw_state()
        self._prev_height    = state["global_height"]
        return self._to_obs(state), {}

    def step(self, action: int):
        self._apply_action(action)
        self._step_frames(self.frame_skip)
        self._step_count += 1

        state = self._get_raw_state()
        obs   = self._to_obs(state)

        # ── Reward ────────────────────────────────────────────────────────────
        # Potential-Based Reward Shaping with Φ(s) = global_height(s):
        #
        #     F(s, a, s') = γ · Φ(s') − Φ(s)
        #                 = γ · h(s') − h(s)
        #                 ≡ height_gain − (1 − γ) · h(s')
        #
        # The (1 − γ)·h(s') correction is what makes the climb-fall oscillation
        # exploit telescope to ≈ 0 over a closed loop — no clip needed.
        h_prev            = self._prev_height
        h_curr            = float(state["global_height"])
        self._prev_height = h_curr

        shaping_reward = self.gamma * h_curr - h_prev          # F(s, a, s')
        base_reward    = STEP_PENALTY                           # base R: time pressure

        # ── Termination ───────────────────────────────────────────────────────
        success    = self._is_at_princess(state)
        fell_out   = bool(state["is_dead"])    # fell off bottom of level 0
        terminated = success or fell_out

        if success:
            base_reward += SUCCESS_REWARD   # sparse terminal bonus — part of base R

        reward = base_reward + shaping_reward

        # ── Truncation ────────────────────────────────────────────────────────
        truncated = (self._step_count >= MAX_EPISODE_STEPS)

        return obs, reward, terminated, truncated, {
            "state":          state,
            "success":        success,
            "shaping_reward": shaping_reward,   # exposed for diagnostics / W&B
            "base_reward":    base_reward,
        }

    def render(self):
        """
        Returns a HxWx3 uint8 numpy array of the current canvas content.

        Uses canvas.toDataURL() (base64 PNG) rather than driver.get_screenshot_as_png()
        because:
          • toDataURL returns *just the canvas* — no browser chrome to crop out
          • it's ~5× faster than a full Selenium screenshot in headless Chrome
          • it works identically in headless and headed modes

        We trigger redraw() first because noLoop() pauses the p5.js draw loop —
        physics advances via player.Update() but the canvas pixels only refresh
        when redraw() is explicitly called.
        """
        self.driver.execute_script("redraw()")
        data_url = self.driver.execute_script(
            "return document.querySelector('canvas').toDataURL('image/png');"
        )
        # Strip the "data:image/png;base64," prefix (always 22 chars for PNG)
        png_b64  = data_url.split(",", 1)[1]
        png_bytes = base64.b64decode(png_b64)
        frame    = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
        return frame

    def close(self):
        self.driver.quit()
