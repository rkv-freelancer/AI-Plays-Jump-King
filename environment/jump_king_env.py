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

# Column layout of _get_raw_state_all output
COL_X, COL_Y, COL_VX, COL_VY     = 0, 1, 2, 3
COL_GROUNDED, COL_JUMP_TIMER     = 4, 5
COL_FACING_RIGHT, COL_LEVEL      = 6, 7
COL_GLOBAL_HEIGHT, COL_IS_DEAD   = 8, 9
N_RAW_STATE_COLS                 = 10

# Default discount factor for PBRS shaping. MUST match PPO's args.gamma — if
# they disagree, the Ng-Harada-Russell invariance theorem no longer applies
# and a new exploit can re-emerge. JumpKingEnv accepts gamma as a constructor
# argument; pass args.gamma explicitly from the training loop.
DEFAULT_GAMMA     = 0.999

class JumpKingEnv_Single(gym.Env):
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

    def __init__(self, driver, frame_skip = 4, gamma: float = DEFAULT_GAMMA):
        super().__init__()
        self.driver     = driver
        self.frame_skip = frame_skip
        # γ for the PBRS shaping term. MUST match the discount factor used by
        # the learner (PPO's args.gamma) — otherwise invariance is broken.
        self.gamma      = float(gamma)

        self._prev_height     = 0.0
        self._prev_jump_held  = False
        self._step_count      = 0

        # 9 - dimensional continuous observation
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

    def reset(self, seed = 42, options = None):
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


class JumpKingEnv_Multi(gym.Env):
    """
    Gymnasium parallel environment for Jump King.

    One step = `frame_skip` physics frames.

    Observation (num_agents, 9 floats):
        x, y, vx, vy, is_grounded, jump_timer, facing_right, level, global_height
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    def __init__(self, num_agents, driver, frame_skip = 4, gamma: float = DEFAULT_GAMMA):
        super().__init__()
        
        self.driver = driver
        self.frame_skip = frame_skip
        self.gamma = float(gamma)
        self.num_agents = num_agents
        
        # updating the current list
        self._prev_height     = np.zeros(num_agents, dtype=np.float32)
        self._prev_jump_held  = np.zeros(num_agents, dtype=bool)
        self._step_count      = np.zeros(num_agents, dtype=np.int32)
        
        # 9 - dimensional continuous observation for each agent
        low  = np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0, 0, 0, 0,  0],      dtype = np.float32)
        high = np.array([ np.inf,  np.inf,  np.inf,  np.inf, 1, 30, 1, 43, np.inf], dtype = np.float32)
        
        self.single_observation_space = spaces.Box(low=low, high=high, shape=(9, ), dtype = np.float32)
        self.single_action_space = spaces.Discrete(len(ACTION_MAP))
        
        # Batched (vectorised) space
        self.observation_space = spaces.Box(
            low = np.tile(low, (num_agents, 1)), 
            high = np.tile(high, (num_agents, 1)), 
            shape = (num_agents, 9), 
            dtype = np.float32, 
        )
        self.action_space = spaces.MultiDiscrete([len(ACTION_MAP)] * num_agents)
        
        # Create the JS Population once, on the window so it persists across calls
        self.driver.execute_script(
            "window.population = new Population(arguments[0]);",
            self.num_agents
        )
        
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
        
    def _get_raw_state_all(self) -> np.ndarray:
        
        flat = self.driver.execute_script(
            """
            const out = new Float32Array(population.players.length * 10);
            for (let idx = 0; idx < population.players.length; idx++) {
                const player = population.players [idx]; 
                const o = idx * 10; 
                
                out[o + 0] = player.currentPos.x, 
                out[o + 1] = player.currentPos.y, 
                out[o + 2] = player.currentSpeed.x,
                out[o + 3] = player.currentSpeed.y,
                out[o + 4] = player.isOnGround,
                out[o + 5] = player.jumpTimer,
                out[o + 6] = player.facingRight,
                out[o + 7] = player.currentLevelNo,
                out[o + 8] = player.GetGlobalHeight(),
                out[o + 9] = player.playersDead
            }
            
            return Array.from(out)
            """
        )
        
        return np.asarray(flat, dtype = np.float32).reshape(self.num_agents, 10)
            
    def _get_obs_all(self, states: np.ndarray) -> np.ndarray:
        """Returns (N, 9) float32 — drops the is_dead column at index 9."""
        assert states.shape == (self.num_agents, 10), (
            f"expected ({self.num_agents}, 10) states, got {states.shape}"
        )
        return states[:, :9]   # already float32, already (N, 9)
                        
    def _apply_action_all(self, action_idxs: np.ndarray) -> None: 
        """
        Apply N actions in a single Selenium round-trip.

        action_idxs : shape (N,), each entry an index into ACTION_MAP (0..5)
        """
        assert len(action_idxs) == self.num_agents, (
            f"expected {self.num_agents} actions, got {len(action_idxs)}"
        )
        
        # Decompose the N action indices into the 4 flag arrays the JS expects. 
        lefts  = np.empty(self.num_agents, dtype=bool)
        rights = np.empty(self.num_agents, dtype=bool)
        jumps  = np.empty(self.num_agents, dtype=bool)
        fires  = np.empty(self.num_agents, dtype=bool)
        
        for k, idx in enumerate(action_idxs): 
            a = ACTION_MAP[int(idx)]
            lefts[k] = a["left"]
            rights[k] = a["right"]
            jumps[k] = a["jump"]  
                      
            # Jump fires when jump key transitions held → released per agent
            fires[k] = bool(self._prev_jump_held[k] and not a["jump"])
            
        # Update per-agent `prev_jump_held` state for next step 
        self._prev_jump_held = jumps.copy()
        
        # One round-trip — Selenium can't marshal numpy arrays directly, so .tolist()
        self.driver.execute_script(
            """
            const lefts = arguments[0];
            const rights = arguments[1];
            const jumps = arguments[2]; 
            const fires = arguments[3];
            const players = window.population.players;
            
            for ( let idx = 0; idx < players.length; idx++ ){ 
                const player = players[idx]; 
                player.leftHeld  = lefts[idx];
                player.rightHeld = rights[idx];
                player.jumpHeld  = jumps[idx];
                if (fires[idx]) { player.Jump(); }
            }
            """, 
            lefts.tolist(), rights.tolist(), jumps.tolist(), fires.tolist()
        )

    def _step_frames_all(self, n: int): 
        """
        Advance physics for all N players in lockstep:
            for each frame f in [0..n):
                for each player p:
                    p.Update()
        """
        
        self.driver.execute_script(
        """
        const players = population.players;
        const n = arguments[0];
        for (let f = 0; f < n; f++) {
            for (let idx = 0; idx < players.length; idx++) {
                players[idx].Update();
            }
        }
        """, n)
        
    def _is_at_princess_all(self, states: np.ndarray) -> np.ndarray:
        """
        True when the agent has entered the success zone on the final level.

        The princess is a background sprite on level 42 (0-indexed).
        There is no JS princess object; the zone is derived from level geometry:
          - Main platform at y = 360; player y ≈ 295 when standing on it.
          - Coin at (986, 306) confirms the right-side princess area.
          - PRINCESS_X_MIN = 900, PRINCESS_Y_MAX = 340 gives a generous landing box.

        To calibrate: run the game manually, reach the princess, and print
        state["x"] / state["y"] to tighten these bounds if needed.
        """
    
        return (
            (states[:, COL_LEVEL] == FINAL_LEVEL)       # level
            & (states[:, COL_X] >= PRINCESS_X_MIN)  # x
            & (states[:, COL_Y] <= PRINCESS_Y_MAX)  # y
        )

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
        
    # ------------------------------------------------------------------
    # Gymnasium API (Custom for Parallel)
    # ------------------------------------------------------------------
    
    def _reset_agents(self, done_idxs: np.ndarray) -> np.ndarray: 
        """
        Internal per-agent reset — called from step() to recycle the agents
        that just terminated/truncated. Other agents are left untouched.

        Parameters
        ----------
        done_idxs : np.ndarray of agent indices to reset (shape (k,), k ≤ N)

        Returns
        -------
        fresh_obs_for_done : np.ndarray shape (k, 9) — the new starting obs of
                            each reset agent, to be substituted into the
                            step()-return obs at those slots.
        """
        
        if len(done_idxs) == 0: 
            return np.empty((0, 9), dtype = np.float32)
        
        # Reset just dead agents in JS
        self.driver.execute_script(
            """
            const players   = population.players; 
            const idxs      = arguments[0]; 
            for (const i of idxs) { 
                players[i].ResetPlayer()
            }
            """, done_idxs.tolist()
        )
        
        # Reset Python-side per agent state
        self._prev_jump_held[done_idxs] = False
        self._step_count[done_idxs]     = 0
        
        # Re-read states so we can return fresh starting obs and update prev_height
        states = self._get_raw_state_all()
        self._prev_height[done_idxs] = states[done_idxs, COL_GLOBAL_HEIGHT] # global_height column
        
        return self._get_obs_all(states)[done_idxs]
        
    def reset(self, seed: int = 42, options = None):
        """
        Resets ALL N agents. Called once at the start of training
        and any time the user explicitly wants a clean slate.

        Returns
        -------
        obs : np.ndarray of shape (N, 9), dtype float32
        info : dict
        """
        
        super().reset(seed=seed)
        
        # Reset all N players in JS — single round-trip
        self.driver.execute_script("""
        const players = population.players;
        for (let i = 0; i < players.length; i++) {
            players[i].ResetPlayer();
        }
        """)
        
        # Reset all per-agent Python state
        self._prev_jump_held[:] = False
        self._step_count[:]     = 0
        
        states = self._get_raw_state_all()
        self._prev_height = states[:, COL_GLOBAL_HEIGHT].copy() # global_height column
        obs = self._get_obs_all(states)
        
        return obs, {}
    
    def step(self, action_idxs: np.ndarray): 
        self._apply_action_all(action_idxs)
        self._step_frames_all(self.frame_skip)
        self._step_count += 1
        
        states = self._get_raw_state_all()
        obs = self._get_obs_all(states)
        
        # ── Reward 
        # ───────────────────────────────────────────────────────
        h_prev = self._prev_height
        h_curr = states[:, COL_GLOBAL_HEIGHT].copy()    # global_height
        
        self._prev_height = h_curr.copy()
        
        shaping_rewards = self.gamma * h_curr - h_prev
        base_rewards    = np.full(self.num_agents, STEP_PENALTY, dtype = np.float32)                     
        
        # ── Termination 
        # ───────────────────────────────────────────────────────
        
        success = self._is_at_princess_all(states)
        fell_out = states[:, COL_IS_DEAD].astype(bool)    # is_dead column -> bool

        terminated = success | fell_out
        
        base_rewards[success] += SUCCESS_REWARD
        rewards = base_rewards + shaping_rewards
            
        # ── Truncation ────────────────────────────────────────────────────────
        truncated = self._step_count >= MAX_EPISODE_STEPS
        done = terminated | truncated
        
        # ── Reset ────────────────────────────────────────────────────────
        # Reset the agents
        done_idxs = np.where(done)[0]
        fresh_obs = self._reset_agents(done_idxs)
        final_obs = obs.copy()
        
        obs[done_idxs] = fresh_obs
        
        return obs, rewards, terminated, truncated, {
            "state": states,
            "final_observation": final_obs,
            "success": success, 
            "shaping_reward": shaping_rewards, 
            "base_reward": base_rewards              
                                                     }