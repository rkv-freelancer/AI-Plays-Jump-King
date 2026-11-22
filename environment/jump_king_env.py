import base64
import io

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image

# Maps discrete action index → which input flags to hold for that step
ACTION_MAP = {
    0: {"left": False, "right": False, "jump": False},  # idle
    1: {"left": True, "right": False, "jump": False},  # walk left
    2: {"left": False, "right": True, "jump": False},  # walk right
    3: {"left": False, "right": False, "jump": True},  # charge jump (neutral)
    4: {"left": True, "right": False, "jump": True},  # charge jump (left)
    5: {"left": False, "right": True, "jump": True},  # charge jump (right)
}

# ── MDP termination / reward constants ───────────────────────────────────────
# Final level is index 42 (0-indexed; "Frame 43" in-game).
# Main platform: y=360, player height=65 → currentPos.y ≈ 295 when standing.
# Coin at (986, 306) confirms the right-side princess zone on that platform.
# SUCCESS_ZONE is generous to ensure the agent gets credit for reaching the babe.
FINAL_LEVEL = 42  # 0-indexed last level
PRINCESS_X_MIN = 930
PRINCESS_X_MAX = 1015
PRINCESS_Y_MIN = 280
PRINCESS_Y_MAX = 360

SUCCESS_REWARD = 10.0  # one-time bonus when episode ends with success
STEP_PENALTY = -0.001  # constant per-step base-MDP cost → encourages speed
# Why -0.001 and not -0.01: this must stay small relative to what one level of
# climbing pays. With Φ scaled to "levels" (see POTENTIAL_SCALE below) a level
# is worth +1.0, and a level takes roughly 200 steps. At -0.01 the step cost of
# a level would be -2.0 against a +1.0 gain — climbing would be NET-NEGATIVE and
# the optimal policy would be to end the episode as fast as possible. At -0.001
# the cost is -0.2 against +1.0, so progress clearly pays. Unlike the PBRS
# shaping term this is part of the base MDP, so changing it does change the
# optimal policy — it trades time-pressure against willingness to climb.
MAX_EPISODE_STEPS = 6000  # hard time-limit; episode is truncated after this

# Column layout of _get_raw_state_all output
COL_X, COL_Y, COL_VX, COL_VY = 0, 1, 2, 3
COL_GROUNDED, COL_JUMP_TIMER = 4, 5
COL_FACING_RIGHT, COL_LEVEL = 6, 7
COL_GLOBAL_HEIGHT, COL_IS_DEAD = 8, 9
N_RAW_STATE_COLS = 10

# ── Scaling: keep rewards AND network inputs O(1) ───────────────────────────
# The game reports everything in PIXELS. `GetGlobalHeight()` (Player.js) is
#     height − y + height × currentLevelNo
# with the p5 `height` = 900, so it spans 0–900 on level 0 and reaches ~38,700
# at the final level. Feeding those raw magnitudes into the reward and the
# network was the cause of two coupled failures:
#
#   • PBRS Φ(s) in pixels → single-step shaping rewards of ±150 and returns in
#     the hundreds → value loss 1e4–1e5, which (at vf_coef=0.5) dominated the
#     combined PPO loss so completely that the policy gradient was ~1e-8 of the
#     total. The policy stopped learning (pg≈0, kl≈0, entropy pinned).
#   • The same pixel magnitudes went into the policy network as RAW features
#     (only the RND path uses obs_rms), so `global_height` ≈ 38,700 sat next to
#     `is_grounded` ∈ {0,1} — a ~1e4 spread across input features.
#
# Both are fixed by fixed, known divisors (NOT running statistics, so there is
# nothing extra to checkpoint and eval is unaffected).
# Check image dimensions with `magick identify`
# Ex: magick identify "images/levelImages/1.png"
HEIGHT_PER_LEVEL = 900.0  # p5 `height` = canvas 950 − 50
N_LEVELS = FINAL_LEVEL + 1  # 43 levels, 0-indexed
CANVAS_WIDTH = 1200.0  # createCanvas(1200, 950)
MAX_JUMP_SPEED = 22.0  # `maxJumpSpeed` in Player.js
MAX_JUMP_TIMER = 30.0  # `maxJumpTimer` in Player.js

# Φ(s) = global_height / HEIGHT_PER_LEVEL  → units of "levels climbed".
# PBRS invariance holds for c·Φ for ANY c > 0, so this rescaling is
# theoretically free. The optimal policy is unchanged. What changes is the
# magnitude, one level of climb is now worth +1.0, which sits sensibly beside
# SUCCESS_REWARD = 10 and makes STEP_PENALTY = −0.01 meaningful again (against
# the un-scaled potential the idle drift was −0.5/step, i.e. 50× the penalty).
POTENTIAL_SCALE = HEIGHT_PER_LEVEL

# Per-column divisors for the 9-D observation, in column order:
#   x, y, vx, vy, is_grounded, jump_timer, facing_right, level, global_height
OBS_SCALE = np.array(
    [
        CANVAS_WIDTH,  # x
        HEIGHT_PER_LEVEL,  # y
        MAX_JUMP_SPEED,  # vx
        MAX_JUMP_SPEED,  # vy
        1.0,  # is_grounded (already 0/1)
        MAX_JUMP_TIMER,  # jump_timer
        1.0,  # facing_right (already 0/1)
        float(N_LEVELS),  # level
        HEIGHT_PER_LEVEL * N_LEVELS,  # global_height → ~[0, 1]
    ],
    dtype=np.float32,
)


# ── Go-Explore state snapshots ───────────────────────────────────────────────
# These mirror the JS `PlayerState` class in game/Player.js. A snapshot is the
# unit we archive and later restore, so the agent can *resume* from a promising
# state instead of re-climbing to it (Go-Explore: "first return, then explore").
#
# Position/velocity are p5.Vector objects in JS. Selenium cannot round-trip a
# p5.Vector reliably, so they are flattened into `<field>_x` / `<field>_y`
# scalars on the wire and rebuilt with createVector() on restore.
#
# IMPORTANT: x/y coordinates alone are NOT a valid snapshot — restoring without
# velocity / isOnGround / currentLevelNo gives a physically inconsistent agent
# whose dynamics won't resemble the trajectory that earned the state.
STATE_VECTOR_FIELDS = ("currentPos", "currentSpeed")
STATE_SCALAR_FIELDS = (
    "isOnGround",
    "blizzardForce",
    "blizzardForceAccelerationDirection",
    "maxBlizzardForceTimer",
    "snowImagePosition",
    "bestHeightReached",
    "bestLevelReached",
    "reachedHeightAtStepNo",
    "bestLevelReachedOnActionNo",
    "brainActionNumber",
    "currentLevelNo",
    "jumpStartingHeight",
    "facingRight",
)
# Full set of keys a snapshot dict must carry (used to validate on restore).
STATE_KEYS = (
    tuple([f"{f}_{axis}" for f in STATE_VECTOR_FIELDS for axis in ("x", "y")])
    + STATE_SCALAR_FIELDS
)

# Default discount factor for PBRS shaping. MUST match PPO's args.gamma — if
# they disagree, the Ng-Harada-Russell invariance theorem no longer applies
# and a new exploit can re-emerge. `JumpKingEnv` accepts gamma as a constructor
# argument; pass args.gamma explicitly from the training loop.
DEFAULT_GAMMA = 0.999

# env.observation_space.shape[0]
OBS_DIM = 9

# RND embedding size
EMBED_DIM = 64

# (MAX_FALL_PENALTY removed — PBRS in JumpKingEnv handles falls via telescoping;
#  no asymmetric clip is needed any more.)
EPS = 1e-5


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

    def __init__(self, driver, frame_skip=4, gamma: float = DEFAULT_GAMMA):
        super().__init__()
        self.driver = driver
        self.frame_skip = frame_skip
        # γ for the PBRS shaping term. MUST match the discount factor used by
        # the learner (PPO's args.gamma) — otherwise invariance is broken.
        self.gamma = float(gamma)

        self._prev_height = 0.0
        self._prev_jump_held = False
        self._step_count = 0

        # 9 - dimensional continuous observation
        low = np.array(
            [-np.inf, -np.inf, -np.inf, -np.inf, 0, 0, 0, 0, 0], dtype=np.float32
        )
        high = np.array(
            [np.inf, np.inf, np.inf, np.inf, 1, 30, 1, 43, np.inf], dtype=np.float32
        )
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
        return np.array(
            [
                state["x"],
                state["y"],
                state["vx"],
                state["vy"],
                float(state["is_grounded"]),
                float(state["jump_timer"]),
                float(state["facing_right"]),
                float(state["level"]),
                float(state["global_height"]),
            ],
            dtype=np.float32,
        )

    def _apply_action(self, action_idx: int):
        action = ACTION_MAP[action_idx]
        curr_jump = action["jump"]

        # Jump fires when jump key transitions from held → released
        release_jump = self._prev_jump_held and not curr_jump
        self._prev_jump_held = curr_jump

        self.driver.execute_script(
            """
            var left  = arguments[0];
            var right = arguments[1];
            var jump  = arguments[2];
            var fire  = arguments[3];
            player.leftHeld  = left;
            player.rightHeld = right;
            player.jumpHeld  = jump;
            if (fire) { player.Jump(); }
        """,
            action["left"],
            action["right"],
            curr_jump,
            release_jump,
        )

    def _step_frames(self, n: int):
        # Advance physics directly — p5.js draw loop is paused via noLoop()
        self.driver.execute_script(
            """
            var n = arguments[0];
            for (var i = 0; i < n; i++) {
                player.Update();
            }
        """,
            n,
        )

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
        self._step_count = 0
        state = self._get_raw_state()
        self._prev_height = state["global_height"]
        return self._to_obs(state), {}

    def step(self, action: int):
        self._apply_action(action)
        self._step_frames(self.frame_skip)
        self._step_count += 1

        state = self._get_raw_state()
        obs = self._to_obs(state)

        # ── Reward ────────────────────────────────────────────────────────────
        # Potential-Based Reward Shaping with Φ(s) = global_height(s):
        #
        #     F(s, a, s') = γ · Φ(s') − Φ(s)
        #                 = γ · h(s') − h(s)
        #                 ≡ height_gain − (1 − γ) · h(s')
        #
        # The (1 − γ)·h(s') correction is what makes the climb-fall oscillation
        # exploit telescope to ≈ 0 over a closed loop — no clip needed.
        h_prev = self._prev_height
        h_curr = float(state["global_height"])
        self._prev_height = h_curr

        shaping_reward = self.gamma * h_curr - h_prev  # F(s, a, s')
        base_reward = STEP_PENALTY  # base R: time pressure

        # ── Termination ───────────────────────────────────────────────────────
        success = self._is_at_princess(state)
        fell_out = bool(state["is_dead"])  # fell off bottom of level 0
        terminated = success or fell_out

        if success:
            base_reward += SUCCESS_REWARD  # sparse terminal bonus — part of base R

        reward = base_reward + shaping_reward

        # ── Truncation ────────────────────────────────────────────────────────
        truncated = self._step_count >= MAX_EPISODE_STEPS

        return (
            obs,
            reward,
            terminated,
            truncated,
            {
                "state": state,
                "success": success,
                "shaping_reward": shaping_reward,  # exposed for diagnostics / W&B
                "base_reward": base_reward,
            },
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

        JPEG (not PNG): encoding a 1200×950 PNG thousands of times back-to-back
        is the dominant renderer-memory cost during a long video capture and was
        crashing the Chrome tab ("tab crashed" = renderer OOM) in constrained
        containers. JPEG at q=0.85 cuts per-call memory/CPU ~5-10× and is visually
        indistinguishable for diagnostic playback.
        """
        self.driver.execute_script("redraw()")
        data_url = self.driver.execute_script(
            "return document.querySelector('canvas').toDataURL('image/jpeg', 0.85);"
        )
        # Strip the "data:image/jpeg;base64," prefix
        jpg_b64 = data_url.split(",", 1)[1]
        jpg_bytes = base64.b64decode(jpg_b64)
        frame = np.array(Image.open(io.BytesIO(jpg_bytes)).convert("RGB"))
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

    # TODO: Changed from 4 -> 1
    def __init__(self, num_agents, driver, frame_skip=1, gamma: float = DEFAULT_GAMMA):
        super().__init__()

        self.driver = driver
        self.frame_skip = frame_skip
        self.gamma = float(gamma)
        self.num_agents = num_agents

        # updating the current list
        self._prev_height = np.zeros(num_agents, dtype=np.float32)
        self._prev_jump_held = np.zeros(num_agents, dtype=bool)
        self._step_count = np.zeros(num_agents, dtype=np.int32)

        # ── Go-Explore hook (optional) ───────────────────────────────────────
        # Set this from the training loop to enable archived restarts:
        #
        #     env.restore_fn = lambda done_idxs: {i: archive.sample() for i in
        #                                         done_idxs if random() < beta}
        #
        # It is called inside step() with the array of agents that just
        # finished, and must return {agent_idx: snapshot_dict} (or {}). Agents
        # it omits get the normal level-0 start — keeping a real share of
        # true-start episodes is what preserves a policy that can play the game
        # from the beginning. Left as None, the env behaves exactly as before.
        self.restore_fn = None

        # 9 - dimensional continuous observation for each agent.
        # NOTE: bounds are in SCALED units — every column is divided by
        # OBS_SCALE before it leaves the env, so the bounded columns
        # (is_grounded, jump_timer, facing_right, level) all top out at 1.0.
        low = np.array(
            [-np.inf, -np.inf, -np.inf, -np.inf, 0, 0, 0, 0, 0], dtype=np.float32
        )
        high = np.array(
            [np.inf, np.inf, np.inf, np.inf, 1, 1, 1, 1, np.inf], dtype=np.float32
        )

        self.single_observation_space = spaces.Box(
            low=low, high=high, shape=(9,), dtype=np.float32
        )
        self.single_action_space = spaces.Discrete(len(ACTION_MAP))

        # Batched (vectorised) space
        self.observation_space = spaces.Box(
            low=np.tile(low, (num_agents, 1)),
            high=np.tile(high, (num_agents, 1)),
            shape=(num_agents, 9),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([len(ACTION_MAP)] * num_agents)

        # Create the JS Population once and bind it to BOTH names that matter:
        #
        #   • the lexical global `population` (declared `let population` in
        #     sketch.js) — this is what draw() and Player.Show() reference, so
        #     it MUST point at our agents or the canvas keeps rendering the
        #     unrelated 600-player GA population created in setup().
        #   • window.population — what every other script in this env reads
        #     (_get_raw_state_all, _apply_action_all, _step_frames_all, reset…).
        #
        # The bare `population = ...` assignment (no var/let/window) resolves up
        # the scope chain to sketch.js's lexical global and reassigns it. Then we
        # alias window.population to the same object so both views stay in sync.
        # Finally flip draw() into RL render mode (draws every agent, no GA step).
        self.driver.execute_script(
            """
            population = new Population(arguments[0]);
            window.population = population;
            rlRenderMode = true;
            """,
            self.num_agents,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_raw_state_all(self) -> np.ndarray:

        flat = self.driver.execute_script("""
            const out = new Float32Array(window.population.players.length * 10);
            for (let idx = 0; idx < window.population.players.length; idx++) {
                const player = window.population.players [idx]; 
                const o = idx * 10; 
                
                out[o + 0] = player.currentPos.x;
                out[o + 1] = player.currentPos.y; 
                out[o + 2] = player.currentSpeed.x;
                out[o + 3] = player.currentSpeed.y;
                out[o + 4] = player.isOnGround;
                out[o + 5] = player.jumpTimer;
                out[o + 6] = player.facingRight;
                out[o + 7] = player.currentLevelNo;
                out[o + 8] = player.GetGlobalHeight();
                out[o + 9] = player.playersDead;
            }
            
            return Array.from(out);
            """)

        return np.asarray(flat, dtype=np.float32).reshape(self.num_agents, 10)

    def _get_obs_all(self, states: np.ndarray) -> np.ndarray:
        """
        Returns (N, 9) float32 — drops the is_dead column and applies OBS_SCALE.

        The division returns a NEW array rather than a view into `states`, which
        matters twice over: `states` must stay in RAW game units for
        `_is_at_princess_all` (its thresholds are pixel/level values) and for
        `info["state"]` (the trainer reads COL_LEVEL as an integer level), and
        step() mutates the returned obs at the done-agent slots.
        """
        assert states.shape == (
            self.num_agents,
            10,
        ), f"expected ({self.num_agents}, 10) states, got {states.shape}"
        return (states[:, :9] / OBS_SCALE).astype(np.float32)

    def _potential_all(self, states: np.ndarray) -> np.ndarray:
        """
        Φ(s) for every agent, in "levels climbed" units.

        Single source of truth for the PBRS potential — reset(), _reset_agents(),
        set_state() and step() all go through here, so `_prev_height` and
        `h_curr` can never drift into different units (which would silently
        corrupt every shaping reward).
        """
        return (states[:, COL_GLOBAL_HEIGHT] / POTENTIAL_SCALE).astype(np.float32)

    def _apply_action_all(self, action_idxs: np.ndarray) -> None:
        """
        Apply N actions in a single Selenium round-trip.

        action_idxs : shape (N,), each entry an index into ACTION_MAP (0..5)
        """
        assert (
            len(action_idxs) == self.num_agents
        ), f"expected {self.num_agents} actions, got {len(action_idxs)}"

        # Decompose the N action indices into the 4 flag arrays the JS expects.
        lefts = np.empty(self.num_agents, dtype=bool)
        rights = np.empty(self.num_agents, dtype=bool)
        jumps = np.empty(self.num_agents, dtype=bool)
        fires = np.empty(self.num_agents, dtype=bool)

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
            lefts.tolist(),
            rights.tolist(),
            jumps.tolist(),
            fires.tolist(),
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
        const players = window.population.players;
        const n = arguments[0];
        for (let f = 0; f < n; f++) {
            for (let idx = 0; idx < players.length; idx++) {
                players[idx].Update();
            }
        }
        """,
            n,
        )

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
            (states[:, COL_LEVEL] == FINAL_LEVEL)  # level
            & (states[:, COL_X] >= PRINCESS_X_MIN)
            & (states[:, COL_X] <= PRINCESS_X_MAX)  # x
            & (states[:, COL_Y] >= PRINCESS_Y_MIN)
            & (states[:, COL_Y] <= PRINCESS_Y_MAX)  # y
        )

    def render(self):
        """
        Returns a H x W x 3 uint8 numpy array of the current canvas content.

        Uses canvas.toDataURL() (base64 PNG) rather than driver.get_screenshot_as_png()
        because:
          • toDataURL returns *just the canvas* — no browser chrome to crop out
          • it's ~5× faster than a full Selenium screenshot in headless Chrome
          • it works identically in headless and headed modes

        We trigger redraw() first because noLoop() pauses the p5.js draw loop —
        physics advances via player.Update() but the canvas pixels only refresh
        when redraw() is explicitly called.

        JPEG (not PNG): encoding a 1200 × 950 PNG thousands of times back-to-back
        is the dominant renderer-memory cost during a long video capture and was
        crashing the Chrome tab ("tab crashed" = renderer OOM) in constrained
        containers. JPEG at q = 0.85 cuts per-call memory/CPU ~5-10× and is visually
        indistinguishable for diagnostic playback.
        """
        self.driver.execute_script("redraw()")
        data_url = self.driver.execute_script(
            "return document.querySelector('canvas').toDataURL('image/jpeg', 0.85);"
        )
        # Strip the "data:image/jpeg;base64," prefix
        jpg_b64 = data_url.split(",", 1)[1]
        jpg_bytes = base64.b64decode(jpg_b64)
        frame = np.array(Image.open(io.BytesIO(jpg_bytes)).convert("RGB"))
        return frame

    # ------------------------------------------------------------------
    # Go-Explore: state snapshot / restore
    # ------------------------------------------------------------------

    def get_state(self, idx: int) -> dict:
        """
        Capture agent `idx`'s full dynamic state as a plain, JSON-safe dict.

        Reuses the game's own `PlayerState.getStateFromPlayer()` so the captured
        field set stays in sync with the game rather than being re-derived here.

        This is the snapshot you archive when an agent first reaches a new level.
        Capture is rare (once per newly-discovered cell), so a dedicated
        round-trip per call is fine — never call this every step.
        """
        state = self.driver.execute_script(
            """
            const p = window.population.players[arguments[0]];
            const s = new PlayerState();
            s.getStateFromPlayer(p);
            return {
                currentPos_x:                       s.currentPos.x,
                currentPos_y:                       s.currentPos.y,
                currentSpeed_x:                     s.currentSpeed.x,
                currentSpeed_y:                     s.currentSpeed.y,
                isOnGround:                         s.isOnGround,
                blizzardForce:                      s.blizzardForce,
                blizzardForceAccelerationDirection: s.blizzardForceAccelerationDirection,
                maxBlizzardForceTimer:              s.maxBlizzardForceTimer,
                snowImagePosition:                  s.snowImagePosition,
                bestHeightReached:                  s.bestHeightReached,
                bestLevelReached:                   s.bestLevelReached,
                reachedHeightAtStepNo:              s.reachedHeightAtStepNo,
                bestLevelReachedOnActionNo:         s.bestLevelReachedOnActionNo,
                brainActionNumber:                  s.brainActionNumber,
                currentLevelNo:                     s.currentLevelNo,
                jumpStartingHeight:                 s.jumpStartingHeight,
                facingRight:                        s.facingRight
            };
            """,
            int(idx),
        )
        missing = [k for k in STATE_KEYS if k not in state]
        assert not missing, f"get_state() snapshot missing keys: {missing}"
        return state

    def _apply_restores(self, idxs: list, snapshots: list) -> None:
        """
        Restore a batch of archived snapshots in ONE Selenium round-trip.

        ── Why ResetPlayer() comes first (this ordering is load-bearing) ──────
        `PlayerState.loadStateToPlayer()` restores position, velocity, level and
        the blizzard/progression bookkeeping — but it does NOT touch:

            playersDead, jumpTimer, jumpHeld, leftHeld, rightHeld,
            hasBumped, hasFallen, previousSpeed

        Restores fire when an agent is *done*, which usually means it just died.
        Loading a snapshot into a still-dead player leaves `playersDead = true`,
        so step() reads it as terminated again immediately → every restored
        agent dies on its first step, forever.

        Calling ResetPlayer() first wipes all of the above (including the death
        flag, the jump timer and the held-input flags, which would otherwise
        leak in from the previous episode); loadStateToPlayer() then overwrites
        position/velocity/level from the snapshot. Reset clears, snapshot fills.
        """
        assert len(idxs) == len(snapshots), "idxs and snapshots must align"
        if not idxs:
            return

        for s in snapshots:
            missing = [k for k in STATE_KEYS if k not in s]
            assert not missing, f"snapshot missing keys: {missing}"

        self.driver.execute_script(
            """
            const players   = window.population.players;
            const idxs      = arguments[0];
            const snapshots = arguments[1];

            for (let j = 0; j < idxs.length; j++) {
                const p = players[idxs[j]];
                const d = snapshots[j];

                // Clear FIRST — playersDead / jumpTimer / held-input flags are
                // not covered by loadStateToPlayer(). See the Python docstring.
                p.ResetPlayer();

                const s = new PlayerState();
                s.currentPos   = createVector(d.currentPos_x,   d.currentPos_y);
                s.currentSpeed = createVector(d.currentSpeed_x, d.currentSpeed_y);
                s.isOnGround                         = d.isOnGround;
                s.blizzardForce                      = d.blizzardForce;
                s.blizzardForceAccelerationDirection = d.blizzardForceAccelerationDirection;
                s.maxBlizzardForceTimer              = d.maxBlizzardForceTimer;
                s.snowImagePosition                  = d.snowImagePosition;
                s.bestHeightReached                  = d.bestHeightReached;
                s.bestLevelReached                   = d.bestLevelReached;
                s.reachedHeightAtStepNo              = d.reachedHeightAtStepNo;
                s.bestLevelReachedOnActionNo         = d.bestLevelReachedOnActionNo;
                s.brainActionNumber                  = d.brainActionNumber;
                s.currentLevelNo                     = d.currentLevelNo;
                s.jumpStartingHeight                 = d.jumpStartingHeight;
                s.facingRight                        = d.facingRight;

                s.loadStateToPlayer(p);
            }
            """,
            [int(i) for i in idxs],
            snapshots,
        )

    def set_state(self, idx: int, snapshot: dict) -> np.ndarray:
        """
        Restore a single agent from a snapshot and return its fresh (9,) obs.

        Convenience wrapper for tests / eval. During training the restore is
        driven through `_reset_agents(restore_states=...)`, which also re-syncs
        the PBRS `_prev_height` bookkeeping — see that method.
        """
        self._apply_restores([idx], [snapshot])
        self._prev_jump_held[idx] = False
        self._step_count[idx] = 0

        states = self._get_raw_state_all()
        # Re-sync the shaping baseline to the RESTORED height (see _reset_agents).
        self._prev_height[idx] = self._potential_all(states)[idx]
        return self._get_obs_all(states)[idx]

    # ------------------------------------------------------------------
    # Gymnasium API (Custom for Parallel)
    # ------------------------------------------------------------------

    def _reset_agents(
        self, done_idxs: np.ndarray, restore_states: dict | None = None
    ) -> np.ndarray:
        """
        Internal per-agent reset — called from step() to recycle the agents
        that just terminated/truncated. Other agents are left untouched.

        Parameters
        ----------
        done_idxs : np.ndarray of agent indices to reset (shape (k,), k ≤ N)
        restore_states : optional {agent_idx: snapshot_dict}. Agents present in
                        this mapping are RESTORED from an archived Go-Explore
                        snapshot instead of being sent back to level 0. Agents
                        absent from it get the normal fresh start. Pass None
                        (the default) for plain vector-env behaviour.

        Returns
        -------
        fresh_obs_for_done : np.ndarray shape (k, 9) — the new starting obs of
                            each reset agent, to be substituted into the
                            step()-return obs at those slots.

        ── PBRS bookkeeping (why the ordering below matters) ──────────────────
        `_prev_height` is the Φ(s) baseline for the shaping term
        F = γ·Φ(s') − Φ(s). It MUST be re-synced to whatever height the agent
        actually starts at. Restore an agent to height H while `_prev_height`
        still reads 0 and its very first shaping reward becomes γ·H − 0 — a
        huge fabricated reward that poisons the value function and silently
        wrecks training. So the order is always:
            restore/reset  →  re-read states  →  set _prev_height from that read
        which is what the tail of this method does for every reset agent,
        restored or not.
        """

        if len(done_idxs) == 0:
            return np.empty((0, 9), dtype=np.float32)

        # Split the done agents into "restore from archive" vs "fresh start".
        restore_states = restore_states or {}
        restore_idxs = [int(i) for i in done_idxs if int(i) in restore_states]
        plain_idxs = [int(i) for i in done_idxs if int(i) not in restore_states]

        # Fresh starts — back to level 0.
        if plain_idxs:
            self.driver.execute_script(
                """
                const players   = window.population.players;
                const idxs      = arguments[0];
                for (const i of idxs) {
                    players[i].ResetPlayer()
                }
                """,
                plain_idxs,
            )

        # Archived restarts — ResetPlayer() then loadStateToPlayer(), batched.
        if restore_idxs:
            self._apply_restores(
                restore_idxs, [restore_states[i] for i in restore_idxs]
            )

        # Reset Python-side per agent state
        self._prev_jump_held[done_idxs] = False
        self._step_count[done_idxs] = 0

        # Re-read states so we can return fresh starting obs and update prev_height.
        # This single read covers BOTH fresh starts (height ≈ 0) and archived
        # restores (height = the snapshot's height), so the shaping baseline is
        # correct for either path — no special-casing needed.
        states = self._get_raw_state_all()
        self._prev_height[done_idxs] = self._potential_all(states)[done_idxs]

        return self._get_obs_all(states)[done_idxs]

    def reset(self, seed: int = 42, options=None):
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
        const players = window.population.players;
        for (let i = 0; i < players.length; i++) {
            players[i].ResetPlayer();
        }
        """)

        # Reset all per-agent Python state
        self._prev_jump_held[:] = False
        self._step_count[:] = 0

        states = self._get_raw_state_all()
        self._prev_height = self._potential_all(states).copy()
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
        h_curr = self._potential_all(states).copy()  # Φ(s') in "levels" units

        self._prev_height = h_curr.copy()

        shaping_rewards = self.gamma * h_curr - h_prev
        base_rewards = np.full(self.num_agents, STEP_PENALTY, dtype=np.float32)

        # ── Termination
        # ───────────────────────────────────────────────────────

        success = self._is_at_princess_all(states).astype(bool)
        fell_out = states[:, COL_IS_DEAD].astype(bool)  # is_dead column -> bool

        terminated = success | fell_out

        base_rewards[success] += SUCCESS_REWARD
        rewards = base_rewards + shaping_rewards

        # ── Truncation ────────────────────────────────────────────────────────
        truncated = self._step_count >= MAX_EPISODE_STEPS
        done = terminated | truncated

        # ── Reset ────────────────────────────────────────────────────────
        # Reset the agents
        done_idxs = np.where(done)[0]
        # Ask the Go-Explore hook (if installed) which done agents should resume
        # from an archived snapshot rather than restarting at level 0.
        restore_states = self.restore_fn(done_idxs) if self.restore_fn else None
        fresh_obs = self._reset_agents(done_idxs, restore_states=restore_states)

        final_obs = obs.copy()
        state_snapshot = states.copy()

        # Return observations with terminal states
        obs[done_idxs] = fresh_obs

        return (
            obs,
            rewards,
            terminated,
            truncated,
            {
                "state": state_snapshot,
                "final_observation": final_obs,
                "success": success,
                "shaping_reward": shaping_rewards,
                "base_reward": base_rewards,
            },
        )
