"""
Watch the trained PPO+RND agent play Jump King.

Usage (from project root):
    python eval.py                            # loads checkpoints/agent_latest.pt
    python eval.py checkpoints/agent_final.pt
    python eval.py --greedy                   # deterministic action selection
    python eval.py checkpoints/agent_final.pt --greedy

How rendering works:
    Training uses noLoop() — the draw loop is paused and Python manually steps physics.
    Eval uses setup_eval_game() — the draw loop runs naturally at ~60fps.
    We inject input flags and sleep; p5.js handles both physics and rendering.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # macOS: suppress duplicate libomp conflict

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.distributions.categorical import Categorical

from environment.jump_king_env import JumpKingEnv
from server.main import launch_server, launch_browser, wait_for_game, setup_eval_game

from models.PPO.ppo_rnd_envpool import Agent

PORT = 8000

ACTION_NAMES = ["idle", "walk_left", "walk_right", "jump_neutral", "jump_left", "jump_right"]


def run_episode(env, agent, device, greedy: bool = False):
    obs_np, _ = env.reset()
    obs = torch.Tensor(obs_np).unsqueeze(0).to(device)

    total_reward = 0.0
    steps        = 0
    prev_height  = float(obs_np[8])   # global_height is index 8 in the obs vector

    while True:
        # ── Choose action ─────────────────────────────────────────────────────
        with torch.no_grad():
            hidden = agent.network(obs)
            logits = agent.actor(hidden)

            if greedy:
                action = logits.argmax(dim=-1)
            else:
                action = Categorical(logits=logits).sample()

        action_idx = action.item()

        # ── Inject input flags — draw loop advances physics + renders ─────────
        # We do NOT call player.Update() manually here.
        # The p5.js draw loop is running at ~60fps and handles that itself.
        # The two Selenium execute_script calls below (~10-15ms total) provide
        # enough natural pacing — no extra sleep needed.
        env._apply_action(action_idx)

        # ── Read state ────────────────────────────────────────────────────────
        state    = env._get_raw_state()
        obs_np   = env._to_obs(state)
        obs      = torch.Tensor(obs_np).unsqueeze(0).to(device)

        height_gain  = state["global_height"] - prev_height
        prev_height  = state["global_height"]
        reward       = float(height_gain) - 0.01
        total_reward += reward
        steps        += 1

        # ── Debug print every 30 steps ────────────────────────────────────────
        if steps % 30 == 0:
            print(
                f"  step={steps:4d} | "
                f"action={ACTION_NAMES[action_idx]:14s} | "
                f"height={state['global_height']:.1f} | "
                f"reward={reward:.3f}"
            )

        terminated = bool(state.get("is_dead", False))
        if terminated:
            return total_reward, steps


def main():
    # ── Parse args ────────────────────────────────────────────────────────────
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    checkpoint = positional[0] if positional else "checkpoints/agent_latest.pt"
    greedy     = "--greedy" in sys.argv

    print(f"Checkpoint     : {checkpoint}")
    print(f"Action mode    : {'greedy (argmax)' if greedy else 'stochastic (sample)'}")

    # ── Start game ────────────────────────────────────────────────────────────
    launch_server()
    driver = launch_browser(headless=False)
    driver.get(f"http://localhost:{PORT}")
    wait_for_game(driver)
    setup_eval_game(driver)   # single-player mode, draw loop stays ON

    env = JumpKingEnv(driver, frame_skip=4)

    # ── Load agent ────────────────────────────────────────────────────────────
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    agent  = Agent(env.action_space.n).to(device)
    agent.load_state_dict(torch.load(checkpoint, map_location=device))
    agent.eval()

    print(f"Agent loaded on {device}. Running...\n")

    # ── Play episodes ─────────────────────────────────────────────────────────
    episode = 1
    try:
        while True:
            reward, steps = run_episode(env, agent, device, greedy=greedy)
            print(f"Episode {episode:3d} | steps={steps:5d} | total_reward={reward:.2f}\n")
            episode += 1
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
