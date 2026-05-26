"""
Watch the trained PPO+RND agent play Jump King.

Usage (from project root):
    python eval.py                          # loads checkpoints/agent_latest.pt
    python eval.py checkpoints/agent_final.pt

Action selection:
    - Greedy  (deterministic): picks the action with the highest logit every step.
    - Stochastic (default):    samples from the policy distribution — looks more natural.

The game runs in a visible browser window so you can watch in real time.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import gymnasium as gym

from environment.jump_king_env import JumpKingEnv
from server.main import launch_server, launch_browser, wait_for_game, setup_game

# Import the model class only — no __main__ block runs on import
from models.PPO.ppo_rnd_envpool import Agent, OBS_DIM

PORT = 8000


def run_episode(env, agent, device, greedy: bool = False):
    obs_np, _ = env.reset()
    obs = torch.Tensor(obs_np).unsqueeze(0).to(device)

    total_reward = 0.0
    steps = 0

    while True:
        with torch.no_grad():
            hidden = agent.network(obs)
            logits = agent.actor(hidden)

            if greedy:
                action = logits.argmax(dim=-1)
            else:
                from torch.distributions.categorical import Categorical
                action = Categorical(logits=logits).sample()

        next_obs_np, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        steps += 1

        obs = torch.Tensor(next_obs_np).unsqueeze(0).to(device)

        if terminated or truncated:
            return total_reward, steps


def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/agent_latest.pt"
    greedy = "--greedy" in sys.argv

    print(f"Loading checkpoint: {checkpoint}")
    print(f"Action selection: {'greedy' if greedy else 'stochastic'}")

    # ── Start game ────────────────────────────────────────────────────────────
    launch_server()
    driver = launch_browser(headless=False)
    driver.get(f"http://localhost:{PORT}")
    wait_for_game(driver)
    setup_game(driver)

    env = JumpKingEnv(driver, frame_skip=4)

    # ── Load agent ────────────────────────────────────────────────────────────
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    agent = Agent(env.action_space.n).to(device)
    agent.load_state_dict(torch.load(checkpoint, map_location=device))
    agent.eval()

    print(f"Agent loaded on {device}. Running...")

    # ── Play episodes ─────────────────────────────────────────────────────────
    episode = 1
    try:
        while True:
            reward, steps = run_episode(env, agent, device, greedy=greedy)
            print(f"Episode {episode:3d} | steps={steps:5d} | total_reward={reward:.2f}")
            episode += 1
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
