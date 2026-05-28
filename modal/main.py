"""
Modal cloud training for Jump King PPO+RND.

Usage (from project root):
    modal run modal/main.py               # train with default config
    modal run modal/main.py --detach      # fire-and-forget (returns immediately)

First time only:
    pip install modal
    modal token new          # authenticate with your Modal account

Checkpoints are saved to a persistent Modal Volume named 'jump-king-checkpoints'.
To download after a run:
    modal volume get jump-king-checkpoints checkpoints/ ./checkpoints
"""

import modal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # repo root

app = modal.App("jump-king-ppo-rnd")

# ── Container image ───────────────────────────────────────────────────────────
# Modal 1.x removed modal.Mount.  Local files are now attached to the Image via
# .add_local_dir(local_path, remote_path=...) which mounts them at container
# *startup* (no image rebuild on source changes — fast iteration).
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "chromium",           # headless browser for the game
        "chromium-driver",    # matching chromedriver binary
        "libgomp1",           # OpenMP (required by stable-baselines3 / PyTorch)
    )
    .pip_install(
        "gymnasium>=0.29",
        "stable-baselines3>=2.0",
        "selenium>=4.0",
        "torch>=2.0",
        "pyyaml>=6.0",
        "tensorboard",
        "wandb",
        "numpy",
        # Video recording: PIL decodes canvas PNGs, imageio + ffmpeg write mp4
        "Pillow",
        "imageio",
        "imageio-ffmpeg",
    )
    # Attach project source — added at container startup, no layer rebuild needed
    # when these change.  Put add_local_dir LAST per Modal docs (no further build
    # steps allowed after, unless copy=True).
    .add_local_dir(str(PROJECT_ROOT / "game"),        remote_path="/root/project/game")
    .add_local_dir(str(PROJECT_ROOT / "environment"), remote_path="/root/project/environment")
    .add_local_dir(str(PROJECT_ROOT / "server"),      remote_path="/root/project/server")
    .add_local_dir(str(PROJECT_ROOT / "models"),      remote_path="/root/project/models")
    .add_local_dir(str(PROJECT_ROOT / "configs"),     remote_path="/root/project/configs")
)

# ── Persistent volume for checkpoints ────────────────────────────────────────
checkpoints_vol = modal.Volume.from_name(
    "jump-king-checkpoints", create_if_missing=True
)


# ── Training function ─────────────────────────────────────────────────────────
@app.function(
    image=training_image,
    gpu="T4",              # ~$1.10/hr — swap to "T4" to save cost
    timeout=3600 * 10,        # 10-hour hard cap
    volumes={"/root/project/checkpoints": checkpoints_vol},
    secrets=[
        modal.Secret.from_dict({
            "CHROME_BIN":       "/usr/bin/chromium",
            "CHROMEDRIVER_BIN": "/usr/bin/chromedriver",
        }),
        modal.Secret.from_name("wandb-secret"),   # WANDB_API_KEY — create with:
                                                  # modal secret create wandb-secret WANDB_API_KEY=<key>
    ],
)
def train():
    import subprocess
    import sys

    try:
        subprocess.run(
            [sys.executable, "models/PPO/ppo_rnd_envpool.py"],
            cwd="/root/project",
            check=True,
        )
    finally:
        # Commit even on crash / early stop so partial checkpoints survive
        checkpoints_vol.commit()


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    # Change GPU 
    print("Submitting training job to Modal (GPU: T4)...")
    train.remote()
    print(
        "Done.\n"
        "Download checkpoints with:\n"
        "  modal volume get jump-king-checkpoints checkpoints/ ./checkpoints"
    )