
from dataclasses import dataclass, fields
import yaml
import os

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    
    seed: int = 42
    """seed of the experiment"""
    
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    
    wandb_entity: str = ""
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    
    num_envs: int = 1
    """the number of parallel game environments"""
    
    num_steps: int = 512
    """the number of steps to runb in each environment per policy rollout"""
    
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    
    # Must be equal to `DEFAULT_GAMMA`
    gamma: float = 0.999
    """the discount factor gamma"""
    
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    
    num_minibatches: int = 4
    """the number of mini-batches"""
    
    update_epochs: int = 4
    """the K epochs to update the policy"""
    
    norm_adv: bool = True
    """Toggles advantages normalization"""
    
    # The PPO clipping range. The policy update is constrained so the new policy can't deviate more than 10% from the old one (ratio stays in [0.9, 1.1]). This is the core stability mechanism of PPO.
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    
    # The entropy bonus weight. Adds −ent_coef × H(π) to the loss, encouraging the policy to stay stochastic (explore). Too high → random forever. Too low → collapses to one action early.
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    
    vf_coef: float = 0.5
    """coefficient of the value function"""
    
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    
    target_kl = None
    """the target KL divergence threshold"""

    # RND arguments
    # Only 25% of the mini-batch randomly updates the RND predictor each pass. Prevents the predictor from memorising the training distribution too quickly.
    update_proportion: float = 0.25
    """proportion of exp used for predictor update"""
    
    # How to weight intrinsic (RND novelty) vs extrinsic (height reward) advantage signals when combining them.
    int_coef: float = 1.0
    """coefficient of extrinsic reward"""
    
    ext_coef: float = 2.5
    """coefficient of intrinsic reward"""
    
    int_gamma: float = 0.999
    """Intrinsic reward discount rate"""
    
    num_iterations_obs_norm_init: int = 10
    """number of iterations to initialize the observations normalization parameters"""

    save_interval: int = 50
    """save a checkpoint every N policy updates"""

    # ── Video recording ──────────────────────────────────────────────────────
    capture_video: bool = False
    """if toggled, periodically record one episode and upload to W&B"""

    video_record_interval: int = 25
    """arm a fresh periodic recording every N policy updates. Once armed, the
       recorder streams frames across as many steps/updates as needed to reach
       `video_length_seconds`, then flushes one mp4 and waits for the next
       interval. Lower = denser feedback but more wall-clock spent rendering."""

    video_folder: str = "models/PPO/video"
    """staging directory for mp4 files. Kept on disk when track=False (so you can
       watch them); deleted after upload when track=True (W&B owns the copy)."""

    video_fps: int = 30
    """playback fps for the saved mp4 (game runs at 4 sim-frames per env.step,
       so 30 fps ≈ real-time visual playback when frame_skip=4). Also sets how
       many captured frames map to one second of `video_length_seconds`."""

    video_length_seconds: float = 120.0
    """target length of each PERIODIC training video. One frame is captured per
       env.step while armed, so length ≈ video_length_seconds × video_fps steps
       of footage. Frames stream straight to disk, so memory stays flat even for
       multi-minute clips. Set 60–300 for 1–5 minute videos."""

    newlevel_video_seconds: float = 30.0
    """length of the dedicated achievement clip recorded the FIRST time any agent
       reaches a brand-new global max level. Independent of the periodic recorder
       — both can run at once and share each rendered frame."""

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    """Python logging level: DEBUG / INFO / WARNING / ERROR"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    
    use_sil: bool = False
    """deploy a replay buffer for Self-Imitation Learning"""
    
    sil_update_epochs: int = 4          
    """M off-policy passes per update"""
    
    sil_batch_size:   int = 512
    
    sil_min_samples:  int = 2048        
    """warmup gate before SIL starts"""
    
    sil_value_coef:   float = 0.01      
    """weak, so it doesn't fight the PPO critic"""
    
    sil_buffer_capacity: int = 100_000  # (or total_timesteps)


def load_args(config_path: str = "configs/ppo_rnd.yml") -> Args:
    """
    Build an Args instance from a YAML config file.

    How it works:
      1. Start with the dataclass defaults (so any key absent from the YAML
         falls back to the value defined in Args).
      2. Open the YAML file and override only the keys that are present.
      3. Skip the three runtime-computed fields — those are always set in __main__.

    The config path is relative to the project root, so run the script from
    the project root:  python models/PPO/ppo_rnd_envpool.py
    """
    # Runtime fields calculated in __main__ — never read from YAML
    RUNTIME_FIELDS = {"batch_size", "minibatch_size", "num_iterations"}

    args = Args()  # start with all dataclass defaults

    if not os.path.exists(config_path):
        print(f"[load_args] Config not found at '{config_path}', using dataclass defaults.")
        return args

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    for field in fields(args):
        if field.name in RUNTIME_FIELDS:
            continue  # always computed, never loaded from file
        if field.name in cfg:
            setattr(args, field.name, cfg[field.name])

    print(f"[load_args] Loaded config from '{config_path}'")
    return args