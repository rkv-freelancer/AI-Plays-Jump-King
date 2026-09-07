# Docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/#ppo_rnd_envpoolpy

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # macOS: suppress duplicate libomp conflict

import sys
import random
import time
import logging
from pathlib import Path

# Video recording (imageio + ffmpeg) is lazy-imported inside the rollout loop so
# that runs with `capture_video = False` have zero extra dependency footprint.

# Ensure the project root is on sys.path so `environment` and `server` resolve
# regardless of which directory VSCode / the terminal launches from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym

from stable_baselines3.common.running_mean_std import RunningMeanStd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# Environment (RL & GPU)
from environment.jump_king_env import JumpKingEnv_Multi
from environment.jump_king_env import COL_LEVEL, OBS_DIM, EMBED_DIM, EPS

# Utilities 
from utilities.Memory import Memory
from utilities.VideoStreamer import VideoStreamer
from utilities.Args import Args, load_args

# Server 
from server.main import launch_server, launch_browser, wait_for_game, setup_game



# ALGO LOGIC: initialize agent here:
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # The weight matrix is set so its rows are mutually perpendicular (orthogonal) with a magnitude controlled by std. This prevents the gradient signal from vanishing or exploding in early training. Gradients flow evenly through the network rather than amplifying or shrinking per layer.
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    
    

class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(OBS_DIM, 256)),
            nn.ReLU(), 
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(), 
        )
        
        self.extra_layer = nn.Sequential(
            layer_init(nn.Linear(256, 256), std=0.1), 
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(256, 256), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(256, num_actions), std=0.01),
        )
        
        self.critic_ext = layer_init(nn.Linear(256, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(256, 1), std=0.01)


    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        features = self.extra_layer(hidden)
        
        if action is None:
            action = probs.sample()
        
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic_ext(features + hidden),
            self.critic_int(features + hidden),
        )


    def get_value(self, x):
        hidden = self.network(x)
        features = self.extra_layer(hidden)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)
    
    
    def calc_sil_loss(self, obs, actions, returns, weights, value_coef):
        # 1. Re-evaluate under the CURRENT policy/value (returns is frozen history)
        _, logprob, _entropy, ext_value, _int_value = self.get_action_and_value(obs, actions)
        
        # (B,1) → (B,), match returns
        ext_value = ext_value.view(-1)                 

        # 2. Clipped advantage  A⁺ = max(R − V, 0)      (V stays in-graph here)
        adv          = returns - ext_value
        adv_clipped  = adv.clamp(min = 0.0)

        # 3. Policy loss: −logπ(a|s)·A⁺, A⁺ as a DETACHED weight, IS-weighted
        sil_policy_loss = -(weights * logprob * adv_clipped.detach()).mean()

        # 4. Value loss: ½(A⁺)², grad flows through V, IS-weighted
        sil_value_loss  = (weights * 0.5 * adv_clipped.pow(2)).mean()

        # 5. Combine
        loss = sil_policy_loss + value_coef * sil_value_loss

        # 6. Fresh priorities (current-net A⁺) + the "is SIL firing?" diagnostic
        new_priorities = adv_clipped.detach()
        valid_fraction = (adv_clipped > 0).float().mean()

        return loss, new_priorities, sil_policy_loss, sil_value_loss, valid_fraction




class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Prediction network
        self.predictor = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, self.output_size)),
        )

        # Target network — frozen random projection.
        # Shape chain: input_size → 256 → 256 → output_size
        # (Was 9→256→[Linear(9,256)] which is impossible — typo: the second
        #  hidden layer must be 256→256, not input_size→256.)
        self.target = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, self.output_size)),
        )

        # target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature



class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems



# ── Main Function ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Constants
    PORT = 8000
    
    args = load_args("configs/ppo_rnd.yml")
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # ── Structured logger ────────────────────────────────────────────────────
    # Replaces ad-hoc print() calls. Format includes timestamp + level + msg,
    # so log lines stream into Modal's container logs with a uniform shape and
    # are greppable. The logger is configured once at module level and used
    # throughout the rollout / update loop.
    logging.basicConfig(
        level   = getattr(logging, args.log_level.upper(), logging.INFO),
        format  = "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt = "%H:%M:%S",
        stream  = sys.stdout,
    )
    log = logging.getLogger("ppo_rnd")
    log.info("Args: %s", {k: v for k, v in vars(args).items() if not k.startswith("_")})

    run_name = f"JumpKing__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
        
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # CUDA (Modal/cloud GPU) → MPS (Apple Silicon) → CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = args.torch_deterministic
    
    if args.torch_deterministic:
        torch.use_deterministic_algorithms(True)

    # -- Setup Server --
    # 1. Start server first — driver.get() needs it running
    launch_server()

    # 2. Open browser and navigate to the game
    driver = launch_browser(headless=True)   # no rendering during training — saves ~30% wall time
    driver.get(f"http://localhost:{PORT}")

    # 3. Wait for p5.js setup() to finish and player to be created
    wait_for_game(driver)

    # 4. Pause the draw loop — we will step physics manually via execute_script
    setup_game(driver)
    
    # -- Setup RL Environment --
    # IMPORTANT: pass args.gamma so the env's PBRS shaping uses the *same*
    # discount factor as PPO's advantage estimation. If they disagree, the
    # Ng-Harada-Russell invariance theorem stops applying.
    env = JumpKingEnv_Multi(driver = driver, num_agents = args.num_envs, frame_skip = 4, gamma = args.gamma)
    assert isinstance(env.single_action_space, gym.spaces.Discrete)

    # (gym.wrappers.RecordVideo deliberately NOT used here — it triggers on
    #  episode index and writes only to disk. We want trigger-on-update +
    #  direct W&B upload + no persistent local files. S ee video logic below.)
    if args.capture_video:
        os.makedirs(args.video_folder, exist_ok=True)

    agent = Agent(env.single_action_space.n).to(device)
    rnd_model = RNDModel(OBS_DIM, EMBED_DIM).to(device)
    
    combined_parameters = list(agent.parameters()) + list(rnd_model.predictor.parameters())
    optimizer = optim.Adam(
        combined_parameters,
        lr = float(args.learning_rate),
        eps = EPS,
    )

    os.makedirs("checkpoints", exist_ok=True)

    reward_rms          = RunningMeanStd()
    obs_rms             = RunningMeanStd(shape=(OBS_DIM, ))
    discounted_reward   = RewardForwardFilter(args.int_gamma)

    # Storage setup
    obs                 = torch.zeros((args.num_steps, args.num_envs, OBS_DIM)).to(device)
    actions             = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    logprobs            = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards             = torch.zeros((args.num_steps, args.num_envs)).to(device)
    curiosity_rewards   = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones               = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ext_values          = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_values          = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step         = 0
    start_time          = time.time()
    next_obs_np, _      = env.reset()
    
    next_obs            = torch.Tensor(next_obs_np).to(device)
    next_done           = torch.zeros(args.num_envs).to(device)
    num_updates         = args.total_timesteps // args.batch_size

    # ── Per-episode diagnostic trackers ──────────────────────────────────────
    # ep_max_level   : highest level the agent reached this episode (progress, not oscillation)
    # ep_jump_fires  : count of (jump-action → non-jump-action) transitions = real jump releases
    # prev_was_jump  : whether the *previous* action was a jump-charge (idx 3, 4, 5)
    # total_episodes : running count of completed episodes across the whole run
    JUMP_ACTIONS        = {3, 4, 5}
    
    ep_max_level        = np.zeros(args.num_envs, dtype = np.int32)
    ep_jump_fires       = np.zeros(args.num_envs, dtype = np.int32)
    prev_was_jump       = np.zeros(args.num_envs, dtype = bool)
    ep_return           = np.zeros(args.num_envs, dtype = np.float32)
    ep_length           = np.zeros(args.num_envs, dtype = np.int32)
    total_episodes      = 0

    # ── Video recording state ────────────────────────────────────────────────
    # Two independent streaming recorders, both fed from the same env.render()
    # frame on any step where at least one is active:
    #
    #   periodic_rec  : a long (1–5 min) training video, re-armed every
    #                   video_record_interval updates. Spans multiple episodes /
    #                   updates until it has video_length_seconds of footage.
    #   newlevel_rec  : a short achievement clip, armed the first time ANY agent
    #                   reaches a new global max level (best_level_so_far).
    #
    # Streaming to disk keeps memory flat no matter how long the clip is.
    periodic_rec = VideoStreamer(
        fps           = args.video_fps,
        target_frames = int(args.video_length_seconds * args.video_fps),
        wandb_key     = "video/training",
        track         = args.track,
    )
    newlevel_rec = VideoStreamer(
        fps           = args.video_fps,
        target_frames = int(args.newlevel_video_seconds * args.video_fps),
        wandb_key     = "video/new_level",
        track         = args.track,
    )
    
    best_level_so_far = 0   # highest global level any agent has reached this run

    log.info("Start to initialize observation normalization parameter.....")
    
    # Separate buffer for warm-up obs. Each entry is one (N, 9) chunk from env.step
    warmup_buffer: list[np.ndarray] = []
    
    # ── Initialization Phase  ────────────────────────────────────────────────
    for step in range(args.num_steps * args.num_iterations_obs_norm_init):        
        # Dim: (N, ) ints
        action_np = env.action_space.sample()
        
        # Automatic reset on vectorized environments
        # Dim: (N, 9)
        next_obs_np, _, _, _, _ = env.step(action_np)
        
        warmup_buffer.append(next_obs_np)
        
        # Every num_steps step-calls, flush the buffer into `obs_rms`.                
        if len(warmup_buffer) % args.num_steps == 0: 
            # Stack the `num_steps` chunks into one (num_steps * N, 9) array
            flat = np.concatenate(warmup_buffer, axis = 0)
            obs_rms.update(flat)
            warmup_buffer.clear()
               
    log.info("End to initialize...")
    
    # Reset env so the main rollout starts from a clean slate
    next_obs_np, _      = env.reset()
    next_obs            = torch.Tensor(next_obs_np).to(device)
    next_done           = torch.zeros(args.num_envs, device = device)
    
    # Added replay buffer for Self-Imitation Learning (SIL)
    memory = Memory(capacity = args.sil_buffer_capacity, obs_dim = OBS_DIM, device = device, epsilon = EPS)

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # ── Arm the periodic recorder ───────────────────────────────────────
        # Only (re)arm when it is idle
        # A long clip may still be filling across several updates, and we must 
        # not interrupt it mid-stream.
        if (
            args.capture_video
            and not periodic_rec.is_active
            and (update % args.video_record_interval == 0)
        ):
            path = os.path.join(
                args.video_folder, f"update_{update:06d}_step_{global_step}.mp4"
            )
            periodic_rec.start(path)
            log.info("[video] periodic recorder armed → %s (target %.0fs ≈ %d frames)",
                     path, args.video_length_seconds, periodic_rec.target_frames)

        # ── Rollout Phase (On-Policy) ───────────────────────────────────────
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                value_ext, value_int = agent.get_value(obs[step])
                ext_values[step], int_values[step] = (
                    value_ext.flatten(),
                    value_int.flatten(),
                )
                # Samples a fresh action from current policy
                action, logprob, _, _, _ = agent.get_action_and_value(obs[step])

            actions[step] = action
            logprobs[step] = logprob

            # ── Jump-fire detection (transition: jump-charge → release) ──────
            # A jump-charge action (3, 4, 5) sets jumpHeld=True in the browser.
            # The actual jump fires the moment we switch back to a non-jump
            # action (0, 1, 2) — that is when _apply_action() calls player.Jump().
            # Counting these transitions confirms whether the agent ever even
            # *attempts* the hold → release sequence the game requires.  
            action_np = action.cpu().numpy() 
            curr_is_jump = np.isin(action_np, list(JUMP_ACTIONS))
            
            # Dim: (N, ) bool, one entry per agent
            fired = prev_was_jump & ~curr_is_jump
            
            # Add 1 to each slot that fired
            ep_jump_fires += fired.astype(np.int32)

            prev_was_jump = curr_is_jump

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs_np, reward, terminated, truncated, info = env.step(action_np)
            
            done = terminated | truncated
            
            # array of agent indices that just finished
            done_idxs = np.where(done)[0] 
            
            # Per-agent return / length accumulation
            # Dim: (N, ) += (N, )
            ep_return += reward
            ep_length += 1 
            
            # Track the highest level reached this episode — distinguishes
            # genuine progress from up-down oscillation on a single platform.
            ep_max_level = np.maximum(ep_max_level, info["state"][:, COL_LEVEL].astype(int))
            
            # ── Video: new-level trigger + shared frame capture ──────────────
            # Wrapped so a render failure (e.g. a Chrome renderer hiccup) degrades
            # gracefully to "no more video" instead of killing the whole run —
            # the policy keeps training and checkpoints keep saving.
            if args.capture_video:
                try:
                    # Did any agent just reach a brand-new global max level? If
                    # so, arm the achievement recorder (unless already rolling).
                    cur_max_level = int(info["state"][:, COL_LEVEL].max())
                    if cur_max_level > best_level_so_far:
                        best_level_so_far = cur_max_level
                        if not newlevel_rec.is_active:
                            path = os.path.join(
                                args.video_folder,
                                f"newlevel_L{cur_max_level:02d}_step_{global_step}.mp4",
                            )
                            newlevel_rec.start(path)
                            log.info(
                                "[video] NEW GLOBAL LEVEL %d reached — recording "
                                "achievement clip → %s (%.0fs)",
                                cur_max_level, path, args.newlevel_video_seconds,
                            )

                    # One render() shared by every active recorder. env.render()
                    # returns an (H, W, 3) uint8 rgb_array of the canvas, now
                    # showing ALL agents (window.population) via rlRenderMode.
                    if periodic_rec.is_active or newlevel_rec.is_active:
                        frame = env.render()
                        periodic_rec.add(frame)   # no-op if that recorder is idle
                        newlevel_rec.add(frame)

                    # Flush whichever recorder has hit its frame budget.
                    if periodic_rec.is_full:
                        periodic_rec.finish(log, global_step)
                    if newlevel_rec.is_full:
                        newlevel_rec.finish(log, global_step)
                
                except Exception as e:
                    log.warning(
                        "[video] capture failed at step %d (%s) — disabling video "
                        "for the rest of the run; training continues.",
                        global_step, e,
                    )
                    args.capture_video = False
                    for _rec in (periodic_rec, newlevel_rec):
                        try:
                            _rec.finish(log, global_step)
                        except Exception:
                            pass

            for k in done_idxs: 
                total_episodes += 1
                log.info(
                    "[episode #%d agent=%d] step=%d return=%.2f length=%d max_level=%d jump_fires=%d success=%s",
                    total_episodes, k, global_step,
                    ep_return[k], ep_length[k], ep_max_level[k], ep_jump_fires[k],
                    bool(info["success"][k]),
                )
                
                writer.add_scalar("charts/episodic_return",     ep_return[k],     global_step)
                writer.add_scalar("charts/episodic_length",     ep_length[k],     global_step)
                writer.add_scalar("charts/episodic_max_level",  ep_max_level[k],  global_step)
                writer.add_scalar("charts/episodic_jump_fires", ep_jump_fires[k], global_step)
                writer.add_scalar("charts/episodic_success",    float(info["success"][k]), global_step)
                writer.add_scalar("charts/total_episodes",      total_episodes,   global_step)

                                
            # Per-agent tracker reset
            ep_max_level[done_idxs]  = 0
            ep_jump_fires[done_idxs] = 0
            ep_return[done_idxs]     = 0.0
            ep_length[done_idxs]     = 0
            prev_was_jump[done_idxs] = False

            rewards[step] = torch.tensor(reward).to(device).view(-1)

            next_obs = torch.Tensor(next_obs_np).to(device)  
            next_done = torch.Tensor(done.astype(np.float32)).to(device)            
            
            rnd_next_obs = (
                (
                    (next_obs - torch.from_numpy(obs_rms.mean).float().to(device))
                    / torch.sqrt(torch.from_numpy(obs_rms.var).float().to(device))
                ).clip(-5, 5)
            ).float()
            
            target_next_feature = rnd_model.target(rnd_next_obs)
            predict_next_feature = rnd_model.predictor(rnd_next_obs)
            curiosity_rewards[step] = ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).data
                     


        # ── Compute GAE (Advantage Computation) & Monte-Carlo Return (R) ───────────────────────────────────────
        monte_carlo_r = torch.zeros((args.num_steps, args.num_envs)).to(device)
        curiosity_reward_per_env = []
        
        for reward_per_step in curiosity_rewards.cpu().data.numpy().T:
            # Reset state per agent's trajectory
            discounted_reward.rewems = None
            curiosity_reward_per_env.append(discounted_reward.update(reward_per_step))
        
        curiosity_reward_per_env = np.array(curiosity_reward_per_env)
        
        mean, std, count = (
            np.mean(curiosity_reward_per_env),
            np.std(curiosity_reward_per_env),
            len(curiosity_reward_per_env),
        )
        
        reward_rms.update_from_moments(mean, std**2, count)

        # Capture RAW intrinsic mean before in-place normalization, so we can
        # compare it apples-to-apples against the (un-normalized) extrinsic stream.
        raw_curiosity_mean = curiosity_rewards.mean().item()

        curiosity_rewards /= np.sqrt(reward_rms.var)
        
        # Bootstrap value if not done
        with torch.no_grad(): 
            next_value_ext, next_value_int = agent.get_value(next_obs)
            next_value_ext = next_value_ext.flatten()
            next_value_int = next_value_int.flatten()
            
            ext_advantages = torch.zeros_like(rewards, device = device)
            int_advantages = torch.zeros_like(curiosity_rewards, device = device)
            
            ext_last_r = next_value_ext
            
            ext_lastgaelam = 0
            int_lastgaelam = 0
            
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ext_nextnonterminal = 1.0 - next_done
                    int_nextnonterminal = 1.0
                    
                    # Dim: (N, 1)
                    ext_nextvalues = next_value_ext
                    int_nextvalues = next_value_int
                else:
                    # Dim: (N, )
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    int_nextnonterminal = 1.0
                    ext_nextvalues = ext_values[t + 1]
                    int_nextvalues = int_values[t + 1]
                    
                ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                
                int_delta = curiosity_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                
                ext_advantages[t] = ext_lastgaelam = (
                    ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                )
                
                int_advantages[t] = int_lastgaelam = (
                    int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
                )
                
                # Compute Monte-Carlo `R` at timestep `t`
                # Mask is applied, so episodes reset at boundaries. When `dones[t + 1] = 1`, the mask zeroes the future and `R[t] = rewards[t]`
                monte_carlo_r[t] = ext_last_r = (rewards[t] + args.gamma * ext_nextnonterminal * ext_last_r)
                
            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values


        # Flatten the batch
        b_obs = obs.reshape((-1, OBS_DIM))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)
        
        b_sil_returns = monte_carlo_r.reshape(-1)
        b_priority = (b_sil_returns - b_ext_values).clamp(min = 0) + EPS
        b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef

        obs_rms.update(b_obs.reshape(-1, OBS_DIM).cpu().numpy())

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)

        # b_obs shape is (batch_size, OBS_DIM) = (512, 9) after flattening
        rnd_next_obs = (
                (
                    (b_obs - torch.from_numpy(obs_rms.mean).float().to(device))
                    / torch.sqrt(torch.from_numpy(obs_rms.var).float().to(device))
                ).clip(-5, 5)
            ).float()

        clipfracs = []
        
        # Update replay buffer
        # Currently, the textbook default (guarantee every new transition is sampled at least once).
        # Max-priority insert would make those "invalid" transitions get sampled heavily at first, wasting minibatches until their priority corrects. Inserting with the real (R−V)₊ immediately means the invalid ones start at ε (near-zero) and are rarely drawn from the very beginning
        # TODO: Validate with tests cases
        """
        Option A: pass the priority in (recommended for SIL): memory.add(b_obs, b_actions, b_sil_returns, b_priority) → the buffer stores all four columns as given.
        Option B: standard PER max-priority insert: memory.add(b_obs, b_actions, b_sil_returns) → the buffer assigns each new entry self.max_priority, and the real (R−V)₊ gets written later on the first update_priorities (after that entry is first sampled and re-evaluated).  
        """
        
        # Dim: (batch_size, )
        memory.add(
            obs = b_obs, 
            actions = b_actions, 
            returns = b_sil_returns, 
            priorities = b_priority
        )

        
        # PPO Loop (On-Policy Updates)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                predict_next_state_feature, target_next_state_feature = rnd_model(rnd_next_obs[mb_inds])
                forward_loss = F.mse_loss(
                    predict_next_state_feature, target_next_state_feature.detach(), reduction="none"
                ).mean(-1)

                mask = torch.rand(len(forward_loss), device = device)
                mask = (mask < args.update_proportion).type(torch.FloatTensor).to(device)
                
                forward_loss = (forward_loss * mask).sum() / torch.max(
                    mask.sum(), torch.tensor([1], device = device, dtype=torch.float32)
                )
                
                # Re-evaluates the same stored action under the UPDATED policy
                # newl  ogprob vs stored logprob → the PPO ratio
                _, newlogprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                if args.clip_vloss:
                    ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                    ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                        new_ext_values - b_ext_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    
                    ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                    ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                    ext_v_loss = 0.5 * ext_v_loss_max.mean()
                else:
                    ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()

                int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                v_loss = ext_v_loss + int_v_loss
                entropy_loss = entropy.mean()
                
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss
                optimizer.zero_grad()
                loss.backward()
                
                if args.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        combined_parameters,
                        args.max_grad_norm,
                    )
                    
                optimizer.step()

            if args.target_kl is not 0.0:
                if approx_kl > args.target_kl:
                    break
            
            
            
        #  SIL LOOP (Off-Policy Updates)
        sil_valids, sil_pgs, sil_vs = [], [], []
        
        # Samples completely random historic minibatches from the replay buffer.
        if args.use_sil and len(memory) >= args.sil_min_samples:
            for _ in range(args.sil_update_epochs):
                sil_obs, sil_actions, sil_returns, idxs, sil_weights = memory.replay_sample(args.sil_batch_size)
                sil_obs     = sil_obs.to(device)
                sil_actions = sil_actions.to(device)
                sil_returns = sil_returns.to(device)
                sil_weights = sil_weights.to(device)

                loss, new_p, sil_pg, sil_v, valid = agent.calc_sil_loss(
                    sil_obs, sil_actions, sil_returns, sil_weights, args.sil_value_coef
                )
                
                optimizer.zero_grad()
                loss.backward()
                
                nn.utils.clip_grad_norm_(
                    combined_parameters, 
                    args.max_grad_norm)
                
                optimizer.step()

                # ← idxs from replay_sample threads back here
                memory.update_priorities(idxs, new_p)    
                
                # Log SIL data
                sil_valids.append(valid.item())
                sil_pgs.append(sil_pg.item())
                sil_vs.append(sil_v.item())



        # ── Plot Data ──────────────────────────────────
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/fwd_loss", forward_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        
        # ── SIL diagnostic ────────────────────────────────
        if sil_valids: 
            writer.add_scalar("sil/valid_fraction", np.mean(sil_valids), global_step)
            writer.add_scalar("sil/policy_loss",    np.mean(sil_pgs),    global_step)
            writer.add_scalar("sil/value_loss",     np.mean(sil_vs),     global_step)
            writer.add_scalar("sil/buffer_size",    len(memory),        global_step)


        # ── Structured per-update debug log ──────────────────────────────────
        # Single one-liner with everything you'd want to eyeball per update.
        log.info(
            "[update %d/%d] step=%d ep=%d sps=%d  "
            "lr=%.2e pg=%.4f v=%.4f ent=%.4f kl=%.4f fwd=%.4f",
            update, num_updates, global_step, total_episodes, sps,
            optimizer.param_groups[0]["lr"],
            pg_loss.item(), v_loss.item(), entropy_loss.item(),
            approx_kl.item(), forward_loss.item(),
        )

        # ── Reward-scale asymmetry diagnostic ────────────────────────────────
        # rewards            : raw extrinsic (height_gain - 0.01, never normalized)
        # raw_curiosity_mean : intrinsic BEFORE divide-by-std (true RND novelty scale)
        # curiosity_rewards  : intrinsic AFTER divide-by-std (what actually feeds GAE)
        # When ext is ±hundreds and int is ~unit, `ext_coef` vs `int_coef` are
        # not really comparable — these scalars confirm the gap.
        writer.add_scalar("charts/extrinsic_reward_mean",    rewards.mean().item(),             global_step)
        writer.add_scalar("charts/intrinsic_reward_mean_raw", raw_curiosity_mean,               global_step)
        writer.add_scalar("charts/intrinsic_reward_mean_norm", curiosity_rewards.mean().item(), global_step)

        # ── Per-action frequency over this rollout ───────────────────────────
        # Directly diagnoses the "agent only walks right" failure mode: if
        # action 2 (walk right) dominates, you can see it here in one glance.
        flat_actions  = actions.cpu().numpy().astype(int).flatten()
        action_counts = np.bincount(flat_actions, minlength = env.single_action_space.n)
        action_freqs  = action_counts / max(len(flat_actions), 1)
        action_names  = ["idle", "left", "right", "jump_neutral", "jump_left", "jump_right"]
        for i, (name, freq) in enumerate(zip(action_names, action_freqs)):
            writer.add_scalar(f"actions/freq_{i}_{name}", freq, global_step)
        # Full histogram view (W&B will render it as a bar chart over time)
        writer.add_histogram("actions/distribution", flat_actions, global_step, bins = env.single_action_space.n)

        if update % args.save_interval == 0:
            torch.save(agent.state_dict(), f"checkpoints/agent_{global_step}.pt")
            torch.save(agent.state_dict(), "checkpoints/agent_latest.pt")
            log.info("[checkpoint] saved at step %d", global_step)

        # (Video flushing now happens inline in the rollout loop the instant a
        #  recorder hits its frame budget — see the "Video: new-level trigger +
        #  shared frame capture" block above. Recorders that are still mid-clip
        #  here simply keep streaming into the next update.)

    # Flush any clips still in progress when training ends, so partial periodic /
    # achievement videos are not lost.
    if args.capture_video:
        periodic_rec.finish(log, global_step)
        newlevel_rec.finish(log, global_step)

    torch.save(agent.state_dict(), "checkpoints/agent_final.pt")
    log.info("[checkpoint] final model saved; %d episodes completed across %d steps",
             total_episodes, global_step)
    env.close()
    writer.close()