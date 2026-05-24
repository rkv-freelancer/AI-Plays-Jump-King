## Project

## How does it work 
Trained using Reinforcement Learning and Imitation Learning:
- Sample Efficient Actor-Critic With Experience Replay [https://arxiv.org/pdf/1611.01224] - On/Off Policy Learning
- Alpha Go - Supervised Learning of Policy Networks [https://arxiv.org/pdf/1704.03732]

**Encode each video/rounds as experiences **
Game UI -> Adapter 
Game Engine (JS) -> Adapter
Adapter -> Game State (Selenium) -> RL Environment (PY) && RL Agent (PY)

## Agenda
- Start with a modest amount of human gameplay (10-20 successful runs)
- Pre-train policy network via behavioral cloning
- Fine-tune with RL 

## State Representation
Variables: 
- Position X: float  # Current X position
- Position Y: float  # Current Y position
- Velocity X: float  # Current X velocity
- Velocity Y: float  # Current Y velocity
- Is_Grounded: bool  # Whether player is on solid ground (can jump or walk)
- Is_Charging: bool  # Whether player is currently charging a jump
- Charge_Time: float # How long the jump has been charging (0 if not charging)
- Wall_Left: float   # Distance to nearest wall/obstacle on the left (or binary collision indicator)
- Wall_Right: float  # Distance to nearest wall/obstacle on the right
- Wall_Up: float     # Distance to ceiling/platform above
- Wall_Down: float   # Distance to floor below (0 if grounded)
- Map_Position: int  # Optional: zone/section identifier to help with progress tracking

## Action Space (The set of all possible moves)
Action_Type: int   # 0: No action, 1: Walk left, 2: Walk right, 3: Start charging jump, 4: Release jump
Direction_Key: int # 0: None, 1: Left, 2: Right (applied during charging or walking)

## For Imitation Learning from Demonstrations (Create a Script for this case)
Record these from human gameplay:
- Key_Pressed: list[bool]  # [Left, Right, Jump]
- Key_Duration: float      # How long jump was held (for jumps only)
- Direction_During_Charge: int  # 0: None, 1: Left, 2: Right 

# Network Structure Feed-Forward Neural Network: 
Input Layer (state) → Hidden Layers → Output Layer (action probabilities)

# Dueling Deep Q-Learning (DQN) Architecture: 
Input Layer → Shared Feature Layers → [Value Stream + Advantage Stream] → Q-values

# Agenda
- Connect with Modal (Cloud GPU Computing)
- Train PPO on Jump King