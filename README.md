# How does it work (Trained using Reinforcement Learning and Imitation Learning)
Sample Efficient Actor-Critic With Experience Replay [https://arxiv.org/pdf/1611.01224] - On/Off Policy Learning
Alpha Go - Supervised Learning of Policy Networks [https://arxiv.org/pdf/1704.03732]
DeepMind - Deep Q-learning from Demonstrations [https://cdn.aaai.org/ojs/11757/11757-13-15285-1-2-20201228.pdf]

**Encode each video/rounds as experiences **
Game UI -> Adapter 
Game Engine (JS) -> Adapter
Adapter -> Game State (Selenium) -> RL Environment (PY) && RL Agent (PY)