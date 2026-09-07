import numpy as np
import torch

# Self-Imitation Learning (SIL) Class
# A single persistent replay buffer of flat (obs, action, MC-return) transitions with priorities; each rollout the trainer computes returns from the existing on-policy storage (resetting on done), adds them, then samples prioritized minibatches for the SIL update after PPO. The buffer then accumulates and never clears per-rollout, further the loss lives outside the buffer.
class Memory:     
    def __init__(self, capacity, obs_dim, device, alpha = 0.6, beta = 0.5, epsilon = 1e-5):         
        self.device         = device
        self.capacity       = capacity
        
        # Controls prioritization strictness (0 = uniform, 1 = strict)
        self.alpha          = alpha     
        
        # Controls importance-sampling correction
        self.beta           = beta      
        
        self.epsilon        = epsilon
        self.pos            = 0         # Current index pointer
        self.size           = 0         # Current buffer fill size
        
        # Tracks max priority for incoming new samples
        self.max_priority   = 1.0        
        
        self.actions        = np.zeros(capacity, np.int64)
        self.obs            = np.zeros((capacity, obs_dim), 
                                       dtype = np.float32)
        self.returns        = np.zeros(capacity, np.float32)
        self.priorities     = np.zeros(capacity, np.float32)
        
    def __len__(self): 
        return self.size
    
    def add(self, obs, actions, returns, priorities = None): 
        # 1024 indices, wraps at the end
        n = obs.shape[0]
        idxs = (self.pos + np.arange(n)) % self.capacity     
        
        self.obs[idxs]           = obs.detach().cpu().numpy().astype(np.float32)
        self.actions[idxs]       = actions.cpu().numpy().astype(np.int64)
        self.returns[idxs]       = returns.cpu().numpy().astype(np.float32)
        
        if priorities is None: 
            self.priorities[idxs] = self.max_priority
        else: 
            self.priorities[idxs]    = priorities.cpu().numpy().astype(np.float32)
            self.max_priority = max(self.max_priority, float(priorities.max()))
        
        # Advance ring buffer pointers
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)
    
    
    def update_priorities(self, indices, new_priorities):
        """
        Updates the priorities of sampled transitions.
        new_priorities should be the clipped advantages: max(returns - value, 0)
        """
        
        # Convert torch tensor if passed
        if isinstance(new_priorities, torch.Tensor):
            new_priorities = new_priorities.detach().cpu().numpy()
            
        indices = np.array(indices, dtype=np.int64)
        
        # Clip at a tiny epsilon value to guarantee they can always be re-sampled
        clipped_priorities = np.maximum(new_priorities, self.epsilon)
        
        self.priorities[indices] = clipped_priorities
        self.max_priority = max(self.max_priority, float(clipped_priorities.max()))
        
    
    def replay_sample(self, batch_size): 
        assert self.size > 0
        
        # Prioritized Experience Replay (PER) 
        # Add epsilon to prevent 0 probability samples from breaking the power calculation
        scaled = (self.priorities[:self.size] + self.epsilon) ** self.alpha
        probs = scaled / scaled.sum()
        
        idxs = np.random.choice(self.size, batch_size, p = probs, replace = True)
        
        # Max-normalized Importance Sampling Weights
        min_prob = probs[:self.size].min()
        max_weight = (self.size * min_prob) ** (-self.beta)
        
        # Normalized by global max weight in the buffer
        weights = (self.size * probs[idxs]) ** (-self.beta)
        weights = weights / max_weight 
        
        b_obs     = torch.as_tensor(self.obs[idxs],     dtype=torch.float32).to(self.device)
        b_actions = torch.as_tensor(self.actions[idxs], dtype=torch.long)
        b_returns = torch.as_tensor(self.returns[idxs], dtype=torch.float32)
        b_weights = torch.as_tensor(weights,            dtype=torch.float32)
        
        return b_obs, b_actions, b_returns, idxs, b_weights