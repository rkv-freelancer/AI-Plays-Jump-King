import torch 
from torch import nn 
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256): 
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Testing 
if __name__ == "__main__":
    state_dim = 4
    action_dim = 2
    model = DQN(state_dim, action_dim)
    
    # Create a random input tensor with the same shape as the state
    x = torch.randn(1, state_dim)
    
    # Forward pass
    output = model(x)
    
    # Hence the shape is a batch of 1 with action_dim outputs
    print("Output shape:", output.shape)  # Should be (1, action_dim)
    print("Output:", output)  # Should be a tensor with action_dim values