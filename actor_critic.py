import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layers):
    fan_in = layers.weight.data.size()[0]
    lt = 1. / np.sqrt(fan_in)
    return(-lt,lt)

class Actor(nn.Module):
    
    def __init__(self,state_size,action_size,seed,fc1_units=256,fc2_units=128):
        
        super(Actor,self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size,fc1_units)
        self.batch_normalization = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.fc3 = nn.Linear(fc2_units,action_size)
        self.reset_parameters()
        
    def reset_parameters(self):
        #uniform_ Fills the given 2-dimensional matrix with values drawn from a uniform distribution parameterized by low and high.
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self,state):
        # Actor network that maps s --> a ( states with actions)
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = F.relu(self.batch_normalization(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    
class Critic(nn.Module):
    def __init__(self,state_size,action_size,seed,fc1_units=256,fc2_units=128):
        
        super(Critic,self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size,fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size,fc2_units)
        self.fc3 = nn.Linear(fc2_units,1)
        self.reset_parameters()
        
    def reset_parameters(self):
        #uniform_ Fills the given 2-dimensional matrix with values drawn from a uniform distribution parameterized by low and high.
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self,state,action):
        # critic network that maps s,a --> Q values
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
            
        xstate = F.relu(self.fc1(state))
        
        x = torch.cat((xstate,action),dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
        
        
        
        
        
