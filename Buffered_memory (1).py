from collections import namedtuple , deque
import random
import torch 
import numpy as np 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ReplayBuffer:
    def __init__(self,action_size,buffer_size,batch_size,seed,device):
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",field_names=["state","action","reward","next_state","done"])
        self.seed = random.seed(seed)
        self.device = device
        
        
    def add_experience(self,state,action,reward,next_state,done):
        #add the new experience to the memory
        exp = self.experience(state,action,reward,next_state,done)
        self.memory.append(exp)
        
    def sample(self):
        # sampling the batch of experiences from buffered memory
        xps = random.sample(self.memory,k = self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in xps if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in xps if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in xps if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in xps if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in xps if e is not None]).astype(np.uint8)).float().to(device)
        
        return states,actions,rewards,next_states,dones

    def __len__(self):
        return len(self.memory) # sizeof internal memory ( capacity)
        
        