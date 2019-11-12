import numpy as np
import random
import copy 

class OUNoise:
    
    def __init__(self,size,seed,mu=0.,theta=0.15,sigma=0.2):
        
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset_internal_state()
        
    def reset_internal_state(self):
        self.state = copy.copy(self.mu)
        
    def sample(self):
        """Update and return the noise sample"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
    
        
        
        