# P3-Continous-Control-RLND
Solving a more difficult continuous control environment, where the goal is to teach a creature with four legs to walk forward without falling.

<img src = "/robotic_arm.gif" width="75%" align="center" alt="robotic_arms" title="Robotic Arms"/>

# Description :

This project deals with the reinforcement learning Agent that closely entails with Robotic arms in Unity Reacher's Environment.

Exploring Unity Agents
        
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 33
        Number of stacked Vector Observation: 1
        Vector Action space type: continuous
        Vector Action space size (per agent): 4
        
### Goal:

The goal of the agent is to maintain its position at target location for many steps to maximum.The action vector entries should vary between -1 and +1

This is an *Episodic* task which solves the environment attaining the maximum average score of 30.0 iterating over consecutive episodes

### Project Info - Getting started :

If you would like to run this project in your local machine do the step as mentioned below :

*Step 1*: Clone the repo (coz Python modules for running the unity agents is not present in my repo)

git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .

*Step 2*: Download the Unity Agents that suits your OS Platform
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

After downloading place the environment file path in Continous_Control.ipynb notebook file

### Project Architecture :

Files        | Description
------------ | -------------
ContinuousControlProject.ipynb | Jupyter Notebook (for training)
ddpg_Agent.py |  DDPG Agent class
actor_critic.py |  Actor and Critic network archi
Buffered_memory.py |  replay buffer class
OrnsteinUhlenbeck_noise.py |  OUInoise class
checkpoint_critic.pth | critic trained model
checkpoint_actor.pth | actor trained model
Report.md |  Description abt project implementation 

All the files gets imported in the notebook files in which ddpg_Agent acts as backend holding Buffered_memory and OUInoise class for the project 

*Step 3*: Running all the cells in the notebook will start training the model 
