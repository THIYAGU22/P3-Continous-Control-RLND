# PROJECT SPECIFICATIONS:

### Learning Algorithm:

DDPG ( Deep Deterministic Policy Gradients ) Algorithm has been implemented with backend as replay-Buffer behind this Project.

The Q-Network is much like Actor-Critic Methods whereas in DDPG  the Actor directly maps states to actions (the output of the network directly the output) instead of outputting the probability distribution across a discrete action space.

### Algo:

<img src = "/algo.png" width="75%" align="center" alt="robotic_arms" title="Robotic Arms"/>

### Key Points:
* Experience replay
* Actor & Critic network updates
* Target network updates
* Exploration

The building block of DDPG algorithm is Agent Class. It holds act ,learn , step and soft_update methods . Act method generates a entries for the corrosponding actions with help of Actor Model.
Noise to be added to the actions to allow the explore the possiblities in the action space. The noise is always generated with the help of Ornstein-Uhlenbeck Process.
Learn method does update the models with respectives to the learning samples in the memory .


### Network Architecture:

* Actor 
  * Hidden_Layer1 = input,400 -> ReLu activation 
  * Hidden_Layer2 = 400,300 -> ReLu activation 
  * Output layer  = 300,4(action_size) --> TanH activation 
  
* Critic
  * Hidden_Layer1 = input,400 -> ReLu activation 
  * Hidden_Layer2 = 400+action_size,300 -> ReLu activation 
  * Output layer  = 300,1 --> Linear


Hyperparameters      | Fine Value
-------------------  | -------------
Replay Batch Size | 128
Replay Buffer Size | 1e5
Actor LR | 3e-3
Critic LR| 3e-3
TAU | 1e-3
GAMMA | 0.99
EPS_DECAY | 1e-6
EPSILON | 1.0
OU_theta | 0.15
OU_sigma | 0.2
OU_MU | 0.0
Max_episodes | 500


### Plot Of Rewards:

After training the model attaining the score of 30.0 , by proving it as episodic task plotting the rewards graph for Episode Vs Score 

<img src = "/plot_of_rewards.png" width="75%" align="center" alt="robotic_arms" title="REWARD_PLOT"/>

Once it gets finished it will throw out 
```Environment solved in 9 episodes!	Moving Average =30.3 over last 100 episodes```

 <img src = "/finish.png" width="75%" align="center" alt="robotic_arms" title="SOLVED"/>

### Ideas for future work :

* prioritized Buffered Memory (replay buffer) -- Selecting the action based on priority respective to the actions (that has been experienced while exploring the action space)
* Controlling the Exploration vs Exploitation by tuning sigma , theta parameters for better exploration in the state space
* Tuning the hyperparmaters to attain maximum efficiency 
