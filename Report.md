[//]: # (Image References)

[image1]: dqn.png "DQN Rewards"
[image2]: dqn_PER.png "DQN Rewards"


### Introduction

This report describes the implementation of the Deep Q-Network to solve the banana environment. The first method implemented uses the double DQN implementation. The original development of the DQN algorithm is decribed [here](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

### Deep Q-Network

The agent is trained with an epsilon greedy policy, to ensure exploration. Epsilon is initialised at 1.0 and decayed by a rate of 0.995 until it is stopped at 0.1. The number of training episodes is set to 2000, but does not require this many to solve the environment. Training is typically complete around 400 episodes, but this can vary with each run.

The hyperparameters for the agent are set in dqn_agent.py as:
* BUFFER_SIZE = int(1e5)  (replay buffer size)
* BATCH_SIZE = 64         (minibatch size)
* GAMMA = 0.99            (discount factor)
* TAU = 1e-3              (for soft update of target parameters)
* LR = 5e-4               (learning rate)
* UPDATE_EVERY = 4        (how often to update the network)

An example plot for rewards per episode is shown below. In this run the agent solved the environment in 405 episodes.

![DQN Rewards][image1]

The neural network used is defined in model.py and consists of three fully connected (dense) layers with 64 units. The activation function used for the first and second layers is the Rectified Linear Unit (RELU).

### Prioritised Experience Replay

Prioritised Experience Replay is a modification to the DQN network to improve the training process. Instead of randomly sampling from the replay buffer, experiences are selected that provide greater training value. These are identified from the TD error (delta), which is calculated during learning. A small offset, e_const=0.1 is added to delta to form the priority, ensuring that experience tuples are not completely starved for selection.

The selection probabilites are determined by raising the priorities to the power alpha and dividing by the sum. This permits some random sampling. When alpha is set to 1, the priorities are used and when it is set to 0, experiences are selected randomly. In this example, alpha is set to 0.2. The weight update hyperparameter (sampling) is set to increase from 0.1 to one at a rate of 1.005. 

An example plot for rewards per episode is shown below. In this run the agent solved the environment in 714 episodes.

![DQN_PER Rewards][image2]

Despite being an improvement, Prioritised Experience Replay slows the training considerably due to the extra calculations required. It also took more episodes to reach the solution, so is not an advantage for this environment.

For the original hyperparameters, training converged before solving the environment. Thus the hyperparametrs were changed to:
* LR = 2.5e-4             (learning rate) 
* UPDATE_EVERY = 8        (how often to update the network)

With these altered hyperparameters and no Prioritised Experience Replay, the solution is solved in 711 epochs. Thus for this environment tuning the hyperparameters has more effect on performance that the method.

### Potential Future Work
The implementation could be expanded with other improvements to the Deep Q-Network, such as the duelling DQN.

Additionally, more alterations of the hyperparameters could possibly yeild an agent that trains more quickly. 

Prioritised Experience Replay may permit a smaller replay buffer size, which could be an advantage in certain applications, so this could be examined.
