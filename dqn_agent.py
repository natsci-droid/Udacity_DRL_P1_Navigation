import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 2.5e-4             # learning rate 
UPDATE_EVERY = 8        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, PER=False, sampling=0.1):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        if PER:
            # Add an initial priority
            self.memory.add_p(1)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                if PER:
                    experiences = self.memory.sample_PER()
                    self.learn(experiences, GAMMA, sampling, PER=True)
                else:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA, sampling)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, sampling, PER=False):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        if PER:
            states, actions, rewards, next_states, dones, probs = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        # dones = 1 if done, so Q_targets collapses to rewards
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
               
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if PER:
            delta = np.abs(Q_expected.detach().numpy() - Q_targets.detach().numpy())

            # clip delta to max of 1
            delta[delta>1] = 1

            # Update experiences with probabilities based on TD Error
            for i in range(len(states)):
                self.memory.add(states[i].numpy(), actions[i].numpy(), rewards[i].numpy(),
                                next_states[i].numpy(), dones[i].numpy())

                self.memory.add_p(delta[i][0])      

            # ------------------- update target network ------------------- #
            w_sample = (1/(len(self.memory) + BATCH_SIZE) * 1/probs )**sampling

        else:
            w_sample = np.ones(len(states))

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU, torch.from_numpy(w_sample))         

    def soft_update(self, local_model, target_model, tau, w_sample):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """

        for target_param, local_param, w_s in zip(
                          target_model.parameters(), local_model.parameters(), w_sample):
            target_param.data.copy_(tau*w_s*local_param.data + (1.0-tau*w_s)*target_param.data)



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

        self.priority = deque(maxlen=buffer_size) # create deque for priority
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def add_p(self, delta, e_const=0.1, alpha = 0.2):
        """Add a new priority."""

        self.priority.append((delta + e_const) **alpha)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def sample_PER(self):
        """Sample a batch of experiences from memory, using prioritised experience."""

        probs = self.priority / np.sum(self.priority)

        inds = np.random.choice(range(len(self.memory)), self.batch_size, replace=False, p=probs)

        experiences = []
        for ind in inds:
            experiences.append(self.memory[ind])

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
 
        # remove from memory (they will be replaced with new priority)
        inds = list(set(inds)) # don't want to delete same index multiple times
        for ind in sorted(inds, reverse = True):  
            del self.memory[ind]
            del self.priority[ind]

        return (states, actions, rewards, next_states, dones, probs[inds])

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
