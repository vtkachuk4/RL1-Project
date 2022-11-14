import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
from FTA import FTA

class DQNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, activation: FTA):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        if type(activation).__name__ == "FTA":
            self.fc3 = nn.Linear(
                self.fc2_dims * activation.expansion_factor, self.n_actions
            )  # need to increase layer size by number of bins
        else:
            self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.activation = activation

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.activation(self.fc2(x))  # FTA(x*W2 + b2)
        actions = self.fc3(
            x
        )  # don't want to activate it because we want raw estimate. Value estimates should indeed be negative
        return actions


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        batch_size,
        n_actions,
        activation,
        device,
        max_mem_size=100000,
        eps_end=0.01,
        eps_dec=5e-5,
        seed=42,
    ):
        self.steps_done = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.mem_cntr = 0
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.activation = activation

        T.manual_seed(seed)

        # self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.device = T.device(device)
        self.QNetwork = DQNetwork(input_dims, 256, 256, n_actions, activation).to(self.device)
        self.targetNetwork = DQNetwork(input_dims, 256, 256, n_actions, activation).to(self.device)
        self.targetNetwork.load_state_dict(self.QNetwork.state_dict())
        self.targetNetwork.eval()

        self.optimizer = optim.Adam(self.QNetwork.parameters(), lr=self.lr)
        self.memory = ReplayMemory(self.mem_size)


    def choose_action(self, state, greedy=False):
        epsilon = 0 if greedy else self.epsilon
        if np.random.random() >= epsilon:
            with T.no_grad():
                return self.QNetwork(state).max(1)[1].view(1, 1)
        else:
            return T.tensor([[random.randrange(4)]], device=self.device, dtype=T.int64)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = T.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=T.bool)
        non_final_next_states = T.cat([s for s in batch.next_state if s is not None])

        state_batch = T.cat(batch.state)
        action_batch = T.cat(batch.action)
        reward_batch = T.cat(batch.reward)


        state_action_values = self.QNetwork(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = T.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.targetNetwork(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch


        loss_fn = nn.MSELoss()
        loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))


        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.QNetwork.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end
        )
