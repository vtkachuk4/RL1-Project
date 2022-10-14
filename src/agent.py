import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)  # part of why we derive straight from the nn class

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(
            x
        )  # don't want to activate it because we want raw estimate. Value estimates should indeed be negative
        return actions


class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        batch_size,
        n_actions,
        max_mem_size=100000,
        eps_end=0.01,
        eps_dec=5e-5,
        seed=0,
    ):
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
        T.manual_seed(seed)
        self.Q_eval = DQNetwork(
            self.lr,
            n_actions=n_actions,
            input_dims=input_dims,
            fc1_dims=256,
            fc2_dims=256,
        )
        # Replay memory arrays; TODO - can switch to trad. replay buffer as in pytorch tutorial
        self.state_memory = np.zeros(
            (self.mem_size, *input_dims), dtype=np.float32
        )  # PyTorch particular about datatypes
        self.new_state_memory = np.zeros(
            (self.mem_size, *input_dims), dtype=np.float32
        )  # for states that resulted from actions
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = (
            self.mem_cntr % self.mem_size
        )  # ensures position will wrap around after we hit mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self, episode=None):
        if self.mem_cntr < self.batch_size:
            return
        self.Q_eval.optimizer.zero_grad()
        max_mem = min(
            self.mem_cntr, self.mem_size
        )  # calculate position of max memory because we want to select a subset of our memory up to the last filled memory
        batch = np.random.choice(
            max_mem, self.batch_size, replace=False
        )  # want to ensure that once you select, you take it out
        batch_index = np.arange(
            self.batch_size, dtype=np.int32
        )  # need this for array slicing
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        q_eval = self.Q_eval.forward(state_batch)[
            batch_index, action_batch
        ]  # want values of the actions we actually took, batch_index is to pull out the batch from tensor
        q_next = self.Q_eval.forward(
            new_state_batch
        )  # if using a target network, would be used here
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end
        )


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        batch_size=64,
        n_actions=4,
        eps_end=0.01,
        input_dims=[8],
        lr=0.0003,
        seed=1,
    )
    scores, eps_history = [], []
    n_games = 500
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        observation = np.array(observation[0])
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        print(
            "episode ",
            i,
            "score %.2f" % score,
            "average score %.2f" % avg_score,
            "epsilon %.2f" % agent.epsilon,
        )
