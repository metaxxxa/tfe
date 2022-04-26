""" 
Experiment with DQN and RNN network (GRU Cells)
"""

from collections import deque
import random

import gym
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

#### Networks ####

class GRUNet(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_actions):
        super().__init__()

        self.hidden_size = hidden_dim
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_size)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
    
#### DQN ####
epsilon = 1.0
eps_decay = 1-1e-5

def epsilon_greedy(agent, state, hidden, eps):
    with torch.no_grad():
        qs, h = agent(torch.from_numpy(state).float()[None, :], hidden)
        if random.random() < eps:
            return random.randint(0, len(qs)-1), h
        else:
            return torch.argmax(qs).item(), h

def generate_episode(env, agent, eval=False):
    global epsilon
    episode = []
    hidden = agent.init_hidden()
    state, done = env.reset(), False
    while not done:
        action, next_hidden = epsilon_greedy(agent, state, hidden, eps=0.0 if eval else 0.0)
        if not eval: epsilon *= eps_decay
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward, done, next_state, hidden, next_hidden))
        state = next_state
        hidden = next_hidden
    return episode


def train(n_steps):
    env = gym.make('CartPole-v1')
    agent = GRUNet(input_shape=env.observation_space.shape[0],
                   hidden_dim=HIDDEN_SIZE,
                   n_actions=env.action_space.n)
    optimizer = optim.Adam(params=agent.parameters(), lr=LEARNING_RATE)
    buffer = deque(maxlen=BUFFER_SIZE)
    for idx in range(n_steps):
        episode = generate_episode(env, agent)
        for transition in episode:
            buffer.append(transition)
        if len(buffer) < BATCH_SIZE:
            continue
        batch = random.sample(buffer, k=BATCH_SIZE)
        states, actions, rewards, dones, next_states, hiddens, next_hiddens = zip(*batch)
        qs, _ = agent(torch.tensor(states), torch.stack(hiddens))
        qs_actions = qs[range(len(batch)), actions]
        next_qs, _ = agent(torch.tensor(next_states), torch.stack(next_hiddens))
        next_qs_max = torch.max(next_qs, dim=1)[0].detach()
        qs_predicted = torch.tensor(rewards) + GAMMA*(1-torch.FloatTensor(dones)) * next_qs_max
        loss = F.mse_loss(qs_actions, qs_predicted)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        length = np.mean([len(generate_episode(env, agent, eval=True)) for _ in range(10)])

        if idx % 100 == 0:
            print(f"Step {idx} - length = {length} - loss = {loss:4.3f}")

if __name__ == '__main__':
    GAMMA = 0.99
    BUFFER_SIZE = 2048
    BATCH_SIZE = 512
    LEARNING_RATE = 0.01
    HIDDEN_SIZE = 128
    train(n_steps=2000)