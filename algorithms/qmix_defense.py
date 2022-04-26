""" 
Implementation of **QMix**
Comments:
    * based on `vdn2.py`
    * integrate mixer in Runner & make use of it optional
    * share network between agents
    * Uses QMixer


Applied to **defense**:
    * observation is a dictionary with keys:
        'obs': the actual observation as a ndarry containing:
            * self: pos_x, pos_y, alive flag, remaining ammo, aiming (-1 if not)
            * team: same information for agents of own team
            * others: same information for agents of other team
            * obstacles: pos_x and pos_y for obstacles
        'action_mask': Box of lenght len(actions) with: 
            0 : action not allowed
            1 : action allowed
"""
import copy
import random
import sys
import inspect # get source code
from collections import deque

import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

from pettingzoo.mpe import simple_v2, simple_spread_v2

from helpers import EpisodeStep, Runner
from helpers import build_network, transform_episode

from torch.utils.tensorboard import SummaryWriter


# hack to allow import of env
import sys; sys.path.insert(0, '.')
from env import defense_v0
ENVIRONMENT = defense_v0

OPTIMIZER =  optim.Adam # optim.RMSprop #
LOCAL_RATIO = 0. #3 #1 # trade-off between local (collisions) and global reward (close to landmarks)

class RandomAgent:
    def __init__(self, actions):
        self.actions = actions
    
    def get_action(self, observation):
        return random.choice(self.actions)

def untangle(batch):
    observations = [{agent: step[agent].observation for agent in step} for step in batch]
    next_obs     = [{agent: step[agent].next_obs for agent in step} for step in batch]
    actions      = [{agent: step[agent].action for agent in step} for step in batch]
    # rewards and dones should be the same for all agents, so choose agent_0
    # => fully cooperative setting: all agents share a common reward function 
    rewards      = torch.tensor([[step[agent].reward for agent in step] for step in batch]).to(device)
    dones        = torch.tensor([[step[agent].done for agent in step] for step in batch], dtype=torch.float32).to(device)
    states       = torch.from_numpy(np.stack([[step[agent].state for agent in step] for step in batch])).to(device)
    next_states  = torch.from_numpy(np.stack([[step[agent].next_state for agent in step] for step in batch])).to(device)
    masks        = torch.from_numpy(np.stack([[step[agent].mask for agent in step] for step in batch])).to(device)
    next_masks   = torch.from_numpy(np.stack([[step[agent].next_mask for agent in step] for step in batch])).to(device)
    return observations, masks, actions, rewards.T, dones.T, next_obs, next_masks, states, next_states

class EpisodeStep:
    def __init__(self, observation, mask, action, reward, done, next_obs, next_mask, state, next_state):
        self.observation = observation
        self.mask = mask
        self.action = action
        self.reward = reward
        self.done = done
        self.next_obs = next_obs
        self.next_mask = next_mask
        self.state = state
        self.next_state = next_state
    
    def __iter__(self):
        all = self.__dict__.values()
        return iter(all)
    
    def __repr__(self):
        s  = f"observation = {self.observation}\n" 
        s += f"action = {self.action}\n"
        s += f"reward = {self.reward}\n"
        s += f"done = {self.done}\n"
        s += f"next_obs = {self.next_obs}\n"
        return s

class QMixRunner(Runner):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.init_params(kwargs)
        self.env = ENVIRONMENT.env(terrain="flat_5x5", max_cycles=self.episode_length, max_distance=4)
        self.env.reset()
        
        
        N = len(self.env.agents)
        dims = self.env.observation_space('blue_0')['obs'].shape[0]
        # dims = 4 + 2 * N + 2 * (N - 1) + 2 * (N-1) # vx, vy, x, y, [landmark.x, landmark.y], [other.x, other.y], comms
        n_actions = self.env.action_space('blue_0').n
        # n_actions = 5 # [no_action, move_left, move_right, move_down, move_up]
        self.net = build_network([dims, *self.layers, n_actions]).to(device)

        self.agents = {}
        for agent in self.env.agents:
            self.agents[agent] = DQNAgent(name=agent, actions=range(n_actions), net=self.net)
                                        
        self.buffer = deque(maxlen=self.buffer_size)
        if self.use_mixer:
            self.mixer = QMixer(self.n_agents, self.env.state().shape[0], self.mixer_layer_size).to(device)
            self.target_mixer = copy.deepcopy(self.mixer).to(device)

        params = list(self.net.parameters()) # shared network amongst (identical) agents
        if self.use_mixer:
            params += list(self.mixer.parameters())
        self.optimizer = OPTIMIZER(params, lr=self.lr)
        
        self.writer = SummaryWriter(log_dir=f"runs/qmix_{str(self.rand_idx)}_{str(self.n_agents)}")
        

    def init_params(self, kwargs):
        self.n_agents = kwargs.get('n_agents', 2)
        self.episode_length = kwargs.get('episode_length', 200)
        self.layers = kwargs.get('layers', [128, 128])
        self.buffer_size = kwargs.get('buffer_size', 1024)
        self.gamma = kwargs.get('gamma', 0.99)
        self.lr = kwargs.get('lr', 0.001)
        self.n_batches = kwargs.get('n_batches', 32)     # K steps
        self.sample_size = kwargs.get('sample_size', 64) # N examples from replay memory
        self.n_evals = kwargs.get('n_evals', 10)

        self.epsilon = 1.0
        self.eps_min = kwargs.get('eps_min', 0.1)
        self.eps_steps = kwargs.get('eps_steps', 200000)
        self.eps_decay = (self.epsilon - self.eps_min)/self.eps_steps

        self.sync_rate = kwargs.get('sync_rate', 1)
        self.use_mixer = kwargs.get('use_mixer', True)
        self.mixer_layer_size = kwargs.get('mixer_layer_size', 16)

        self.verbose = kwargs.get('verbose', True)
        self.rand_idx = random.randint(0, 100000)
    
    def __str__(self):
        s  = f"##QMix applied to {self.env}\n------------------------------------\n"
        s += f"Number of agents = {len(self.agents)}  \n"
        s += f"Episode length = {self.episode_length}  \n"
        s += f"buffer size = {self.buffer_size}  \n"
        s += f"gamma = {self.gamma}  \n"
        s += f"learning rate = {self.lr}  \n"
        s += f"layers = {self.layers}  \n"
        s += f"eps_min = {self.eps_min}  \n"
        s += f"eps_steps = {self.eps_steps}  \n"
        s += f"sync_rate = {self.sync_rate}  \n"
        s += f"n_batches = {self.n_batches}  \n"
        s += f"sample_size = {self.sample_size}  \n"
        s += f"optimizer = {OPTIMIZER.__module__.split('.')[-1]}  \n"
        s += f"{'with' if self.use_mixer else 'without'} mixer  \n"
        s += f"device = {device}  \n"
        s += f"identifier = {self.rand_idx}  \n"
        return s
    
    def run(self, n_iters=10):
        for indx in range(n_iters):
            for _ in range(10):
                episode = self.generate_episode()
                self.buffer.extend(episode[:-1]) # last elem of episode contains no useful information
            if len(self.buffer) < self.sample_size:
                continue

            cum_loss = 0.0
            for _ in range(self.n_batches):
                batch = random.sample(self.buffer, k=self.sample_size)
                observations, actions, rewards, dones, next_obs, states, next_states = untangle(batch)
                qs = torch.zeros((len(self.agents), len(observations))).to(device)
                next_qs = torch.zeros((len(self.agents), len(observations))).to(device)
                for idx, agent in enumerate(self.agents):
                    agent_actions = [a[agent] for a in actions]
                    observation = torch.from_numpy(np.stack([o[agent] for o in observations])).float().to(device)
                    qs[idx, :] = self.agents[agent].net(observation)[range(len(observations)), agent_actions]
                    next_o = torch.from_numpy(np.stack([o[agent] for o in next_obs])).float().to(device)
                    next_qs[idx, :] = self.agents[agent].target_net(next_o).max(dim=1)[0].detach()

                if self.use_mixer:
                    rewards = rewards[0, :]     # rewards and dones are common for all agents, (in this environment)
                    dones = dones[0, :]         # thus this just reduces size of matrix to vector
                    states = states[:, 0, :]    # => fully cooperative setting, all agents share a common reward function
                    next_states = next_states[:, 0, :]                                          
                    qs = self.mixer(qs.T, states) # = Qtot
                    next_qs = self.target_mixer(next_qs.T, next_states).detach()                                           
                    
                q_target = rewards + self.gamma * (1 - dones) * next_qs
                loss = F.mse_loss(qs, q_target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cum_loss += loss.item()
            
            # log statistics 
            self.log(indx, n_iters, cum_loss)

            # sync networks
            if indx % self.sync_rate == 0:
                for agent in self.agents.values():
                    agent.sync()
                    agent.save(self.rand_idx)
                if self.use_mixer:
                    self.target_mixer.load_state_dict(self.mixer.state_dict())
    
    def generate_episode(self, render=False, train=True):
        self.env.reset()
        episode = {agent: [] for agent in self.env.agents}
        done = False
        for agent in self.env.agent_iter():
            if render:
                self.env.render()
            observation, reward, done, _ =  self.env.last()
            obs = observation['obs']
            mask = observation['action_mask']
            state = self.env.state()
            # set observation, done and reward as next_obs of previous step
            if episode[agent]:
                episode[agent][-1].next_obs = obs
                episode[agent][-1].next_mask = mask
                episode[agent][-1].next_state = state
                episode[agent][-1].reward = reward
                episode[agent][-1].done = done
            action = None if done else self.agents[agent].get_action(obs, mask, epsilon=self.epsilon if train else 0.0)
            self.env.step(action)
            episode[agent].append(EpisodeStep(observation, mask, action, None, None, None, None, state, None))
            if train:
                self.epsilon = max(self.epsilon-self.eps_decay, self.eps_min)
        return transform_episode(episode)
    
    def log(self, indx, losses):
        avg_rwd, std_rwd, avg_length = self.eval(self.n_evals)
        for agent in self.learners:
            avg_loss, std_loss = np.mean(losses[agent]), np.std(losses[agent])
            if self.verbose:
                print(f"{indx} - {agent:11s}: loss = {avg_loss:5.4f}, avg reward = {avg_rwd[agent]:5.4f}")
            self.writers[agent].add_scalar('avg_loss', avg_loss, indx)
            self.writers[agent].add_scalar('avg_reward', avg_rwd[agent], indx)
            self.writers[agent].add_scalar('avg_length', avg_length, indx)

    def eval(self, n):
        rewards = {agent: [] for agent in self.agents}
        lengths = []
        for _ in range(n):
            episode = self.generate_episode(train=False)
            for agent in self.agents:
                rewards[agent].append(np.sum([step[agent].reward for step in episode[:-1]]))
            lengths.append(len(episode))
        means, stds = {}, {}
        for agent in self.agents:
            means[agent] = np.mean(rewards[agent]) # TODO: better solution (eg. divide by initial reward)
            stds[agent]  = np.std(rewards[agent]) 

        return means, stds, np.mean(lengths)

class DQNAgent:
    def __init__(self, name, actions, net):
        self.name = name
        self.actions = actions
        self.net = net
        self.target_net = copy.deepcopy(self.net)
        self.epsilon = 1.0

    def get_action(self, observation, action_mask, epsilon=0.0):
        if np.random.rand() < epsilon:
            p = action_mask/sum(action_mask)
            action = np.random.choice(range(len(action_mask)), p=p)
        else:
            with torch.no_grad():
                qs = self.net(torch.from_numpy(observation).float().to(device))
                qs[action_mask == 0.0] = -np.infty # set all invalid actions to lowest q_value possible
                action = qs.argmax(axis=0).item()
        return action
    
    def sync(self):
        self.target_net.load_state_dict(self.net.state_dict())
    
    def save(self, rand_idx):
        filename = f"{rand_idx}_{self.name}"
        torch.save(self.net.state_dict(), './nets/qmix/'+filename)
    
    def load(self, rand_idx):
        filename = f"{rand_idx}_{self.name}"
        self.net.load_state_dict(torch.load('./nets/qmix/'+filename))
        self.net.to(device)

class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, layer_size=16):
        super().__init__()
        self.n_agents = n_agents
        self.layer_size = layer_size
        self.W1_layer = nn.Linear(state_dim, n_agents*layer_size)
        self.W2_layer = nn.Linear(state_dim, layer_size)
        self.b1_layer = nn.Linear(state_dim, layer_size)
        self.b2_layer = nn.Linear(state_dim, 1)
        #self.W1_layer = nn.Linear(state_dim, n_agents*1)

    def forward(self, qs, state):
        W1 = torch.abs(self.W1_layer(state))
        W1 = W1.reshape(-1, self.layer_size, self.n_agents)
        W2 = torch.abs(self.W2_layer(state))
        W2 = W2.reshape(-1, 1, self.layer_size)
        b1 = self.b1_layer(state)
        b2 = torch.relu(self.b2_layer(state))

        x = F.elu(torch.bmm(W1, qs[:, :, None]) + b1[:,:, None])
        Q = torch.bmm(W2, x).squeeze() + b2
        return Q
    
    def forward_(self, qs, state):
        "Alternative - testing only"
        W1 = torch.abs(self.W1_layer(state))
        W1 = W1.reshape(-1, 1, self.n_agents)

        Q = torch.bmm(W1, qs[:, :, None]).squeeze()
        return Q

def train(args):
    runner = QMixRunner(episode_length=int(args.length),
                        buffer_size=int(args.buffer_size),
                        n_batches=int(args.n_batches),
                        sample_size=int(args.sample_size), # 64
                        lr=float(args.lr),
                        gamma=float(args.gamma),
                        sync_rate=int(args.sync_rate),
                        layers=[32, 32],
                        eps_steps=int(args.eps_steps),
                        eps_min=0.1,
                        use_mixer=args.use_mixer,
                        n_agents=int(args.n_agents),
                        mixer_layer_size=int(args.mixer_layer_size),
    )
    print(runner)
    runner.writer.add_text('parameters', runner.__str__())
    runner.writer.add_text('source', inspect.getsource(sys.modules[__name__])) # write source code to tensorboard - usefull for experiment tracking
    runner.run(n_iters=int(args.n_iters))

    rand_idx = random.randint(0, 100000)
    for agent in runner.agents.values():
        agent.save(runner.rand_idx)
    print(f'--- Saved with id {rand_idx} ---')
    return runner

def demo():
    """Keep default parameters but:
    * gamma = 0.9
    * learning_rate = 0.001
    * sync_rate = 100
    Gives good result for spread_v2 after 200 iterations
    """

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--gamma", default=0.9)
    parser.add_argument("--sync_rate", default=100)
    parser.add_argument("--n_iters", default=5000)
    parser.add_argument("--buffer_size", default=2048)
    parser.add_argument("--n_batches", default=64)
    parser.add_argument("--sample_size", default=128)
    parser.add_argument("--length", default=100) # episode_length
    parser.add_argument("--eps_steps", default=int(5e6)) # how many steps before epsilon = eps_min
    parser.add_argument('--use_mixer', dest='use_mixer', action='store_true')
    parser.add_argument('--no_mixer', dest='use_mixer', action='store_false')
    parser.set_defaults(use_mixer=True)
    parser.add_argument("--mixer_layer_size", default=16)
    parser.add_argument("--n_agents", default=2)
    args = parser.parse_args()

    runner = train(args)
    while True:
        runner.generate_episode(train=False, render=True)

if __name__ == '__main__':
    demo()