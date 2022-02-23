from tkinter import N
import torch
import torch.nn as nn
import numpy as np
from pettingzoo.mpe import simple_spread_v2
from collections import deque
import itertools
import random
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev) 

#setting up TensorBoard

writer = SummaryWriter()

#parameters

BUFFER_SIZE = 1000
REW_BUFFER_SIZE = 100
LEARNING_RATE = 1e-4
MIN_BUFFER_LENGTH = 100
BATCH_SIZE = 100
GAMMA = 0.99
EPSILON_START = 1
EPSILON_END = 0.02
EPSILON_DECAY = 10000
SYNC_TARGET_FRAMES = 1000

env = simple_spread_v2.env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False)
env.reset()

class QMixer(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.to(device)
        self.agent_nets = dict()
        #params

        dim_L1_agents_net = 32
        dim_L2_agents_net = 32

        total_state_dim = 0
        for agent in env.agents:
            nb_inputs = np.prod(env.observation_space(agent).shape)
            nb_outputs = env.action_space(agent).n
            total_state_dim += nb_inputs
            self.agent_nets[agent] = nn.Sequential(
                nn.Linear(nb_inputs, dim_L1_agents_net),
                nn.ELU(),
                nn.Linear(dim_L1_agents_net,dim_L2_agents_net), #nn.GRU(dim_L2_agents_net    ,32),
                nn.ELU(),
                nn.Linear(dim_L2_agents_net, nb_outputs)
            ).to(device)

        mixer_hidden_dim = 32

        self.weightsL1_net = nn.Linear(total_state_dim, mixer_hidden_dim).to(device)
        self.biasesL1_net = nn.Linear(total_state_dim, mixer_hidden_dim).to(device)
        
        self.weightsL2_net = nn.Linear(total_state_dim, mixer_hidden_dim).to(device)
        self.biasesL2_net = nn.Sequential(
            nn.Linear(total_state_dim, mixer_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixer_hidden_dim, 1)
        ).to(device)
        agent_params = list()
        for net in self.agent_nets.values():
            agent_params += net.parameters()
        self.net_params = agent_params + list(self.weightsL1_net.parameters()) + list(self.biasesL1_net.parameters())  +list(self.weightsL2_net.parameters()) + list(self.biasesL2_net.parameters())
        
    
    def forward(self, obs_tot,Qin_t):
        weightsL1 = torch.abs(self.weightsL1_net(obs_tot)) # abs: monotonicity constraint
        biasesL1 = self.biasesL1_net(obs_tot)
        weightsL2 = torch.abs(self.weightsL2_net(obs_tot))
        biasesL2 = self.biasesL2_net(obs_tot)
        l1 = weightsL1*Qin_t.sum(1)[:,None] + biasesL1
        l1 = nn.ELU(l1).alpha
        Qtot = weightsL2*l1.sum(1)[:,None]
        Qint = Qtot.sum(1)
        Qtot = Qtot.sum(1).unsqueeze(-1) + biasesL2
        
        return Qtot
        
    def get_Q_values(self, agent, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
        q_values = self.agent_nets[agent](obs_t.unsqueeze(0))
        return q_values

    def get_Q_max(self, q_values):
        max_q_index = torch.argmax(q_values, dim=1)[0]
        max_q_index = max_q_index.detach().item()
        max_q = q_values[0,max_q_index]
        return max_q_index, max_q

    def act(self, agent, obs):
        action, _ = self.get_Q_max(self.get_Q_values(agent, obs))
        return action
    
    



replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0.0], maxlen=REW_BUFFER_SIZE)
loss_buffer = deque([0.0], maxlen=REW_BUFFER_SIZE)

episode_reward = 0.0

online_net = QMixer(env)
target_net = QMixer(env)

target_net.load_state_dict(online_net.state_dict())


###


optimizer = torch.optim.Adam(online_net.net_params, lr = LEARNING_RATE)

#Init replay buffer

env.reset()
one_agent_done = 0

observation_prev = dict()
observation = dict()
episode_reward = 0.0
for agent in env.agents:
    observation_prev[agent], _, _, _ = env.last()
for _ in range(MIN_BUFFER_LENGTH):
    
    transition = dict()
    for agent in env.agent_iter(max_iter=len(env.agents)):
        action = env.action_space(agent).sample()
        if one_agent_done:
            env.step(None)
        else:
            env.step(action)
        observation[agent], reward, done, info = env.last()
        episode_reward += reward
        transition[agent] = (observation_prev[agent], action,reward,done,observation[agent])
        observation_prev[agent] = observation[agent]
        if done:
            one_agent_done = 1 #if one agent is done, all have to stop
    if one_agent_done:
        obs = env.reset()
        rew_buffer.append(episode_reward)
        episode_reward = 0.0
        one_agent_done = 0
        for agent in env.agents:
            observation_prev[agent], _, _, _ = env.last()
    
    replay_buffer.append(transition)

# trainingoptim

env.reset()
episode_reward = 0.0
for agent in env.agents:
    observation_prev[agent], _, _, _ = env.last()

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    rnd_sample = random.random()
    transition = dict()
    for agent in env.agent_iter(max_iter=len(env.agents)):
        if rnd_sample <= epsilon:
            action = env.action_space(agent).sample()
        else:
            action = online_net.act(agent, observation_prev[agent])
        if one_agent_done:
            env.step(None)
        else:
            env.step(action)
        observation[agent], reward, done, info = env.last()
        episode_reward += reward
        transition[agent] = (observation_prev[agent], action,reward,done,observation[agent])
        observation_prev[agent] = observation[agent]
        if done:
            one_agent_done = 1 #if one agent is done, all have to stop
    if one_agent_done:
        obs = env.reset()
        rew_buffer.append(episode_reward)
        writer.add_scalar("Reward", episode_reward,step  )
        episode_reward = 0.0
        one_agent_done = 0
        for agent in env.agents:
            observation_prev[agent], _, _, _ = env.last()
    
    replay_buffer.append(transition)
    


########## checkpoint
    #gradient step

    transitions = random.sample(replay_buffer, BATCH_SIZE)
    obses_t = np.empty((0,len(env.agents)*np.prod(env.observation_space(agent).shape)), np.float64)
    actions = np.empty((0,len(env.agents)), np.float64)
    Q_ins_target = np.empty((0,len(env.agents)), np.float64)
    Q_ins_online = np.empty((0,len(env.agents)), np.float64)
    rewards = np.empty((0,len(env.agents)), np.float64)
    dones = np.empty((0,len(env.agents)), np.float64)
    new_obses_t = np.empty((0,len(env.agents)*np.prod(env.observation_space(agent).shape)), np.float64)
    for t in transitions:
        obs = np.array([])
        q_max_online = np.array([])
        q_max_target = np.array([])
        acts = np.array([])
        rews = np.array([])
        done = np.array([]) 
        new_obs = np.array([])
        for agent in env.agents:
            obs = np.concatenate((obs, t[agent][0]))
            q_max_online = np.append(q_max_online, online_net.get_Q_max(online_net.get_Q_values(agent, t[agent][0]))[1].cpu().detach())
            q_max_target = np.append(q_max_target, target_net.get_Q_max(target_net.get_Q_values(agent, t[agent][4]))[1].cpu().detach())
            acts = np.append(acts, t[agent][1])
            rews = np.append(rews, t[agent][2])
            done = np.append(done, t[agent][3])
            new_obs = np.concatenate((new_obs, t[agent][4]))
        obses_t = np.append(obses_t, [obs], axis = 0)
        Q_ins_online = np.append(Q_ins_online, [q_max_online], axis = 0)
        Q_ins_target = np.append(Q_ins_target, [q_max_target], axis = 0)
        actions = np.append(actions, [acts], axis = 0)
        rewards = np.append(rewards, [rews], axis = 0)
        dones = np.append(dones, [done], axis = 0)
        new_obses_t = np.append(new_obses_t, [new_obs], axis = 0)
    Q_ins_online_t = torch.as_tensor(Q_ins_online, dtype=torch.float32, device=device)
    Q_ins_target_t = torch.as_tensor(Q_ins_target, dtype=torch.float32, device=device)
    obses_t = torch.as_tensor(obses_t, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(-1)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses_t, dtype=torch.float32, device=device)
    
    #compute reward for all agents
    rewards_t = rewards_t.sum(1)
    #if one agent is done all are
    dones_t = dones_t.sum(1)
    dones_t = dones_t > 0
    # targets
    Qtot_max_target = target_net.forward(new_obses_t, Q_ins_target_t) #.max(dim=1, keepdim=True)[0]
    Qtot_online = online_net.forward(obses_t, Q_ins_online_t)
    y_tot = rewards_t + GAMMA*(1 + (-1)*dones_t)*Qtot_max_target

########### busy
    # loss 
    error = y_tot + (-1)*Qtot_online
    
    loss = error**2
    loss = loss.sum()
    loss_buffer.append(loss.detach().item())
    writer.add_scalar("Loss", loss, step)
    


    # gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #update target network

    if step % SYNC_TARGET_FRAMES == 0:
        target_net.load_state_dict(online_net.state_dict())

    #logging
    if step % 1000 == 0:
        print('\n Step', step )
        print('Avg Rew', np.mean(rew_buffer))
        print('Avg Loss', np.mean(loss_buffer))
writer.close() 

###
