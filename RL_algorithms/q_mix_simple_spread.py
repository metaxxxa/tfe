from ast import arg
from tkinter import N
from types import AsyncGeneratorType
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
class Args:
    def __init__(self, env):
            
        self.BUFFER_SIZE = 1000
        self.REW_BUFFER_SIZE = 100
        self.LEARNING_RATE = 1e-4
        self.MIN_BUFFER_LENGTH = 1000
        self.BATCH_SIZE = 200
        self.GAMMA = 0.9
        self.EPSILON_START = 1
        self.EPSILON_END = 0.001
        self.EPSILON_DECAY = 100000
        self.SYNC_TARGET_FRAMES = 100
        self.VISUALIZE_WHEN_LEARNED = True
        
        #agent network parameters
        self.COMMON_AGENTS_NETWORK = True
        self.dim_L1_agents_net = 32
        self.dim_L2_agents_net = 32
        #mixing network parameters
        self.mixer_hidden_dim = 32
        self.mixer_hidden_dim2 = 32
        #environment specific parameters calculation
        self.params(env)

    def params(self, env):  #environment specific parameters calculation
        self.n_agents = env.num_agents
        agent = 'agent_0'
        self.nb_inputs_agent = np.prod(env.observation_space(agent).shape)
        self.observations_dim = env.observation_space(agent).shape[0]
        self.n_actions = env.action_space(agent).n



class QMixer(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.to(device)
        self.agent_nets = dict()
        #params
        self.args = args
        total_state_dim = 0
        if self.args.COMMON_AGENTS_NETWORK:
            agents_net = AgentRNN(self.args)
            for agent in env.agents:
                self.agent_nets[agent] = agents_net
                total_state_dim += np.prod(env.observation_space(agent).shape)
        else:
            for agent in env.agents:
                self.agent_nets[agent] = AgentRNN(args)
                total_state_dim += np.prod(env.observation_space(agent).shape)
        
        self.weightsL1_net = nn.Linear(total_state_dim, self.args.mixer_hidden_dim*self.args.n_agents).to(device)
        self.biasesL1_net = nn.Linear(total_state_dim, self.args.mixer_hidden_dim).to(device)
        
        self.weightsL2_net = nn.Linear(total_state_dim, self.args.mixer_hidden_dim2*self.args.mixer_hidden_dim).to(device)
        self.biasesL2_net = nn.Sequential(
            nn.Linear(total_state_dim, self.args.mixer_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.mixer_hidden_dim, self.args.mixer_hidden_dim2)
        ).to(device)
        agent_params = list()
        if self.args.COMMON_AGENTS_NETWORK:
            self.net_params = list(agents_net.parameters()) + list(self.weightsL1_net.parameters()) + list(self.biasesL1_net.parameters())  +list(self.weightsL2_net.parameters()) + list(self.biasesL2_net.parameters())
        else:
            for agent_net in self.agent_nets.values():
                agent_params += agent_net.net.parameters()
            self.net_params = agent_params + list(self.weightsL1_net.parameters()) + list(self.biasesL1_net.parameters())  +list(self.weightsL2_net.parameters()) + list(self.biasesL2_net.parameters())
            
    
    def forward(self, obs_tot,Qin_t):
        weightsL1 = torch.abs(self.weightsL1_net(obs_tot)) # abs: monotonicity constraint
        weightsL1_tensor = weightsL1.unsqueeze(-1).reshape([self.args.BATCH_SIZE, self.args.mixer_hidden_dim, self.args.n_agents])
        biasesL1 = self.biasesL1_net(obs_tot)
        weightsL2 = torch.abs(self.weightsL2_net(obs_tot))
        weightsL2_tensor = weightsL2.unsqueeze(-1).reshape([self.args.BATCH_SIZE, self.args.mixer_hidden_dim2, self.args.mixer_hidden_dim])
        biasesL2 = self.biasesL2_net(obs_tot)
        l1 = torch.matmul(weightsL1_tensor, Qin_t.unsqueeze(-1)).squeeze(-1) + biasesL1
        l1 = nn.ELU(l1).alpha
        Qtot = torch.matmul(weightsL2_tensor, l1.unsqueeze(-1)).squeeze(-1) + biasesL2
        Qtot = Qtot.sum(1)
        
        return Qtot
        
    def get_Q_values(self, agent, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
        q_values = self.agent_nets[agent].net(obs_t.unsqueeze(0))
        return q_values

    def get_Q_max(self, q_values):
        max_q_index = torch.argmax(q_values, dim=1)[0]
        max_q_index = max_q_index.detach().item()
        max_q = q_values[0,max_q_index]
        return max_q_index, max_q

    def act(self, agent, obs):
        action, _ = self.get_Q_max(self.get_Q_values(agent, obs))
        return action
    
    
class AgentRNN(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(args.nb_inputs_agent, args.dim_L1_agents_net),
                nn.ELU(),
                nn.Linear(args.dim_L1_agents_net,args.dim_L2_agents_net), #nn.GRU(dim_L2_agents_net    ,32),
                nn.ELU(),
                nn.Linear(args.dim_L2_agents_net, args.n_actions)
            ).to(device)

class runner_QMix:
    def __init__(self, env, args):
        self.args = args
        self.env = env
        
        self.replay_buffer = deque(maxlen=self.args.BUFFER_SIZE)
        self.rew_buffer = deque([0.0], maxlen=args.REW_BUFFER_SIZE)
        self.loss_buffer = deque([0.0], maxlen=args.REW_BUFFER_SIZE)

        self.online_net = QMixer(self.env, self.args)
        self.target_net = QMixer(self.env, self.args)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.net_params, lr = self.args.LEARNING_RATE)

    def run(self):
        
        #Init replay buffer

        self.env.reset()
        one_agent_done = 0

        observation_prev = dict()
        observation = dict()
        episode_reward = 0.0
        for agent in self.env.agents:
            observation_prev[agent], _, _, _ = self.env.last()
        for _ in range(args.MIN_BUFFER_LENGTH):
            
            transition = dict()
            for agent in self.env.agent_iter(max_iter=len(self.env.agents)):
                action = self.env.action_space(agent).sample()
                if one_agent_done:
                    self.env.step(None)
                else:
                    self.env.step(action)
                observation[agent], reward, done, info = self.env.last()
                episode_reward += reward
                transition[agent] = (observation_prev[agent], action,reward,done,observation[agent])
                observation_prev[agent] = observation[agent]
                if done:
                    one_agent_done = 1 #if one agent is done, all have to stop
            if one_agent_done:
                obs = self.env.reset()
                self.rew_buffer.append(episode_reward)
                episode_reward = 0.0
                one_agent_done = 0
                for agent in self.env.agents:
                    observation_prev[agent], _, _, _ = self.env.last()
            
            self.replay_buffer.append(transition)

        # trainingoptim

        self.env.reset()
        episode_reward = 0.0
        for agent in self.env.agents:
            observation_prev[agent], _, _, _ = self.env.last()

        for step in itertools.count():
            epsilon = np.interp(step, [0, args.EPSILON_DECAY], [args.EPSILON_START, args.EPSILON_END])
            rnd_sample = random.random()
            transition = dict()
            for agent in self.env.agent_iter(max_iter=len(self.env.agents)):
                if rnd_sample <= epsilon:
                    action = self.env.action_space(agent).sample()
                else:
                    action = self.online_net.act(agent, observation_prev[agent])
                if one_agent_done:
                    self.env.step(None)
                else:
                    self.env.step(action)
                observation[agent], reward, done, info = self.env.last()
                episode_reward += reward
                transition[agent] = (observation_prev[agent], action,reward,done,observation[agent])
                observation_prev[agent] = observation[agent]
                if done:
                    one_agent_done = 1 #if one agent is done, all have to stop
            if one_agent_done:
                obs = self.env.reset()
                self.rew_buffer.append(episode_reward)
                writer.add_scalar("Reward", episode_reward,step  )
                episode_reward = 0.0
                one_agent_done = 0
                for agent in self.env.agents:
                    observation_prev[agent], _, _, _ = self.env.last()
            
            self.replay_buffer.append(transition)
            

            #gradient stept[agent][1]

            transitions = random.sample(self.replay_buffer, args.BATCH_SIZE)
            obses_t = torch.empty((self.args.BATCH_SIZE,self.args.n_agents*self.args.observations_dim)).to(device)
            actions_t = torch.empty((self.args.BATCH_SIZE,self.args.n_agents)).to(device)
            Q_ins_target_t = torch.empty((self.args.BATCH_SIZE,self.args.n_agents)).to(device)
            Q_action_online_t = torch.empty((self.args.BATCH_SIZE,self.args.n_agents)).to(device)
            rewards_t = torch.empty((self.args.BATCH_SIZE,self.args.n_agents)).to(device)
            dones_t = torch.empty((self.args.BATCH_SIZE,self.args.n_agents)).to(device)
            new_obses_t = torch.empty((self.args.BATCH_SIZE,self.args.n_agents*self.args.observations_dim)).to(device)
            transition_nb = 0
            for t in transitions:
                
                agent_nb = 0
                for agent in self.env.agents:
                    obses_t[transition_nb][self.args.observations_dim*agent_nb:(self.args.observations_dim*(agent_nb+1))] = torch.as_tensor(t[agent][0], dtype=torch.float32, device=device)
                    actions_t[transition_nb][agent_nb] = t[agent][1]
                    rewards_t[transition_nb][agent_nb] = t[agent][2]
                    dones_t[transition_nb][agent_nb] = t[agent][3]
                    new_obses_t[transition_nb][self.args.observations_dim*agent_nb:(self.args.observations_dim*(agent_nb+1))] = torch.as_tensor(t[agent][4], dtype=torch.float32, device=device)

                    Q_action_online_t[transition_nb][agent_nb] = torch.gather(self.online_net.get_Q_values(agent, t[agent][0]).squeeze(0), 0,torch.tensor([t[agent][1]]).to(device))
                    Q_ins_target_t[transition_nb][agent_nb] = self.target_net.get_Q_max(self.target_net.get_Q_values(agent, t[agent][4]))[1]
                     
                    agent_nb += 1


                transition_nb += 1

            #compute reward for all agents
            rewards_t = rewards_t.sum(1)
            #if one agent is done all are
            dones_t = dones_t.sum(1)
            dones_t = dones_t > 0
            # targets
            Qtot_max_target = self.target_net.forward(new_obses_t, Q_ins_target_t) 
            Qtot_online = self.online_net.forward(obses_t, Q_action_online_t)
            y_tot = rewards_t + self.args.GAMMA*(1 + (-1)*dones_t)*Qtot_max_target

        ########### busy
            # loss 
            error = y_tot + (-1)*Qtot_online
            
            loss = error**2
            loss = loss.sum()
            self.loss_buffer.append(loss.detach().item())
            writer.add_scalar("Loss", loss, step)
            


            # gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #update target network

            if step % args.SYNC_TARGET_FRAMES == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            #logging
            if step % 1000 == 0:
                print('\n Step', step )
                print('Avg Reward', np.mean(self.rew_buffer))
                print('Avg Loss', np.mean(self.loss_buffer))
        writer.close() 

        ###
if __name__ == "__main__":
    env = simple_spread_v2.env(N=2, local_ratio=0.5, max_cycles=25, continuous_actions=False)
    env.reset()
    args = Args(env)
    runner = runner_QMix(env, args)
    runner.run()
