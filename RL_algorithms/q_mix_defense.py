
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import itertools
import random
from torch.utils.tensorboard import SummaryWriter
import copy
import os
import sys
import time
import re
#importing the defense environment
os.chdir('/home/jack/Documents/ERM/Master thesis/tfe')
sys.path.insert(0, '/home/jack/Documents/ERM/Master thesis/tfe')
from env import defense_v0

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
  
device = torch.device(dev) 


#environment constants
EPISODE_MAX_LENGTH = 200
MAX_DISTANCE = 5
TERRAIN = 'flat_5x5_2v2'
#parameters
class Args:
    def __init__(self, env):
            
        self.BUFFER_SIZE = 200
        self.REW_BUFFER_SIZE = 100
        self.LEARNING_RATE = 1e-4
        self.MIN_BUFFER_LENGTH = 300
        self.BATCH_SIZE = 32
        self.GAMMA = 0.99
        self.EPSILON_START = 1
        self.EPSILON_END = 0.02
        self.EPSILON_DECAY = 200000
        self.SYNC_TARGET_FRAMES = 200
        #visualization parameters
        self.VISUALIZE_WHEN_LEARNED = True
        self.VISUALIZE_AFTER = 500000
        self.VISUALIZE = False
        self.WAIT_BETWEEN_STEPS = 0.1
        self.GREEDY = True
        self.SAVE_CYCLE = 10000
        self.MODEL_DIR = 'defense_params'
        self.RUN_NAME = ''
        #agent network parameters
        self.COMMON_AGENTS_NETWORK = True
        self.dim_L1_agents_net = 32
        self.dim_L2_agents_net = 32
        #mixing network parameters
        self.mixer_hidden_dim = 32
        self.mixer_hidden_dim2 = 32
        #environment specific parameters calculation
        self.TEAM_TO_TRAIN = 'blue'
        self.OPPOSING_TEAM = 'red'
        self.ADVERSARY_TACTIC = 'random'
        self.params(env)

    def params(self, env):  #environment specific parameters calculation
        
        self.blue_agents = [key for key in env.agents if re.match(rf'^{self.TEAM_TO_TRAIN}',key)]
        self.all_agents = env.agents
        self.opposing_agents = [key for key in env.agents if re.match(rf'^{self.OPPOSING_TEAM}',key)]
        self.n_blue_agents = len(self.blue_agents)
        agent = self.blue_agents[0]
        self.nb_inputs_agent = np.prod(env.observation_space(agent).spaces['obs'].shape)
        self.observations_dim = np.prod(env.observation_space(agent).spaces['obs'].shape)
        self.n_actions = env.action_space(agent).n
    def log_params(self, writer):
        hparams = {'envparam/terrrain': TERRAIN, 'Adversary tactic' : self.ADVERSARY_TACTIC, 'Learning rate': self.LEARNING_RATE, 'Batch size': self.BATCH_SIZE, 'Buffer size': self.BUFFER_SIZE, 'Min buffer length': self.MIN_BUFFER_LENGTH, '\gamma': self.GAMMA, 'Epsilon range': f'{self.EPSILON_START} - {self.EPSILON_END}', 'Epsilon decay': self.EPSILON_DECAY, 'Synchronisation rate': self.SYNC_TARGET_FRAMES, 'Timestamp': int(datetime.timestamp(datetime.now()) - datetime.timestamp(datetime(2022, 2, 1, 11, 26, 31,0))), 'Common agent network': int(self.COMMON_AGENTS_NETWORK)}
        metric_dict = { 'hparam/dim L1 agent net': self.dim_L1_agents_net, 'hparam/dim L2 agent net': self.dim_L2_agents_net, 'hparam/mixer hidden dim 1': self.mixer_hidden_dim, 'hparam/mixer hidden dim 2': self.mixer_hidden_dim2}
        writer.add_hparams(hparams, metric_dict)

def mask_array(array, mask):
    int = np.ma.compressed(np.ma.masked_where(mask==0, array) )

    return np.ma.compressed(np.ma.masked_where(mask==0, array) )

class QMixer(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.to(device)
        #params
        self.args = args
        total_state_dim = 0
        if self.args.COMMON_AGENTS_NETWORK:
            self.agents_net = AgentRNN(self.args)
            for agent in self.args.blue_agents:
                #self.agent_nets[agent] = self.agents_net
                total_state_dim += np.prod(env.observation_space(agent).spaces['obs'].shape)               

        else:
            for agent in env.agents:
                self.agents_nets = dict()
                self.agents_nets[agent] = AgentRNN(args)
                total_state_dim += np.prod(env.observation_space(agent).shape)


        self.weightsL1_net = nn.Linear(total_state_dim, self.args.mixer_hidden_dim*self.args.n_blue_agents, device=device)
        self.biasesL1_net = nn.Linear(total_state_dim, self.args.mixer_hidden_dim, device=device)
        
        self.weightsL2_net = nn.Linear(total_state_dim, self.args.mixer_hidden_dim2*self.args.mixer_hidden_dim, device=device)
        self.biasesL2_net = nn.Sequential(
            nn.Linear(total_state_dim, self.args.mixer_hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(self.args.mixer_hidden_dim, self.args.mixer_hidden_dim2, device=device)
        )
        agent_params = list()
        if self.args.COMMON_AGENTS_NETWORK:
            self.net_params = list(self.agents_net.gru.parameters()) + list(self.agents_net.mlp1.parameters()) + list(self.agents_net.mlp2.parameters()) + list(self.weightsL1_net.parameters()) + list(self.biasesL1_net.parameters())  +list(self.weightsL2_net.parameters()) + list(self.biasesL2_net.parameters())
        else:
            for agent_net in self.agents_nets.values():
                agent_params += list(agent_net.gru.parameters()) + list(agent_net.mlp1.parameters()) + list(agent_net.mlp2.parameters())
            self.net_params = agent_params + list(self.weightsL1_net.parameters()) + list(self.biasesL1_net.parameters())  +list(self.weightsL2_net.parameters()) + list(self.biasesL2_net.parameters())

    def get_agent_nets(self, agent):
        if self.args.COMMON_AGENTS_NETWORK:
            return self.agents_net
        else:
            return self.agent_nets[agent]
    
        
    def forward(self, obs_tot,Qin_t):
        weightsL1 = torch.abs(self.weightsL1_net(obs_tot)) # abs: monotonicity constraint
        weightsL1_tensor = weightsL1.unsqueeze(-1).reshape([self.args.BATCH_SIZE, self.args.mixer_hidden_dim, self.args.n_blue_agents])
        biasesL1 = self.biasesL1_net(obs_tot)
        weightsL2 = torch.abs(self.weightsL2_net(obs_tot))
        weightsL2_tensor = weightsL2.unsqueeze(-1).reshape([self.args.BATCH_SIZE, self.args.mixer_hidden_dim2, self.args.mixer_hidden_dim])
        biasesL2 = self.biasesL2_net(obs_tot)
        l1 = torch.matmul(weightsL1_tensor, Qin_t.unsqueeze(-1)).squeeze(-1) + biasesL1
        l1 = nn.ELU(l1).alpha
        Qtot = torch.matmul(weightsL2_tensor, l1.unsqueeze(-1)).squeeze(-1) + biasesL2
        Qtot = Qtot.sum(1)
        
        return Qtot
        
    def get_Q_values(self, agent, obs,hidden_state):
        obs_t = torch.as_tensor(obs['obs'], dtype=torch.float32,device=device)
        q_values, hidden_state = self.get_agent_nets(agent)(obs_t, hidden_state)
        return q_values, hidden_state

    def get_Q_max(self, q_values, all_q_values=None):
        max_q_index = torch.argmax(q_values, dim=1)[0].detach().item()
        max_q = q_values[0,max_q_index]
        if all_q_values != None:
            max_q_index = (all_q_values == max_q.item()).nonzero(as_tuple=True)[1].item()
        return max_q_index, max_q


    def act(self, agent, obs, hidden_state):
        with torch.no_grad():
            q_values, hidden_state = self.get_Q_values(agent, obs, hidden_state)
            #taking only masked q values to choose action to take
            masked_q_val = torch.masked_select(q_values, torch.as_tensor(obs['action_mask'], dtype=torch.bool,device=device)).unsqueeze(0)
            if masked_q_val.numel() == 0:
                return None, hidden_state
            action, _ = self.get_Q_max(masked_q_val, q_values)
            return action, hidden_state
    
        
        
    
class AgentRNN(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.mlp1 = nn.Linear(args.nb_inputs_agent, args.dim_L1_agents_net, device=device)
        self.gru = nn.GRUCell(args.dim_L1_agents_net,args.dim_L2_agents_net, device=device)
        self.mlp2 = nn.Linear(args.dim_L2_agents_net, args.n_actions, device=device)
        self.relu = nn.ReLU()
    def forward(self,obs_t, hidden_state_t):
        in_gru = self.relu(self.mlp1(obs_t))
        hidden_next = self.gru(in_gru.unsqueeze(0), hidden_state_t)
        q_values = self.mlp2(hidden_next)
        return q_values, hidden_next

class buffers:
    def __init__(self, env, args, agents):
        self.observation_prev = dict()
        self.observation = dict()
        self.hidden_state_prev = dict()
        self.hidden_state = dict()
        self.episode_reward = 0.0
        self.nb_transitions = 0
        self.replay_buffer = deque(maxlen=args.BUFFER_SIZE)
        self.rew_buffer = deque([0.0], maxlen=args.REW_BUFFER_SIZE)
        self.loss_buffer = deque([0.0], maxlen=args.REW_BUFFER_SIZE)

        for agent in agents:
            self.observation_prev[agent] = env.observe(agent)
            self.hidden_state_prev[agent] = torch.zeros(args.dim_L2_agents_net, device=device).unsqueeze(0)
            self.hidden_state[agent] = torch.zeros(args.dim_L2_agents_net, device=device).unsqueeze(0)
        

class runner_QMix:
    def __init__(self, env, args):
        self.args = args
        self.env = env
        self.blue_team_buffers = buffers(self.env, self.args, self.args.blue_agents)
        self.opposing_team_buffers = buffers(self.env, self.args, self.args.opposing_agents)
        self.online_net = QMixer(self.env, self.args)
        self.target_net = copy.deepcopy(self.online_net) #QMixer(self.env, self.args)

        self.blue_team_buffers = buffers(self.env, self.args, self.args.blue_agents)
        self.opposing_team_buffers = buffers(self.env, self.args, self.args.opposing_agents)


        #display model graphs in tensorboard
        self.writer = SummaryWriter()
        args.log_params(self.writer)
        self.writer.add_graph(self.online_net.get_agent_nets(self.args.blue_agents[0]),(torch.empty((self.args.observations_dim),device=device), torch.empty((1, self.args.dim_L2_agents_net),device=device)) )
        self.writer.add_graph(self.online_net, (torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents*self.args.observations_dim), device=device)
, torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents), device=device)))
        self.sync_networks()
        self.optimizer = torch.optim.Adam(self.online_net.net_params, lr = self.args.LEARNING_RATE)
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True, patience =15000)  #patience, min lr... Parameters still to find
    def random_action(self, agent, obs):
        if all(element == 0 for element in obs['action_mask']):
            return None
        return random.choice(mask_array(range(self.env.action_space(agent).n), obs['action_mask']))
    def update_buffer(self, agent, action):
        if action == None:
            action = -1
            reward = 0
            done = True
        else:
            reward = self.env._cumulative_rewards[agent]
            done = self.env.dones[agent]
            
        if self.is_opposing_team(agent):
            
            self.opposing_team_buffers.observation[agent] = self.env.observe(agent)
            self.opposing_team_buffers.episode_reward += reward
            
            #self.transition[agent] = (self.opposing_team_buffers.observation_prev[agent], action,reward,done,self.opposing_team_buffers.observation[agent],self.opposing_team_buffers.hidden_state_prev[agent], self.opposing_team_buffers.hidden_state[agent])
            self.opposing_team_buffers.observation_prev[agent] = self.opposing_team_buffers.observation[agent]
            #self.opposing_team_buffers.hidden_state_prev[agent] = self.opposing_team_buffers.hidden_state[agent]
        else:
            
            self.blue_team_buffers.observation[agent] = self.env.observe(agent)
            
            self.blue_team_buffers.episode_reward += reward
            self.transition[agent] = (self.blue_team_buffers.observation_prev[agent], action,reward,done,self.blue_team_buffers.observation[agent],self.blue_team_buffers.hidden_state_prev[agent], self.blue_team_buffers.hidden_state[agent])
            self.blue_team_buffers.observation_prev[agent] = self.blue_team_buffers.observation[agent]
            self.blue_team_buffers.hidden_state_prev[agent] = self.blue_team_buffers.hidden_state[agent]
            
        
    def step_buffer(self):
        self.blue_team_buffers.nb_transitions += 1
        self.opposing_team_buffers.nb_transitions +=1
    def reset_buffer(self, agent):
        if self.is_opposing_team(agent):
            self.opposing_team_buffers.observation_prev[agent] = self.env.observe(agent)
            #self.opposing_team_buffers.hidden_state_prev[agent] = torch.zeros(self.args.dim_L2_agents_net, device=device).unsqueeze(0)

        else:
            self.blue_team_buffers.observation_prev[agent] = self.env.observe(agent)
            self.blue_team_buffers.hidden_state_prev[agent] = torch.zeros(self.args.dim_L2_agents_net, device=device).unsqueeze(0)

    def reset_buffers(self):
            self.opposing_team_buffers.episode_reward = 0
            self.opposing_team_buffers.nb_transitions = 0
            
            self.blue_team_buffers.episode_reward = 0
            self.blue_team_buffers.nb_transitions = 0
            for agent in self.args.all_agents:
                self.reset_buffer(agent)
    def is_opposing_team(self, agent):
        if re.match(rf'^{self.args.OPPOSING_TEAM}',agent):
            return True
        return False

    def adversary_tactic(self, agent, obs):
        #random tactic
        if self.args.ADVERSARY_TACTIC == 'random':
            return self.random_action(agent, obs)
        elif   self.args.ADVERSARY_TACTIC == 'passive':
            return 0

    def sync_networks(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        if self.args.COMMON_AGENTS_NETWORK:
            self.target_net.agents_net.load_state_dict(self.online_net.agents_net.state_dict())
        else:
            for agent in self.args.blue_agents:
                self.target_net.agents_nets[agent].load_state_dict(self.online_net.agents_nets[agent].state_dict())

    def visualize(self):
          #evaluating the actual policy of the agents
        if self.args.VISUALIZE:
            self.args.GREEDY = False
            self.env.render()

    def save_model(self, train_step):  #taken from https://github.com/koenboeckx/qmix/blob/main/qmix.py to save learnt model
        num = str(train_step // self.args.SAVE_CYCLE)
        if self.args.RUN_NAME != '':
            dirname = self.args.MODEL_DIR + '/' + self.args.RUN_NAME + '/' +datetime.now().strftime("%d%H%M%b%Y") + f'step_{train_step}'
        else:
            dirname = self.args.MODEL_DIR + '/' +datetime.now().strftime("%d%H%M%b%Y")

        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(self.online_net.state_dict(), dirname + '/' + '/qmix_net_params.pt')
        if self.args.COMMON_AGENTS_NETWORK:
            torch.save(self.online_net.agents_net.state_dict(),  dirname + '/agents_net_params.pt')
        else:
            for agent in self.args.blue_agents:
                torch.save(self.online_net.agents_nets.state_dict(),  dirname + '/agent_nets_params/'  + agent + '.pt')

    def load_model(self, dir):
        mixer_model = dir + '/qmix_net_params.pt'
        self.online_net.load_state_dict(torch.load(mixer_model))
        if self.args.COMMON_AGENTS_NETWORK:
            agent_model = dir + '/agents_net_params.pt'
            self.online_net.agents_net.load_state_dict(torch.load(agent_model))
        else:
            for agent in self.args.blue_agents:
                agent_model = dir + '/agent_nets_params/' + agent +'.pt'
                self.online_net.agents_nets[agent].load_state_dict(torch.load(agent_model))

    def run(self):
        
        #Init replay buffer
        
        self.env.reset()
        for _ in range(self.args.MIN_BUFFER_LENGTH):
            
            self.transition = dict() #to store the transition 
            self.step_buffer() #count the number of transitions per episode
            for agent in self.env.agent_iter(max_iter=len(self.env.agents)):

                if self.is_opposing_team(agent):                   
                    self.opposing_team_buffers.observation[agent], _, done, _ = self.env.last()
                    action = self.adversary_tactic(agent, self.opposing_team_buffers.observation[agent])
                    if done:
                        action = None
                    
                else:
                    self.blue_team_buffers.observation[agent], _, done, _ = self.env.last()
                    action = self.random_action(agent, self.blue_team_buffers.observation[agent])
                    if done:
                        action = None
                    _, self.blue_team_buffers.hidden_state[agent] = self.online_net.act(agent, self.blue_team_buffers.observation[agent], self.blue_team_buffers.hidden_state_prev[agent])
                
                self.env.step(action)
                self.update_buffer(agent, action)
                self.visualize()

                #update buffer here
                
            self.blue_team_buffers.replay_buffer.append(self.transition)
            
            
            if all(x == True for x in self.env.dones.values()):  #if all agents are done, episode is done, -> reset the environment

                self.blue_team_buffers.episode_reward = self.blue_team_buffers.episode_reward/(self.args.n_blue_agents) #
                self.blue_team_buffers.rew_buffer.append(self.blue_team_buffers.episode_reward)
                self.env.reset()
                
                self.reset_buffers()
        # trainingoptim

        self.env.reset()
        self.reset_buffers()

        for step in itertools.count():

            if step > self.args.VISUALIZE_AFTER:
                self.args.VISUALIZE = True
            epsilon = np.interp(step, [0, self.args.EPSILON_DECAY], [self.args.EPSILON_START, self.args.EPSILON_END])
            rnd_sample = random.random()

            self.transition = dict()
            self.step_buffer()
            for agent in self.env.agent_iter(max_iter=len(self.env.agents)):
                
                if self.is_opposing_team(agent):
                    self.opposing_team_buffers.observation[agent], _, done, _ = self.env.last()
                    action = self.adversary_tactic(agent, self.opposing_team_buffers.observation[agent])
                    if done:
                        action = None
                    
                    
                else:
                    self.blue_team_buffers.observation[agent], _, done, _ = self.env.last()
                    action, self.blue_team_buffers.hidden_state[agent] = self.online_net.act(agent, self.blue_team_buffers.observation[agent], self.blue_team_buffers.hidden_state_prev[agent])
                    if rnd_sample <= epsilon and self.args.GREEDY:
                        action = self.random_action(agent, self.blue_team_buffers.observation[agent])
                    if done:
                        action = None
                    
                self.env.step(action)
                self.update_buffer(agent, action)
                self.visualize()
            
            self.blue_team_buffers.replay_buffer.append(self.transition)
           
            if all(x == True for x in self.env.dones.values()):  #if all agents are done, episode is done, -> reset the environment
        
                self.blue_team_buffers.episode_reward = self.blue_team_buffers.episode_reward/(self.args.n_blue_agents)
                self.blue_team_buffers.rew_buffer.append(self.blue_team_buffers.episode_reward)
                self.env.reset()
                self.writer.add_scalar("Reward", self.blue_team_buffers.episode_reward,step  )
                self.reset_buffers()


            

            transitions = random.sample(self.blue_team_buffers.replay_buffer, self.args.BATCH_SIZE)
            obses_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents*self.args.observations_dim), device=device)
            actions_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents), device=device)
            Q_ins_target_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents), device=device)
            Q_action_online_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents), device=device)
            rewards_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents), device=device)
            dones_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents), device=device)
            new_obses_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents*self.args.observations_dim), device=device)
            transition_nb = 0
            for t in transitions:
                
                agent_nb = 0
                for agent in self.args.blue_agents:
                    ###
                    if agent in t: #check if agent was not done during the transition

                        obses_t[transition_nb][self.args.observations_dim*agent_nb:(self.args.observations_dim*(agent_nb+1))] = torch.as_tensor(t[agent][0]['obs'], dtype=torch.float32, device=device)
                        actions_t[transition_nb][agent_nb] = t[agent][1]
                        rewards_t[transition_nb][agent_nb] = t[agent][2]
                        dones_t[transition_nb][agent_nb] = t[agent][3]
                        new_obses_t[transition_nb][self.args.observations_dim*agent_nb:(self.args.observations_dim*(agent_nb+1))] = torch.as_tensor(t[agent][4]['obs'], dtype=torch.float32, device=device) #.detach()
                        if t[agent][1] == -1:
                            Q_action_online_t[transition_nb][agent_nb] = 0
                        else:
                            Q_action_online_t[transition_nb][agent_nb] = torch.gather(self.online_net.get_Q_values(agent, t[agent][0], t[agent][5])[0].squeeze(0), 0,torch.tensor([t[agent][1]], device=device))
                        Q_ins_target_t[transition_nb][agent_nb] = self.target_net.get_Q_max(self.target_net.get_Q_values(agent, t[agent][4], t[agent][6])[0])[1]#.detach()
                        
                        agent_nb += 1
                    else: #in case the agent was done during the transition
                        obses_t[transition_nb][self.args.observations_dim*agent_nb:(self.args.observations_dim*(agent_nb+1))] = torch.zeros((self.args.observations_dim), dtype=torch.float32, device=device)
                        actions_t[transition_nb][agent_nb] = -1
                        rewards_t[transition_nb][agent_nb] = 0
                        dones_t[transition_nb][agent_nb] = True
                        new_obses_t[transition_nb][self.args.observations_dim*agent_nb:(self.args.observations_dim*(agent_nb+1))] = torch.zeros((self.args.observations_dim), dtype=torch.float32, device=device)
                        
                        Q_action_online_t[transition_nb][agent_nb] = 0
                        Q_ins_target_t[transition_nb][agent_nb] = 0
                        
                        agent_nb += 1


                transition_nb += 1

            #compute reward for all agents
            rewards_t = rewards_t.mean(1)
            #if one agent is done all are
            dones_t = dones_t.sum(1)
            dones_t = dones_t > 0
            # targets
            Qtot_max_target = self.target_net.forward(new_obses_t, Q_ins_target_t).detach()
            Qtot_online = self.online_net.forward(obses_t, Q_action_online_t)
            y_tot = rewards_t + self.args.GAMMA*(1 + (-1)*dones_t)*Qtot_max_target

        ########### busy
            # loss 
            error = y_tot + (-1)*Qtot_online
            
            loss = error**2
            mean_loss = torch.mean(loss)
            loss = loss.sum()
            self.blue_team_buffers.loss_buffer.append(mean_loss.item())  # detach ?????????????????
            self.writer.add_scalar("Loss", mean_loss.item(), step)
            


            # gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #self.scheduler.step(np.mean(self.loss_buffer))
            #update target network

            if step % args.SYNC_TARGET_FRAMES == 0:
                self.sync_networks()

            #save model

            if step % self.args.SAVE_CYCLE == 0:
                self.save_model(step)

            #logging
            if step % self.args.REW_BUFFER_SIZE == 0:
                print('\n Step', step )
                print('Avg Episode Reward /agent ', np.mean(self.blue_team_buffers.rew_buffer))
                print('Avg Loss over a batch', np.mean(self.blue_team_buffers.loss_buffer))
        self.writer.close() 

    
        ###

    def eval(self, params_directory):

        self.env.reset()
        self.reset_buffers()
        self.load_model(params_directory)
        self.transition = dict()
        for step in itertools.count():
            
            self.step_buffer()
            for agent in self.env.agent_iter(max_iter=len(self.env.agents)):
                
                if self.is_opposing_team(agent):
                    self.opposing_team_buffers.observation[agent], _, done, _ = self.env.last()
                    action = self.adversary_tactic(agent, self.opposing_team_buffers.observation[agent])
                    if done:
                        action = None
                    
                else:
                    self.blue_team_buffers.observation[agent], _, done, _ = self.env.last()
                    action, self.blue_team_buffers.hidden_state[agent] = self.online_net.act(agent, self.blue_team_buffers.observation[agent], self.blue_team_buffers.hidden_state_prev[agent])
                    if done:
                        action = None
                    
                self.env.step(action)
                print(f'{agent} : {action}')
                self.env.render()
                #time.sleep(self.args.WAIT_BETWEEN_STEPS)
                self.update_buffer(agent, action)

            if all(x == True for x in self.env.dones.values()):  #if all agents are done, episode is done, -> reset the environment
        
                self.blue_team_buffers.episode_reward = self.blue_team_buffers.episode_reward/(self.args.n_blue_agents)
                self.blue_team_buffers.rew_buffer.append(self.blue_team_buffers.episode_reward)
                self.env.reset()
                print(f'Episode reward /agent: {self.blue_team_buffers.episode_reward}')
                self.reset_buffers()




if __name__ == "__main__":
    env = defense_v0.env(terrain=TERRAIN, max_cycles=EPISODE_MAX_LENGTH, max_distance=MAX_DISTANCE )
    env.reset()
    args = Args(env)
    runner = runner_QMix(env, args)
    if len(sys.argv) == 1:
        #       setting up TensorBoard
        runner.run()
    else:
        runner.eval(sys.argv[1])

