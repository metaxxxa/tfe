from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from pettingzoo.mpe import simple_spread_v2
from collections import deque
import itertools
import random
from torch.utils.tensorboard import SummaryWriter
import copy
import os
import sys
import time
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
            
        self.BUFFER_SIZE = 200
        self.REW_BUFFER_SIZE = 100
        self.LEARNING_RATE = 1e-4
        self.MIN_BUFFER_LENGTH = 300
        self.BATCH_SIZE = 32
        self.GAMMA = 0.95
        self.EPSILON_START = 1
        self.EPSILON_END = 0.01
        self.EPSILON_DECAY = 200000
        self.SYNC_TARGET_FRAMES = 200
        #visualization parameters
        self.VISUALIZE_WHEN_LEARNED = True
        self.VISUALIZE_AFTER = 500000
        self.VISUALIZE = False
        self.WAIT_BETWEEN_STEPS = 0.1
        self.GREEDY = True
        self.SAVE_CYCLE = 5000
        self.MODEL_DIR = 'simple_spread_QMix'
        self.RUN_NAME = ''
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
        self.agents = env.agents
        agent = self.agents[0]
        self.nb_inputs_agent = np.prod(env.observation_space(agent).shape)
        self.observations_dim = env.observation_space(agent).shape[0]
        self.n_actions = env.action_space(agent).n
    def log_params(self, writer):
        hparams = {'Learning rate': self.LEARNING_RATE, 'Batch size': self.BATCH_SIZE, 'Buffer size': self.BUFFER_SIZE, 'Min buffer length': self.MIN_BUFFER_LENGTH, '\gamma': self.GAMMA, 'Epsilon range': f'{self.EPSILON_START} - {self.EPSILON_END}', 'Epsilon decay': self.EPSILON_DECAY, 'Synchronisation rate': self.SYNC_TARGET_FRAMES, 'Timestamp': int(datetime.timestamp(datetime.now())), 'Common agent network': int(self.COMMON_AGENTS_NETWORK)}
        metric_dict = { 'hparam/dim L1 agent net': self.dim_L1_agents_net, 'hparam/dim L2 agent net': self.dim_L2_agents_net, 'hparam/mixer hidden dim 1': self.mixer_hidden_dim, 'hparam/mixer hidden dim 2': self.mixer_hidden_dim2, 'envparam/nb_agents': self.n_agents}
        writer.add_hparams(hparams, metric_dict)

class QMixer(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.to(device)
        #params
        self.args = args
        total_state_dim = 0
        if self.args.COMMON_AGENTS_NETWORK:
            self.agents_net = AgentRNN(self.args)
            for agent in env.agents:
                #self.agent_nets[agent] = self.agents_net
                total_state_dim += np.prod(env.observation_space(agent).shape)                

        else:
            for agent in env.agents:
                self.agents_nets = dict()
                self.agents_nets[agent] = AgentRNN(args)
                total_state_dim += np.prod(env.observation_space(agent).shape)


        self.weightsL1_net = nn.Linear(total_state_dim, self.args.mixer_hidden_dim*self.args.n_agents, device=device)
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
        
    def get_Q_values(self, agent, obs,hidden_state):
        obs_t = torch.as_tensor(obs, dtype=torch.float32,device=device)
        q_values, hidden_state = self.get_agent_nets(agent)(obs_t, hidden_state)
        return q_values, hidden_state

    def get_Q_max(self, q_values):
        max_q_index = torch.argmax(q_values, dim=1)[0]
        max_q_index = max_q_index.detach().item()
        max_q = q_values[0,max_q_index]
        return max_q_index, max_q

    def act(self, agent, obs, hidden_state):
        with torch.no_grad():
            q_values, hidden_state = self.get_Q_values(agent, obs, hidden_state)
            action, _ = self.get_Q_max(q_values)
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



class runner_QMix:
    def __init__(self, env, args):
        self.args = args
        self.env = env
        
        self.replay_buffer = deque(maxlen=self.args.BUFFER_SIZE)
        self.rew_buffer = deque([0.0], maxlen=args.REW_BUFFER_SIZE)
        self.loss_buffer = deque([0.0], maxlen=args.REW_BUFFER_SIZE)

        self.online_net = QMixer(self.env, self.args)
        self.target_net = copy.deepcopy(self.online_net) #QMixer(self.env, self.args)

        self.sync_networks()
        self.optimizer = torch.optim.Adam(self.online_net.net_params, lr = self.args.LEARNING_RATE)
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True, patience =15000)  #patience, min lr... Parameters still to find
    def sync_networks(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        if self.args.COMMON_AGENTS_NETWORK:
            self.target_net.agents_net.load_state_dict(self.online_net.agents_net.state_dict())
        else:
            for agent in self.args.agents:
                self.target_net.agents_nets[agent].load_state_dict(self.online_net.agents_nets[agent].state_dict())

    def visualize(self):
          #evaluating the actual policy of the agents
        if self.args.VISUALIZE:
            self.args.GREEDY = False
            self.env.render()

    def save_model(self, train_step):  #taken from https://github.com/koenboeckx/qmix/blob/main/qmix.py to save learnt model
        num = str(train_step // self.args.SAVE_CYCLE)
        if self.args.RUN_NAME != '':
            dirname = self.args.MODEL_DIR + '/' + self.args.RUN_NAME + '/' +datetime.now().strftime("%d%H%M%b%Y")
        else:
            dirname = self.args.MODEL_DIR + '/' +datetime.now().strftime("%d%H%M%b%Y")

        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(self.online_net.state_dict(), dirname + '/' + '/qmix_net_params.pt')
        if self.args.COMMON_AGENTS_NETWORK:
            torch.save(self.online_net.agents_net.state_dict(),  dirname + '/agents_net_params.pt')
        else:
            for agent in self.env.agents:
                torch.save(self.online_net.agents_nets.state_dict(),  dirname + '/agent_nets_params/'  + agent + '.pt')

    def load_model(self, dir):
        mixer_model = dir + '/qmix_net_params.pt'
        self.online_net.load_state_dict(torch.load(mixer_model))
        if self.args.COMMON_AGENTS_NETWORK:
            agent_model = dir + '/agents_net_params.pt'
            self.online_net.agents_net.load_state_dict(torch.load(agent_model))
        else:
            for agent in self.env.agents:
                agent_model = dir + '/agent_nets_params/' + agent +'.pt'
                self.online_net.agents_nets[agent].load_state_dict(torch.load(agent_model))

    def run(self):
        
        #Init replay buffer
        
        self.env.reset()
        one_agent_done = 0

        observation_prev = dict()
        observation = dict()
        hidden_state_prev = dict()
        hidden_state = dict()
        episode_reward = 0.0
        nb_transitions = 0
        for agent in self.env.agents:
            observation_prev[agent], _, _, _ = self.env.last()
            hidden_state_prev[agent] = torch.zeros(self.args.dim_L2_agents_net, device=device).unsqueeze(0)
        for _ in range(args.MIN_BUFFER_LENGTH):
            
            transition = dict()
            for agent in self.env.agent_iter(max_iter=len(self.env.agents)):
                action = self.env.action_space(agent).sample()
                _, hidden_state[agent] = self.online_net.act(agent, observation_prev[agent], hidden_state_prev[agent])
                if one_agent_done:
                    self.env.step(None)
                    self.visualize()
                else:
                    self.env.step(action)
                    self.visualize()
                observation[agent], reward, done, info = self.env.last()
                episode_reward += reward
                nb_transitions += 1
                transition[agent] = (observation_prev[agent], action,reward,done,observation[agent],hidden_state_prev[agent], hidden_state[agent])
                observation_prev[agent] = observation[agent]
                hidden_state_prev[agent] = hidden_state[agent]
                if done:
                    one_agent_done = 1 #if one agent is done, all have to stop
            if one_agent_done:
                episode_reward = episode_reward/(self.args.n_agents*nb_transitions)
                self.rew_buffer.append(episode_reward)
                self.env.reset()
                episode_reward = 0.0
                nb_transitions = 0
                one_agent_done = 0
                for agent in self.env.agents:
                    observation_prev[agent], _, _, _ = self.env.last()
                    hidden_state_prev[agent] = torch.zeros(self.args.dim_L2_agents_net, device=device).unsqueeze(0)
            
            self.replay_buffer.append(transition)

        # trainingoptim

        hidden_state_prev = dict()
        hidden_state = dict()
        self.env.reset()
        episode_reward = 0.0
        nb_transitions = 0
        for agent in self.env.agents:
            observation_prev[agent], _, _, _ = self.env.last()
            hidden_state_prev[agent] = torch.zeros(self.args.dim_L2_agents_net, device=device).unsqueeze(0)
        for step in itertools.count():
            if step > self.args.VISUALIZE_AFTER:
                self.args.VISUALIZE = True
            epsilon = np.interp(step, [0, args.EPSILON_DECAY], [args.EPSILON_START, args.EPSILON_END])
            rnd_sample = random.random()
            transition = dict()
            for agent in self.env.agent_iter(max_iter=len(self.env.agents)):

                action, hidden_state[agent] = self.online_net.act(agent, observation_prev[agent], hidden_state_prev[agent])
                if rnd_sample <= epsilon and self.args.GREEDY:
                    action = self.env.action_space(agent).sample()
                if one_agent_done:
                    self.env.step(None)
                    self.visualize()
                else:
                    self.env.step(action)
                    self.visualize()
                observation[agent], reward, done, info = self.env.last()
                nb_transitions += 1
                episode_reward += reward
                transition[agent] = (observation_prev[agent], action,reward,done,observation[agent], hidden_state_prev[agent], hidden_state[agent])
                observation_prev[agent] = observation[agent]
                hidden_state_prev[agent] = hidden_state[agent]
                if done:
                    one_agent_done = 1 #if one agent is done, all have to stop
            if one_agent_done:
                episode_reward = episode_reward/(self.args.n_agents*nb_transitions)
                self.rew_buffer.append(episode_reward)
                self.env.reset()
                writer.add_scalar("Reward", episode_reward,step  )
                episode_reward = 0.0
                nb_transitions = 0
                one_agent_done = 0
                for agent in self.env.agents:
                    observation_prev[agent], _, _, _ = self.env.last()
                    hidden_state_prev[agent] = torch.zeros(self.args.dim_L2_agents_net, device=device).unsqueeze(0)
            
            self.replay_buffer.append(transition)

            transitions = random.sample(self.replay_buffer, args.BATCH_SIZE)
            obses_t = torch.empty((self.args.BATCH_SIZE,self.args.n_agents*self.args.observations_dim), device=device)
            actions_t = torch.empty((self.args.BATCH_SIZE,self.args.n_agents), device=device)
            Q_ins_target_t = torch.empty((self.args.BATCH_SIZE,self.args.n_agents), device=device)
            Q_action_online_t = torch.empty((self.args.BATCH_SIZE,self.args.n_agents), device=device)
            rewards_t = torch.empty((self.args.BATCH_SIZE,self.args.n_agents), device=device)
            dones_t = torch.empty((self.args.BATCH_SIZE,self.args.n_agents), device=device)
            new_obses_t = torch.empty((self.args.BATCH_SIZE,self.args.n_agents*self.args.observations_dim), device=device)
            transition_nb = 0
            for t in transitions:
                
                agent_nb = 0
                for agent in self.env.agents:
                    obses_t[transition_nb][self.args.observations_dim*agent_nb:(self.args.observations_dim*(agent_nb+1))] = torch.as_tensor(t[agent][0], dtype=torch.float32, device=device)
                    actions_t[transition_nb][agent_nb] = t[agent][1]
                    rewards_t[transition_nb][agent_nb] = t[agent][2]
                    dones_t[transition_nb][agent_nb] = t[agent][3]
                    new_obses_t[transition_nb][self.args.observations_dim*agent_nb:(self.args.observations_dim*(agent_nb+1))] = torch.as_tensor(t[agent][4], dtype=torch.float32, device=device) #.detach()
                    
                    Q_action_online_t[transition_nb][agent_nb] = torch.gather(self.online_net.get_Q_values(agent, t[agent][0], t[agent][5])[0].squeeze(0), 0,torch.tensor([t[agent][1]], device=device))
                    Q_ins_target_t[transition_nb][agent_nb] = self.target_net.get_Q_max(self.target_net.get_Q_values(agent, t[agent][4], t[agent][6])[0])[1]#.detach()
                     
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
            self.loss_buffer.append(mean_loss.item())  # detach ?????????????????
            writer.add_scalar("Loss", mean_loss.item(), step)
            


            # gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #self.scheduler.step(np.mean(self.loss_buffer))
            #update target network

            if step % args.SYNC_TARGET_FRAMES == 0:
                self.sync_networks()

            #save model

            if step % self.args.SAVE_CYCLE:
                self.save_model(step)

            #logging
            if step % self.args.REW_BUFFER_SIZE == 0:
                print('\n Step', step )
                print('Avg Reward /agent /transition', np.mean(self.rew_buffer))
                print('Avg Loss over a batch', np.mean(self.loss_buffer))
        writer.close() 

    def eval(self, params_directory):

        self.load_model(params_directory)
        observation_prev = dict()
        observation = dict()
        hidden_state_prev = dict()
        hidden_state = dict()
        self.env.reset()
        episode_reward = 0.0
        nb_transitions = 0
        one_agent_done = 0
        for agent in self.env.agents:
            observation_prev[agent], _, _, _ = self.env.last()
            hidden_state_prev[agent] = torch.zeros(self.args.dim_L2_agents_net, device=device).unsqueeze(0)
        for step in itertools.count():
            for agent in self.env.agent_iter(max_iter=len(self.env.agents)):

                action, hidden_state[agent] = self.online_net.act(agent, observation_prev[agent], hidden_state_prev[agent])
                if one_agent_done:
                    self.env.step(None)
                    self.env.render()
                    time.sleep(self.args.WAIT_BETWEEN_STEPS)
                else:
                    self.env.step(action)
                    self.env.render()
                    time.sleep(self.args.WAIT_BETWEEN_STEPS)
                observation[agent], reward, done, info = self.env.last()
                nb_transitions += 1
                episode_reward += reward
                observation_prev[agent] = observation[agent]
                hidden_state_prev[agent] = hidden_state[agent]
                if done:
                    one_agent_done = 1 #if one agent is done, all have to stop
            if one_agent_done:
                episode_reward = episode_reward/(self.args.n_agents*nb_transitions)
                print('Mean episode reward /agent /transition : {episode_reward}')
                self.env.reset()
                episode_reward = 0.0
                nb_transitions = 0
                one_agent_done = 0
                for agent in self.env.agents:
                    observation_prev[agent], _, _, _ = self.env.last()
                    hidden_state_prev[agent] = torch.zeros(self.args.dim_L2_agents_net, device=device).unsqueeze(0)
            

        ###
if __name__ == "__main__":
    env = simple_spread_v2.env(N=1, local_ratio=0.5, max_cycles=25, continuous_actions=False)
    env.reset()
    args = Args(env)
    args.log_params(writer)
    runner = runner_QMix(env, args)
    if len(sys.argv) == 1:
        runner.run()
    else:
        runner.eval(sys.argv[1])

