
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import itertools
import random
import copy
import os
import sys
import time
import re
import getopt
import pickle

#importing the defense environment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)
from env import defense_v0

from Utils import helper
from Utils.helper import Buffers, Params, Metrics, Constants, mask_array, get_device
from Utils.params import DQNArgs as Args





device = get_device() #get cuda if available
#environment constants
constants = Constants()
TERRAIN = 'benchmark_10x10_1v1'
TERRAIN_SIZE = 10
MODEL_DIR = 'defense_params_dqn'
RUN_NAME = 'benchmarking'
ADVERSARY_TACTIC = 'random'
ENV_SIZE = 10 #todo : calculate for dqn class

class DQN(nn.Module):
    def __init__(self, env, args, observation_space_shape, action_space_n):
        super().__init__()
        if args.CONVOLUTIONAL_INPUT:
            outconv1 = int((ENV_SIZE - args.DILATION*(args.KERNEL_SIZE-1) + 2*args.PADDING - 1)/args.STRIDE + 1)
            outmaxPool = int((outconv1 + 2*args.PADDING_POOL - args.DILATION*(args.KERNEL_SIZE_POOL - 1) -1)/args.STRIDE + 1) #see full formula on pytorch's documentation website
            outconv2 = int((outmaxPool - args.DILATION2*(args.KERNEL_SIZE2-1) + 2*args.PADDING2 - 1)/args.STRIDE2 + 1)
            outmaxPool2 = int((outconv2 + 2*args.PADDING_POOL2 - args.DILATION2*(args.KERNEL_SIZE_POOL2 - 1) -1)/args.STRIDE2 + 1) 
            self.net = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=args.CONV_OUT_CHANNELS, kernel_size=(args.KERNEL_SIZE, args.KERNEL_SIZE), padding=args.PADDING, stride = args.STRIDE, dilation = args.DILATION),
                nn.ReLU(),
                nn.MaxPool2d(args.KERNEL_SIZE_POOL, args.STRIDE, dilation=args.DILATION, padding=args.PADDING_POOL),
                nn.Conv2d(in_channels=args.CONV_OUT_CHANNELS, out_channels=args.CONV_OUT_CHANNELS2, kernel_size=(args.KERNEL_SIZE2, args.KERNEL_SIZE2), padding=args.PADDING2, stride = args.STRIDE2, dilation = args.DILATION2),
                nn.ReLU(),
                nn.MaxPool2d(args.KERNEL_SIZE_POOL2, args.STRIDE2, dilation=args.DILATION2, padding=args.PADDING_POOL2),
                
                nn.Flatten(),
                nn.Linear(args.CONV_OUT_CHANNELS2*outmaxPool2**2, args.hidden_layer1_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_layer1_dim,args.hidden_layer2_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_layer2_dim, action_space_n)
            ).to(device)

        else:
            self.net = nn.Sequential(
                nn.Linear(np.prod(observation_space_shape) , args.hidden_layer1_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_layer1_dim,args.hidden_layer2_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_layer2_dim, action_space_n)
            ).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = args.LEARNING_RATE)
        
    def forward(self, obs):
        return self.net(torch.as_tensor(obs, dtype=torch.float32,device=device))
        
    def get_Q_values(self, obs):
        obs_t = torch.as_tensor(obs['obs'], dtype=torch.float32,device=device)
        q_values = self.net(obs_t.unsqueeze(0))
        return q_values

    def get_Q_max(self, q_values, obs, all_q_values=None):
        if len(q_values) == 0:
            return -1, torch.tensor([0],device=device)
        max_q_index = torch.argmax(q_values, dim=-1).detach().item()
        max_q = q_values[max_q_index]
        if all_q_values != None:
            indexes = ((all_q_values == max_q.item()).cpu() * (obs['action_mask'] == 1)).nonzero(as_tuple=True)
            max_q_index = indexes[-1][-1].item()
        return max_q_index, max_q


    def act(self, obs):
        with torch.no_grad():
            q_values = self.get_Q_values(obs)
            #taking only masked q values to choose action to take
            masked_q_val = torch.masked_select(q_values, torch.as_tensor(obs['action_mask'], dtype=torch.bool,device=device))
            if masked_q_val.numel() == 0:
                return None
            action, _ = self.get_Q_max(masked_q_val, obs, q_values)
            return action
    
        

class Runner:
    def __init__(self, env, args):
        self.args = args
        self.env = env
        if self.args.ENV_FOLDER != '':
            self.args.CHANGE_ENV = True
            self.ter_array = []
            self.ter_ind = 0
            for filename in os.listdir(f'env/terrains/{self.args.ENV_FOLDER}'):
                self.ter_array.append(f'{self.args.ENV_FOLDER}/{filename[0:-4]}')
            self.env = defense_v0.env(terrain=self.ter_array[self.ter_ind] , max_cycles=constants.EPISODE_MAX_LENGTH, max_distance=constants.MAX_DISTANCE )
            self.env.reset()

        self.env_size = 10  #try to get from environment
        self.blue_team_buffers = Buffers(self.env, self.args, self.args.blue_agents, device)
        self.opposing_team_buffers = Buffers(self.env, self.args, self.args.red_agents, device)
        self.online_nets = {}
        self.target_nets = {}
        self.online_nets_params = list()
        for agent in self.args.blue_agents:
            self.online_nets[agent] = DQN(self.env, self.args, self.env.observation_space(agent).spaces['obs'].shape, self.env.action_space(agent).n)

        self.target_nets = copy.deepcopy(self.online_nets)
        if self.args.TENSORBOARD:
            #setting up TensorBoard
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter()
            args.log_params(TERRAIN, self.writer)
            if self.args.EXTRA_PLOTS:
                from Utils.plotter import plot_nn
                plot_nn(self.online_nets[self.args.blue_agents[0]], 'DQN') #to visualize the neural network
                
                    #display model graphs in tensorboard
                if self.args.CONVOLUTIONAL_INPUT:
                    pass #to implement if time
                else:
                    pass #self.writer.add_graph(self.online_nets[self.args.blue_agents[0]],torch.empty((self.args.observations_dim),device=device) )
        
        self.sync_networks()
        
        if self.args.MODEL_TO_LOAD != '':
            self.load_model(self.args.MODEL_TO_LOAD)

        if self.args.ADVERSARY_TACTIC == 'dqn':
            self.adversary_nets = {}
            for agent in self.args.red_agents:
                self.adversary_nets[agent] = DQN(self.env, self.args, self.env.observation_space(agent).spaces['obs'].shape, self.env.action_space(agent).n)
            self.load_model(self.args.ADVERSARY_MODEL, True)

        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True, patience =15000)  #patience, min lr... Parameters still to find
    
    def change_terrain(self):
        if self.args.CHANGE_ENV:
            self.ter_ind += 1
            self.env = defense_v0.env(terrain=self.ter_array[self.ter_ind] , max_cycles=constants.EPISODE_MAX_LENGTH, max_distance=constants.MAX_DISTANCE )
                        
            if self.ter_ind >= (len(self.ter_array) - 1):
                self.ter_ind = 0
        self.env.reset()

    def observe(self, agent):
        observation = copy.deepcopy(self.env.observe(agent))
        
        if self.args.CONVOLUTIONAL_INPUT:
            observation['obs'] = helper.obs_to_convInput(observation, self.env_size, self.env.env.max_num_agents)
        return observation
    
    def last(self):
        observation, reward, done, info = self.env.last()
        if self.args.CONVOLUTIONAL_INPUT:
            return self.observe(self.env.agent_selection), reward, done, info
        else:
            return observation, reward, done, info

    def random_action(self, agent, obs):
        if all(element == 0 for element in obs['action_mask']):
            return None
        return random.choice(mask_array(range(self.env.action_space(agent).n), obs['action_mask']))
    def update_buffer(self):
        for agent in self.env.agents:
            
            if self.is_opposing_team(agent):
                pass
                # if self.opposing_team_buffers.action[agent] == None:
                #     self.opposing_team_buffers.action[agent] = -1
                #     reward = 0
                #     done = True
                # else:
                #     reward = self.env._cumulative_rewards[agent]
                #     done = self.env.dones[agent]
                # self.opposing_team_buffers.observation_next[agent] = self.observe(agent)
                # self.opposing_team_buffers.episode_reward += reward
                # #store transition ?
                # self.opposing_team_buffers.observation[agent] = self.opposing_team_buffers.observation_next[agent]
            else:
                if self.blue_team_buffers.action[agent] == None:
                    self.blue_team_buffers.action[agent] = -1
                    reward = 0
                    done = True
                else:
                    reward = self.env._cumulative_rewards[agent]
                    done = self.env.dones[agent]
                self.blue_team_buffers.observation_next[agent] = self.observe(agent)
                self.blue_team_buffers.episode_reward += reward
                self.transition[agent] = [self.blue_team_buffers.observation[agent], self.blue_team_buffers.action[agent],reward,done,self.blue_team_buffers.observation_next[agent]]
                self.blue_team_buffers.observation[agent] = self.blue_team_buffers.observation_next[agent]
            
        return done
    def store_transition(self, intitialisation=False): #not adapted for multiple agents
        if self.args.USE_PER:
            if intitialisation:
                p = 1
                w = (self.args.BUFFER_SIZE*(p/self.args.MIN_BUFFER_LENGTH))**-self.args.B_PER
            else:
                p = max(self.blue_team_buffers.priority)
                w = (self.args.BUFFER_SIZE*(p/sum(self.blue_team_buffers.priority)))**-self.args.B_PER
            self.blue_team_buffers.priority.append(p)
            
            self.blue_team_buffers.weights.append(w)
        
        self.blue_team_buffers.replay_buffer.append(self.transition)

    def sample(self):
        if self.args.USE_PER:

            priorities = np.asarray(self.blue_team_buffers.priority)
            self.index = np.random.choice(range(len(self.blue_team_buffers.replay_buffer)), self.args.BATCH_SIZE, p=(priorities/sum(priorities)))
            self.weights = np.asarray([self.blue_team_buffers.weights[i] for i in self.index])
            return [self.blue_team_buffers.replay_buffer[i] for i in self.index]
        else:
            return random.sample(self.blue_team_buffers.replay_buffer, self.args.BATCH_SIZE)
    
    def update_priorities(self, error):
        for i, index in enumerate(self.index):
            self.blue_team_buffers.priority[index] = (self.args.EPSILON_PER + error[i].item())**self.args.ALPHA_PER
            self.blue_team_buffers.weights[index] = (self.args.BUFFER_SIZE*(self.blue_team_buffers.priority[index]/sum(self.blue_team_buffers.priority)))**-self.args.B_PER
         
            
    def anneal(self, step):
        
        self.args.B_PER = np.interp(step, [0, self.args.EPSILON_DECAY], [self.args.B_PER_START, 1])
        return np.interp(step, [0, self.args.EPSILON_DECAY], [self.args.EPSILON_START, self.args.EPSILON_END])

    def step_buffer(self):
        self.blue_team_buffers.nb_transitions += 1
    def reset_buffer(self, agent):
        if self.is_opposing_team(agent):
            self.opposing_team_buffers.observation[agent] = self.observe(agent)
            
        else:
            self.blue_team_buffers.observation[agent] = self.observe(agent)

    def reset_buffers(self):
            self.opposing_team_buffers.episode_reward = 0
            self.opposing_team_buffers.nb_transitions = 0
            
            self.blue_team_buffers.episode_reward = 0
            self.blue_team_buffers.nb_transitions = 0
            for agent in self.args.all_agents:
                self.reset_buffer(agent)
    def complete_transition(self):
        if all([agent in self.transition for agent in self.args.blue_agents]):
            return
        else:
            for agent in self.args.blue_agents:
                if agent not in self.transition:  #done agents get 0 reward and keep same observation
                    self.transition[agent] = [self.blue_team_buffers.observation[agent], -1, -0.01, True,self.blue_team_buffers.observation_next[agent]]
    def winner_is_blue(self):
        first_agent_in_list = [*self.env.infos][0]
        if len(self.env.infos.keys()) == 1:
            return not self.is_opposing_team(first_agent_in_list)
        else: 
            if self.is_opposing_team(first_agent_in_list):
                out = False
            else:
                out = True
            winner = self.env.infos[first_agent_in_list].get('winner','is_a_tie')
            if winner == 'self':
                return out
            elif winner == 'other':
                return not out
            elif winner == 'is_a_tie':
                return False #a tie is considered a loss
    def give_global_reward(self):

        if self.winner_is_blue() == True: #need to find win criterium, normally first agent in agents left is blue team . > ??? to verify
            reward = self.args.WINNING_REWARD
        else:
            reward = self.args.LOSING_REWARD
        for agent in self.transition:
            self.transition[agent][2] = reward

    def is_opposing_team(self, agent):
        if re.match(rf'^{self.args.OPPOSING_TEAM}',agent):
            return True
        return False

    def adversary_tactic(self, agent, obs):
        #random tactic
        if self.args.ADVERSARY_TACTIC == 'random':
            return self.random_action(agent, obs)
        if self.args.ADVERSARY_TACTIC == 'passive':
            return 0 #return first available action : do nothijng
        if self.args.ADVERSARY_TACTIC == 'dqn':
            return self.adversary_nets[agent].act(obs)

    def sync_networks(self):
        for agent in self.args.blue_agents:
                self.target_nets[agent].net.load_state_dict(self.online_nets[agent].net.state_dict())
        

    def visualize(self):
          #evaluating the actual policy of the agents
        if self.args.VISUALIZE:
            self.args.GREEDY = False
            self.env.render()

    def save_model(self, train_step):  #taken from https://github.com/koenboeckx/qmix/blob/main/qmix.py to save learnt model
        
        params = Params(train_step)
        params.blue_team_buffers = self.blue_team_buffers
        if self.args.RUN_NAME != '':
            dirname = self.args.MODEL_DIR + '/' + self.args.RUN_NAME + '/' +datetime.now().strftime("%d%H%M%b%Y") +f'step_{train_step}'
            dirname_agents = self.args.MODEL_DIR + '/' + self.args.RUN_NAME + '/' +datetime.now().strftime("%d%H%M%b%Y") +f'step_{train_step}'+ '/agent_dqn_params/'
        else:
            dirname = self.args.MODEL_DIR + '/' +datetime.now().strftime("%d%H%M%b%Y")+ f'step_{train_step}'
            dirname_agents = self.args.MODEL_DIR + '/' +datetime.now().strftime("%d%H%M%b%Y")+ f'step_{train_step}' + '/agent_dqn_params/'

        if not os.path.exists(dirname_agents):
            os.makedirs(dirname_agents)
        for agent in self.args.blue_agents:
                torch.save(self.online_nets[agent].net.state_dict(),  dirname_agents   + agent + '.pt')
                torch.save(self.online_nets[agent].net.state_dict(),  dirname_agents + 'target_'   + agent + '.pt')
        with open(f'{dirname}/loading_parameters.bin',"wb") as f:
            pickle.dump(params, f)

    def load_model(self, dir, red = False):
        if red:
            for index in range(len(self.args.blue_agents)):
                agent_model = dir + '/agent_dqn_params/' + self.args.blue_agents[index] +'.pt'
                target_model = dir + '/agent_dqn_params/' + 'target_' + self.args.blue_agents[index] +'.pt'
                self.adversary_nets[self.args.red_agents[index]].net.load_state_dict(torch.load(agent_model))
            with open(f'{dir}/loading_parameters.bin',"rb") as f:
                self.loading_parameters = pickle.load(f)
            self.opposing_team_buffers.replay_buffer = self.loading_parameters.blue_team_replay_buffer
        else:            
            for agent in self.args.blue_agents:
                    agent_model = dir + '/agent_dqn_params/' + agent +'.pt'
                    self.online_nets[agent].net.load_state_dict(torch.load(agent_model))
                    try:
                        self.target_nets[agent].net.load_state_dict(torch.load(target_model))
                    except:
                        self.target_nets[agent].net.load_state_dict(torch.load(agent_model))
            with open(f'{dir}/loading_parameters.bin',"rb") as f:
                self.loading_parameters = pickle.load(f)
            self.args.ITER_START_STEP = self.loading_parameters.step
            try:  #for previous version compatibility
                self.blue_team_buffers = self.loading_parameters.blue_team_buffers
            except:
                self.blue_team_buffers.replay_buffer = self.loading_parameters.blue_team_replay_buffer

    def run(self):
        
        #Init replay buffer
        
        
        self.env.reset()
        if len(self.blue_team_buffers.replay_buffer) < self.args.MIN_BUFFER_LENGTH:
            for _ in range(self.args.MIN_BUFFER_LENGTH):
                
                self.transition = dict() #to store the transition 
                self.step_buffer() #count the number of transitions per episode
                for agent in self.env.agent_iter(max_iter=len(self.env.agents)):
                    if self.is_opposing_team(agent):
                        self.opposing_team_buffers.observation[agent], _, done, _ = self.last()
                        action = self.adversary_tactic(agent, self.opposing_team_buffers.observation[agent])
                        if done:
                            action = None
                        self.opposing_team_buffers.action[agent] = action
                        
                    else:
                        self.blue_team_buffers.observation[agent], _, done, _ = self.last()
                        action = self.random_action(agent, self.blue_team_buffers.observation[agent])
                        if done:
                            action = None
                        self.blue_team_buffers.action[agent] = action
                    self.env.step(action)
                    self.visualize()
                self.update_buffer()
                
                
                #self.complete_transition()
                if all(x == True for x in self.env.dones.values()):  #if all agents are done, episode is done, -> reset the environment
                    #self.give_global_reward()
                    self.blue_team_buffers.episode_reward = self.blue_team_buffers.episode_reward/(self.args.n_blue_agents) #
                    self.blue_team_buffers.rew_buffer.append(self.blue_team_buffers.episode_reward)
                    self.blue_team_buffers.steps_buffer.append(self.blue_team_buffers.episode_reward)
                    self.blue_team_buffers.wins_buffer.append(self.blue_team_buffers.episode_reward)
                    self.change_terrain()
                    
                    self.reset_buffers()
                self.store_transition(True)
        # trainingoptim

        self.env.reset()
        self.reset_buffers()
        transitions_counter = 0
        for step in itertools.count(start=self.args.ITER_START_STEP):   #modifying to continue training
            if transitions_counter > self.args.STOP_TRAINING:
                break
            if step > self.args.VISUALIZE_AFTER:
                self.args.VISUALIZE = True
            epsilon = self.anneal(step)
            rnd_sample = random.random()

            self.transition = dict()
            self.step_buffer()
            for agent in self.env.agent_iter(max_iter=len(self.env.agents)):
                
                if self.is_opposing_team(agent):
                    self.opposing_team_buffers.observation[agent], _, done, _ = self.last()
                    action = self.adversary_tactic(agent, self.opposing_team_buffers.observation[agent])
                    if done:
                        action = None
                    self.opposing_team_buffers.action[agent] = action
                else:
                    self.blue_team_buffers.observation[agent], _, done, _ = self.last()
                    action = self.online_nets[agent].act(self.blue_team_buffers.observation[agent])
                    if rnd_sample <= epsilon and self.args.GREEDY:
                        action = self.random_action(agent, self.blue_team_buffers.observation[agent])
                    if done:
                        action = None
                    self.blue_team_buffers.action[agent] = action
                self.env.step(action)
                self.visualize()
                
            self.update_buffer()
            #self.complete_transition()
            if all(x == True for x in self.env.dones.values()):  #if all agents are done, episode is done, -> reset the environment
                #self.give_global_reward()
                self.blue_team_buffers.episode_reward = self.blue_team_buffers.episode_reward/(self.args.n_blue_agents)
                self.blue_team_buffers.rew_buffer.append(self.blue_team_buffers.episode_reward)
                
                if self.args.TENSORBOARD:
                    self.writer.add_scalar("Reward", self.blue_team_buffers.episode_reward,step  )
                    self.writer.add_scalar("Steps", self.blue_team_buffers.nb_transitions,step  )
                    self.writer.add_scalar("Win", int(self.winner_is_blue()),step  )
                self.change_terrain()
                self.reset_buffers()

            self.store_transition()
            
            

            transitions = self.sample()
            loss_sum = 0
            for agent in self.args.blue_agents: #in case we use IDQN
        
                
                obses = np.asarray([t[agent][0]['obs'] for t in transitions])
                actions = np.asarray([t[agent][1] for t in transitions])
                rews = np.asarray([ t[agent][2] for t in transitions])
                dones = np.asarray([t[agent][3] for t in transitions])
                #new_obses = np.asarray([t[agent][4]['obs'] for t in transitions])
                obses_t = torch.as_tensor(obses, dtype=torch.float32,device=device)
                actions_t = torch.as_tensor(actions, dtype=torch.int64,device=device).unsqueeze(-1)
                rews_t = torch.as_tensor(rews, dtype=torch.float32,device=device)
                dones_t = torch.as_tensor(dones, dtype=torch.float32,device=device)
                #new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)
                
                if self.args.DOUBLE_DQN:
                    max_target_q_values = np.asarray([self.target_nets[agent].get_Q_values(t[agent][4])[0][self.online_nets[agent].get_Q_max(torch.masked_select(self.online_nets[agent].get_Q_values(t[agent][4]), torch.as_tensor(t[agent][4]['action_mask'], dtype=torch.bool,device=device)),t[agent][4],  self.online_nets[agent].get_Q_values(t[agent][4]))[0]].item() for t in transitions])
                else:
                    max_target_q_values = np.asarray([self.target_nets[agent].get_Q_max(torch.masked_select(self.target_nets[agent].get_Q_values(t[agent][4]), torch.as_tensor(t[agent][4]['action_mask'], dtype=torch.bool,device=device)),t[agent][4],  self.target_nets[agent].get_Q_values(t[agent][4]))[1].detach().item() for t in transitions])

                max_target_q_values_t = torch.as_tensor(max_target_q_values, dtype=torch.float32,device=device)
                # targets
                
                #self.target_nets[agent].get_q_max(torch.masked_select(self.target_nets[agent](torch.as_tensor(t[agent][4]['obs'], dtype=torch.float32)), torch.as_tensor(t[agent][4]['action_mask'], dtype=torch.bool,device=device)),t[agent][4],  self.target_nets[agent](torch.as_tensor(t[agent][4]['obs'], dtype=torch.float32)))
                #masked_target_q_values = 
                #max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

                targets = rews_t + self.args.GAMMA*(1 - dones_t)*max_target_q_values_t


                # loss 
                
                q_values = self.online_nets[agent](obses_t)
            
                action_q_values = torch.gather(input=q_values, dim=1, index=actions_t).squeeze(-1)

                error = targets - action_q_values
                
                if self.args.USE_PER:
                    self.update_priorities(abs(error))
                    error = error * torch.as_tensor(self.weights/max(self.weights), device = device)
                loss = error**2
                loss = loss.sum()
                # gradient descent
                self.online_nets[agent].optimizer.zero_grad()
                loss.backward()
                self.online_nets[agent].optimizer.step()
                loss_sum += loss.detach().item()


            loss_sum = loss_sum/(self.args.n_blue_agents*self.args.BATCH_SIZE)
            self.blue_team_buffers.loss_buffer.append(loss_sum)  # detach ?????????????????
            if self.args.TENSORBOARD:
                self.writer.add_scalar("Loss /agent", loss_sum, step)
            



            if step % self.args.SYNC_TARGET_FRAMES == 0:
                self.sync_networks()

            #save model

            if step % self.args.SAVE_CYCLE == 0:
                self.save_model(step)

            #logging
            if  self.args.PRINT_LOGS and step % self.args.REW_BUFFER_SIZE == 0:
                print('\n Step', step )
                print('Avg Episode Reward /agent ', np.mean(self.blue_team_buffers.rew_buffer))
                print('Avg Loss over a batch', np.mean(self.blue_team_buffers.loss_buffer))
            transitions_counter += 1

        if self.args.TENSORBOARD:
            self.writer.close() 

            
        ###

    def eval(self, params_directory, nb_episodes=200, visualize = False, log = False):
        results = Metrics(nb_episodes)
        self.env.reset()
        self.reset_buffers()
        if params_directory == 'random':
            self.transition = dict()
            ep_counter = 0
            for _ in itertools.count():
                if ep_counter == nb_episodes:
                    break
                self.step_buffer()
                for agent in self.env.agent_iter(max_iter=len(self.env.agents)):
                    if self.is_opposing_team(agent):
                        self.opposing_team_buffers.observation[agent], _, done, _ = self.last()
                        action = self.adversary_tactic(agent, self.opposing_team_buffers.observation[agent])
                        
                    else:
                        self.blue_team_buffers.observation[agent], _, done, _ = self.last()
                        action = self.random_action(agent, self.blue_team_buffers.observation[agent])
                    if done:
                        action = None
                    
                    self.env.step(action)
                    self.blue_team_buffers.action[agent] = action
                    if visualize:
                        self.env.render()
                        time.sleep(self.args.WAIT_BETWEEN_STEPS)

                self.update_buffer()
                
                if all(x == True for x in self.env.dones.values()):  #if all agents are done, episode is done, -> reset the environment
            
                    self.blue_team_buffers.episode_reward = self.blue_team_buffers.episode_reward/(self.args.n_blue_agents)
                    results.rewards_buffer.append(self.blue_team_buffers.episode_reward)
                    results.nb_steps.append(self.blue_team_buffers.nb_transitions)
                    results.wins.append(self.winner_is_blue())
                    if visualize:
                        self.env.render()
                    if log:
                        print(f'Episode reward /agent: {self.blue_team_buffers.episode_reward}')
                    ep_counter += 1
                    self.env.reset()
                    self.reset_buffers()
        else:
            self.load_model(params_directory)
            self.transition = dict()
            ep_counter = 0
            for _ in itertools.count():
                if ep_counter == nb_episodes:
                    break
                self.step_buffer()
                for agent in self.env.agent_iter(max_iter=len(self.env.agents)):
                    if self.is_opposing_team(agent):
                        self.opposing_team_buffers.observation[agent], _, done, _ = self.last()
                        action = self.adversary_tactic(agent, self.opposing_team_buffers.observation[agent])
                        
                    else:
                        self.blue_team_buffers.observation[agent], _, done, _ = self.last()
                        action = self.online_nets[agent].act(self.blue_team_buffers.observation[agent])
                    if done:
                        action = None
                    
                    self.env.step(action)
                    self.blue_team_buffers.action[agent] =  action
                    if visualize:
                        self.env.render()
                        time.sleep(self.args.WAIT_BETWEEN_STEPS)

                self.update_buffer()

                if all(x == True for x in self.env.dones.values()):  #if all agents are done, episode is done, -> reset the environment
            
                    self.blue_team_buffers.episode_reward = self.blue_team_buffers.episode_reward/(self.args.n_blue_agents)
                    results.rewards_buffer.append(self.blue_team_buffers.episode_reward)
                    results.nb_steps.append(self.blue_team_buffers.nb_transitions)
                    results.wins.append(self.winner_is_blue())
                    
                    
                    if visualize:
                        self.env.render()
                    if log:
                        print(f'Episode reward /agent: {self.blue_team_buffers.episode_reward}')
                        print(f'Episode steps: {self.blue_team_buffers.nb_transitions}')
                    ep_counter += 1
                    self.env.reset()
                    self.reset_buffers()

        reward = np.mean(results.rewards_buffer)
        steps = np.mean(results.nb_steps)
        wins = sum(results.wins)/nb_episodes
        print(f'Mean reward per episode: {np.mean(results.rewards_buffer)}')
        print(f'Mean steps per episode: {np.mean(results.nb_steps)}')
        print(f'Mean win rate: {sum(results.wins)/nb_episodes}')
        return results



def main(argv):
    env = defense_v0.env(terrain=TERRAIN, max_cycles=constants.EPISODE_MAX_LENGTH, max_distance=constants.MAX_DISTANCE )
    env.reset()
    args_runner = Args(env)
    args_runner.MODEL_DIR = MODEL_DIR
    args_runner.RUN_NAME = RUN_NAME
    args_runner.ADVERSARY_TACTIC = ADVERSARY_TACTIC
    
    try:
        opts, args = getopt.getopt(argv,"ha:l:e:",["load_adversary","load_model=","eval_model="])
    except getopt.GetoptError:
        print('error')
    if len(argv) == 0:
        runner = Runner(env, args_runner)
        runner.run()
    for opt, arg in opts:
        if opt == '-h':
            print('dqn.py')
            print('OR')
            print ('dqn.py -l <model_folder_to_load>')
            print('OR')
            print('dqn.py  -e <model_folder_to_eval> <folder_to_save_metrics>')
            sys.exit()
        elif opt in ('-a', "--load_adversary"):
            args_runner.ADVERSARY_MODEL = arg
            args_runner.ADVERSARY_TACTIC = 'dqn'
        elif opt in ("-l", "--load_model"):
            args_runner.MODEL_TO_LOAD = arg
            runner = Runner(env, args_runner)
            runner.run()
        elif opt in ("-e", "--eval_model"):
            runner = Runner(env, args_runner)
            runner.eval(arg)



if __name__ == "__main__":
    main(sys.argv[1:])
    # env = defense_v0.env(terrain=TERRAIN, max_cycles=constants.EPISODE_MAX_LENGTH, max_distance=constants.MAX_DISTANCE )
    # env.reset()
    # args_runner = Args(env)
    # args_runner.MODEL_DIR = MODEL_DIR
    # args_runner.RUN_NAME = RUN_NAME
    # args_runner.ADVERSARY_TACTIC = ADVERSARY_TACTIC
    # runner = Runner(env, args_runner)
    # runner.eval('defense_params_dqn/benchmarking/151503mai2022step_50000/')