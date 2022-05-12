import copy
from re import S
from Utils import helper
from Utils.helper import Buffers, Params, Metrics, Constants, mask_array, get_device
from Utils.params import QMIXArgs as Args
from Utils.agent import AgentNet
import random
import torch
import numpy as np
from datetime import datetime
import os
import pickle
import re
import itertools
import sys

#importing the defense environment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)
from env import defense_v0

from Utils import helper
from Utils.helper import Buffers, Params, Metrics, Constants, mask_array, get_device
constants = Constants()

class Runner:
    def __init__(self, env, args, TERRAIN):
        self.args = args
        self.env = env

        if self.args.ENV_FOLDER != '':
            self.CHANGE_ENV = True
            self.ter_array = []
            self.ter_ind = 0
            for filename in os.listdir(f'env/terrains/{self.args.ENV_FOLDER}'):
                self.ter_array.append(f'{self.args.ENV_FOLDER}/{filename[0:-4]}')
            self.env = defense_v0.env(terrain=self.ter_array[self.ter_ind] , max_cycles=constants.EPISODE_MAX_LENGTH, max_distance=constants.MAX_DISTANCE )
            self.env.reset()
          

        self.blue_team_buffers = Buffers(self.env, self.args, self.args.blue_agents, args.device)
        self.opposing_team_buffers = Buffers(self.env, self.args, self.args.opposing_agents, args.device)
        if self.args.ALGO == 'qmix':
            from RL_algorithms.q_mix_defense import QMixer
            self.online_net = QMixer(self.env, self.args)
        elif self.args.ALGO =='vdn':
            from RL_algorithms.vdn_defense import VDNMixer
            self.online_net = VDNMixer(self.env, self.args)
        self.target_net = copy.deepcopy(self.online_net) 

        if self.args.ADVERSARY_TACTIC == self.args.ALGO:
            self.adversary_net = copy.deepcopy(self.online_net) 
            self.load_model(self.args.ADVERSARY_MODEL, True)

        #display model graphs in tensorboard
        if self.args.TENSORBOARD:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter()
            args.log_params(self.writer, self.args.ALGO, TERRAIN)
            if self.args.EXTRA_PLOTS:
                from Utils.plotter import plot_nn

                plot_nn(self.online_net.agents_nets[self.args.blue_agents[0]], f'{self.args.ALGO.upper()} Agent') #to visualize the neural network
                
                plot_nn(self.online_net, f'{self.args.ALGO.upper()} Mixer')
                
                #self.writer.add_graph(self.online_net.agents_nets[self.args.blue_agents[0]],(torch.empty((self.args.observations_dim),device=device), torch.empty((1, self.args.dim_L2_agents_net),device=device)) )
                #self.writer.add_graph(self.online_net, (torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents*self.args.observations_dim), device=device)
            #                 , torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents), device=device)))
        self.sync_networks()
        
        
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True, patience =15000)  #patience, min lr... Parameters still to find
    
    def change_terrain(self):
        if self.args.CHANGE_ENV:
            self.ter_ind += 1
            self.env = defense_v0.env(terrain=self.ter_array[self.ter_ind] , max_cycles=constants.EPISODE_MAX_LENGTH, max_distance=constants.MAX_DISTANCE )
            
            
            if self.ter_ind >= (len(self.ter_array) - 1):
                self.ter_ind = 0
        self.env.reset()
    
    def random_action(self, agent, obs):
        if all(element == 0 for element in obs['action_mask']):
            return None
        return random.choice(mask_array(range(self.env.action_space(agent).n), obs['action_mask']))
    def update_buffer(self):
        for agent in self.env.agents:
            
            if self.is_opposing_team(agent):
                if self.opposing_team_buffers.action[agent] == None:
                    self.opposing_team_buffers.action[agent] = -1
                    reward = 0
                    done = True
                else:
                    reward = self.env._cumulative_rewards[agent]
                    done = self.env.dones[agent]
                
                self.opposing_team_buffers.observation_next[agent] = self.observe(agent)
                self.opposing_team_buffers.episode_reward += reward
                #store transition ?
                self.opposing_team_buffers.observation[agent] = self.opposing_team_buffers.observation_next[agent]
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
                self.transition[agent] = [self.blue_team_buffers.observation[agent], self.blue_team_buffers.action[agent],reward,done,self.blue_team_buffers.observation_next[agent], self.blue_team_buffers.hidden_state[agent], self.blue_team_buffers.hidden_state_next[agent]]
                self.blue_team_buffers.observation[agent] = self.blue_team_buffers.observation_next[agent]
                self.blue_team_buffers.hidden_state[agent] = self.blue_team_buffers.hidden_state_next[agent]
    
    def observe(self, agent):
        observation = copy.deepcopy(self.env.observe(agent))
        
        if self.args.CONVOLUTIONAL_INPUT:
            observation['obs'] = helper.obs_to_convInput(observation, self.args.ENV_SIZE, self.env.env.max_num_agents)
        return observation

    def last(self):
        observation, reward, done, info = self.env.last()
        if self.args.CONVOLUTIONAL_INPUT:
            return self.observe(self.env.agent_selection), reward, done, info
        else:
            return observation, reward, done, info
    def store_transition(self, intitialisation=False): #not adapted for multiple agents
        if self.args.USE_PER:
            if intitialisation:
                p = 1
                
            else:
                p = max(self.blue_team_buffers.priority)
            
            self.blue_team_buffers.priority.append(p)
            w = (self.args.BUFFER_SIZE*(p/sum(self.blue_team_buffers.priority)))**-self.args.B_PER
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
        self.args.ALPHA_PER = np.interp(step, [0, self.args.EPSILON_DECAY], [self.args.ALPHA_PER_START, 0])
        self.args.B_PER = np.interp(step, [0, self.args.EPSILON_DECAY], [self.args.B_PER_START, 1])
        return np.interp(step, [0, self.args.EPSILON_DECAY], [self.args.EPSILON_START, self.args.EPSILON_END])   
        
    def step_buffer(self):
        self.blue_team_buffers.nb_transitions += 1
        self.opposing_team_buffers.nb_transitions +=1
    def reset_buffer(self, agent):
        if self.is_opposing_team(agent):
            self.opposing_team_buffers.observation[agent] = self.observe(agent)
            self.opposing_team_buffers.hidden_state_next[agent] = torch.zeros(self.args.dim_L2_agents_net, device=self.args.device).unsqueeze(0)

        else:
            self.blue_team_buffers.observation[agent] = self.observe(agent)
            self.blue_team_buffers.hidden_state[agent] = torch.zeros(self.args.dim_L2_agents_net, device=self.args.device).unsqueeze(0)

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
            reward = 0
            if len(self.transition) != 0:
                reward = list(self.transition.items())[0][-1][2]  #all agents have the same reward as the agents still in the game
            for agent in self.args.blue_agents:
                if agent not in self.transition:  #done agents get 0 reward and keep same observation
                    self.transition[agent] = [self.blue_team_buffers.observation[agent], -1, reward, True,self.blue_team_buffers.observation_next[agent], self.blue_team_buffers.hidden_state[agent], self.blue_team_buffers.hidden_state_next[agent]]
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
        for agent in self.transition.keys():
            self.transition[agent][2] = reward

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
        elif self.args.ADVERSARY_TACTIC == 'qmix':
            action, self.opposing_team_buffers.hidden_state[agent] = self.adversary_net.act(agent, obs, self.opposing_team_buffers.hidden_state[agent])
            return action

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

    def get_obs_tot(self, obs):
        obs_t = torch.as_tensor(obs['obs'], dtype=torch.float32, device=self.args.device)
        if self.args.CONVOLUTIONAL_INPUT:
            return obs_t.reshape(np.prod(obs_t.shape))
        else:
            return obs_t

    def save_model(self, train_step):  #taken from https://github.com/koenboeckx/qmix/blob/main/qmix.py to save learnt model

        params = Params(train_step)
        params.blue_team_replay_buffer = self.blue_team_buffers.replay_buffer
        if self.args.RUN_NAME != '':
            dirname = self.args.MODEL_DIR + '/' + self.args.RUN_NAME + '/' +datetime.now().strftime("%d%H%M%b%Y") + f'step_{train_step}'
        else:
            dirname = self.args.MODEL_DIR + '/' +datetime.now().strftime("%d%H%M%b%Y")+ f'step_{train_step}'

        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(self.online_net.state_dict(), dirname + '/' + '/qmix_net_params.pt')
        torch.save(self.target_net.state_dict(), dirname + '/' + '/qmix_target_net_params.pt')
        if self.args.COMMON_AGENTS_NETWORK:
            torch.save(self.online_net.agents_net.state_dict(),  dirname + '/agents_net_params.pt')
        else:
            for agent in self.args.blue_agents:
                torch.save(self.online_net.agents_nets.state_dict(),  dirname + '/agent_nets_params/'  + agent + '.pt')
        with open(f'{dirname}/loading_parameters.bin',"wb") as f:
            pickle.dump(params, f)

    def load_model(self, dir, red=False):

        mixer_model = dir + '/qmix_net_params.pt'
        target_model = dir + '/qmix_target_net_params.pt'
        
        if red:
            self.adversary_net.load_state_dict(torch.load(mixer_model))
            if self.args.COMMON_AGENTS_NETWORK:
                agent_model = dir + '/agents_net_params.pt'
                self.adversary_net.agents_net.load_state_dict(torch.load(agent_model))
            else:
                for agent in self.args.red_agents:
                    agent_model = dir + '/agent_nets_params/' + agent +'.pt'
                    self.adversary_net.agents_nets[agent].load_state_dict(torch.load(agent_model))
            with open(f'{dir}/loading_parameters.bin',"rb") as f:
                self.loading_parameters = pickle.load(f)
            self.opposing_team_buffers.replay_buffer = self.loading_parameters.blue_team_replay_buffer

        else:
            self.online_net.load_state_dict(torch.load(mixer_model))
            self.target_net.load_state_dict(torch.load(target_model))
            if self.args.COMMON_AGENTS_NETWORK:
                agent_model = dir + '/agents_net_params.pt'
                self.online_net.agents_net.load_state_dict(torch.load(agent_model))
            else:
                for agent in self.args.blue_agents:
                    agent_model = dir + '/agent_nets_params/' + agent +'.pt'
                    self.online_net.agents_nets[agent].load_state_dict(torch.load(agent_model))
            with open(f'{dir}/loading_parameters.bin',"rb") as f:
                self.loading_parameters = pickle.load(f)
            self.blue_team_buffers.replay_buffer = self.loading_parameters.blue_team_replay_buffer

    def train(self, step):
          
        transitions = self.sample()
        obses_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents*self.args.observations_dim), device=self.args.device)
        actions_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents), device=self.args.device)
        Q_ins_target_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents), device=self.args.device, requires_grad=False)
        Q_action_online_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents), device=self.args.device)
        rewards_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents), device=self.args.device)
        dones_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents), device=self.args.device)
        next_obses_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents*self.args.observations_dim), device=self.args.device)
        transition_nb = 0
        for t in transitions:
            
            agent_nb = 0
            for agent in self.args.blue_agents:

                obses_t[transition_nb][self.args.observations_dim*agent_nb:(self.args.observations_dim*(agent_nb+1))] = self.get_obs_tot(t[agent][0])
                actions_t[transition_nb][agent_nb] = t[agent][1]
                rewards_t[transition_nb][agent_nb] = t[agent][2]
                dones_t[transition_nb][agent_nb] = t[agent][3]
                next_obses_t[transition_nb][self.args.observations_dim*agent_nb:(self.args.observations_dim*(agent_nb+1))] = self.get_obs_tot(t[agent][4])
                target_q_values = self.target_net.get_Q_values(agent, t[agent][4], t[agent][6])[0].squeeze(0)
                masked_target_q = torch.masked_select(target_q_values, torch.as_tensor(t[agent][4]['action_mask'], dtype=torch.bool,device=self.args.device))

                if self.args.DOUBLE_DQN:
                    Q_ins_target_t[transition_nb][agent_nb] = self.target_net.get_Q_values(agent, t[agent][4], t[agent][6])[0].squeeze(0)[self.online_net.get_Q_max(torch.masked_select(self.online_net.get_Q_values(agent, t[agent][4], t[agent][6])[0], torch.as_tensor(t[agent][4]['action_mask'], dtype=torch.bool,device=self.args.device)),t[agent][4],  self.online_net.get_Q_values(agent, t[agent][4], t[agent][6])[0])[0]].detach()
                    
                else:
                    Q_ins_target_t[transition_nb][agent_nb] = self.target_net.get_Q_max(masked_target_q, t[agent][4], target_q_values)[1].detach()
                if t[agent][1] == -1:
                    Q_action_online_t[transition_nb][agent_nb] = 0
                    Q_ins_target_t[transition_nb][agent_nb] = 0  #if agent is done also set target q value to 0 ?
                else:
                    Q_action_online_t[transition_nb][agent_nb] = torch.gather(self.online_net.get_Q_values(agent, t[agent][0], t[agent][5])[0].squeeze(0), 0,torch.tensor([t[agent][1]], device=self.args.device))
                
                
                agent_nb += 1
                


            transition_nb += 1

        #compute reward for all agents
        rewards_t = rewards_t.mean(1)
        
        #if all agents are done, done = 1
        all_agents_done_t = dones_t.sum(1)
        all_agents_done_t = all_agents_done_t == dones_t.shape[1]
        # targets
        Qtot_max_target = self.target_net.forward(next_obses_t, Q_ins_target_t).detach()
        Qtot_online = self.online_net.forward(obses_t, Q_action_online_t)
        y_tot = rewards_t + self.args.GAMMA*(1 - 1*all_agents_done_t)*Qtot_max_target

    ########### busy
        # loss 
        error = y_tot - Qtot_online
        
        if self.args.USE_PER:
            self.update_priorities(abs(error))
            error = error * torch.as_tensor(self.weights/max(self.weights), device = self.args.device)
                

        loss = error**2
        loss = loss.sum() 
        

        # gradient descent
        self.online_net.optimizer.zero_grad()
        loss.backward()
        self.online_net.optimizer.step()

        self.blue_team_buffers.loss_buffer.append(loss.item()/self.args.BATCH_SIZE)
        if self.args.TENSORBOARD:
            self.writer.add_scalar("Loss", loss.item()/self.args.BATCH_SIZE, step)
        

    def run(self):
        
        #Init replay buffer
        if self.args.MODEL_TO_LOAD != '':
            self.load_model(self.args.MODEL_TO_LOAD)
            self.args.ITER_START_STEP = self.loading_parameters.step
        self.env.reset()
        if len(self.blue_team_buffers.replay_buffer) == 0: #initializing replay buffer
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
                        _, self.blue_team_buffers.hidden_state_next[agent] = self.online_net.act(agent, self.blue_team_buffers.observation[agent], self.blue_team_buffers.hidden_state[agent])
                        self.blue_team_buffers.action[agent] = action
                    self.env.step(action)
                    
                    self.visualize()

                    #update buffer here
                    
                
                self.update_buffer()
                self.complete_transition()
                if all(x == True for x in self.env.dones.values()):  #if all agents are done, episode is done, -> reset the environment
                    self.give_global_reward()
                    self.blue_team_buffers.episode_reward = self.blue_team_buffers.episode_reward/(self.args.n_blue_agents) #
                    self.blue_team_buffers.rew_buffer.append(self.blue_team_buffers.episode_reward)

                    self.change_terrain()
                    
                    self.reset_buffers()
                self.store_transition(True)
        # trainingoptim

        self.env.reset()
        self.reset_buffers()
        transitions_counter = 0
        for step in itertools.count(start=self.args.ITER_START_STEP):
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
                    action, self.blue_team_buffers.hidden_state_next[agent] = self.online_net.act(agent, self.blue_team_buffers.observation[agent], self.blue_team_buffers.hidden_state[agent])
                    if rnd_sample <= epsilon and self.args.GREEDY:
                        action = self.random_action(agent, self.blue_team_buffers.observation[agent])
                    if done:
                        action = None
                    self.blue_team_buffers.action[agent] = action
                self.env.step(action)
                
                self.visualize()
            self.update_buffer()
            self.complete_transition()
            if all(x == True for x in self.env.dones.values()):  #if all agents are done, episode is done, -> reset the environment
                self.give_global_reward()
                self.blue_team_buffers.episode_reward = self.blue_team_buffers.episode_reward/(self.args.n_blue_agents)
                self.blue_team_buffers.rew_buffer.append(self.blue_team_buffers.episode_reward)
                
                if self.args.TENSORBOARD:
                    self.writer.add_scalar("Reward", self.blue_team_buffers.episode_reward,step  )
                    self.writer.add_scalar("Steps", self.blue_team_buffers.nb_transitions,step  )
                    self.writer.add_scalar("Win", int(self.winner_is_blue()),step  )
                self.change_terrain()
                self.reset_buffers()
                #self.train(step) #training only after each episode
            self.store_transition()
            self.train(step)  #training at each step
            
            #self.scheduler.step(np.mean(self.loss_buffer))
            #update target network

            if step % self.args.SYNC_TARGET_FRAMES == 0:
                self.sync_networks()

            #save model

            if step % self.args.SAVE_CYCLE == 0:
                self.save_model(step)

            #logging
            if step % self.args.REW_BUFFER_SIZE == 0:
                print('\n Step', step )
                print('Avg Episode Reward /agent ', np.mean(self.blue_team_buffers.rew_buffer))
                print('Avg Loss over a batch', np.mean(self.blue_team_buffers.loss_buffer))
            
            transitions_counter += 1
        if self.args.TENSORBOARD:
            self.writer.close() 

    
        ###

    def eval(self, params_directory, nb_episodes=10000, visualize = True, log = True):
        results = Metrics(nb_episodes)
        self.env.reset()
        self.reset_buffers()
        if params_directory == 'random':
            randomtest = True
        else:
            self.load_model(params_directory)
            randomtest = False
        self.transition = dict()
        ep_counter = 0
        
        for _ in itertools.count(start=self.args.ITER_START_STEP):
            if ep_counter == nb_episodes:
                break
            
            self.step_buffer()
            for agent in self.env.agent_iter(max_iter=len(self.env.agents)):
                
                if self.is_opposing_team(agent):
                    self.opposing_team_buffers.observation[agent], _, done, _ = self.last()
                    action = self.adversary_tactic(agent, self.opposing_team_buffers.observation[agent])
                   
                    
                else:
                    self.blue_team_buffers.observation[agent], _, done, _ = self.last()
                    if randomtest:
                        action = self.random_action(agent, self.blue_team_buffers.observation[agent])
                    else:
                        action, self.blue_team_buffers.hidden_state_next[agent] = self.online_net.act(agent, self.blue_team_buffers.observation[agent], self.blue_team_buffers.hidden_state[agent])

                if done:
                    action = None
                    
                self.env.step(action)
                print(f'{agent} : {action}')
                self.env.render()
                #time.sleep(self.args.WAIT_BETWEEN_STEPS)
                self.update_buffer(agent, action)
            ep_counter += 1
            if all(x == True for x in self.env.dones.values()):  #if all agents are done, episode is done, -> reset the environment
        
                self.blue_team_buffers.episode_reward = self.blue_team_buffers.episode_reward/(self.args.n_blue_agents)
                self.blue_team_buffers.rew_buffer.append(self.blue_team_buffers.episode_reward)
                self.env.reset()
                print(f'Episode reward /agent: {self.blue_team_buffers.episode_reward}')
                self.reset_buffers()
        return results
