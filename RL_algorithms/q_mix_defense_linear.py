
from ast import Param
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
import sys, getopt
import time
import re
import pickle
import cProfile 

#importing the defense environment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)
from env import defense_v0


    
from Utils.helper import Buffers, Params, Metrics, Constants, mask_array, get_device
from Utils.params import QMIXArgs as Args

device = get_device()
#environment constants
constants = Constants()
TERRAIN = 'flat_5x5'
MODEL_DIR = 'defense_params_qmixlin'
RUN_NAME = ''
ADVERSARY_TACTIC = 'random'


class QMixer(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.to(device)
        #params
        self.args = args
        total_state_dim = 0
        if self.args.COMMON_AGENTS_NETWORK:
            self.agents_net = AgentNet(self.args)
            for agent in self.args.blue_agents:
                total_state_dim += np.prod(env.observation_space(agent).spaces['obs'].shape)               

        else:
            self.agents_nets = dict()
            for agent in env.agents:
                self.agents_nets[agent] = AgentNet(args)
                total_state_dim += np.prod(env.observation_space(agent).shape)


        self.weightsL1_net = nn.Linear(total_state_dim, self.args.mixer_hidden_dim*self.args.n_blue_agents, device=device)
        self.biasesL1_net = nn.Linear(total_state_dim, self.args.mixer_hidden_dim, device=device)
        
        self.weightsL2_net = nn.Linear(total_state_dim, self.args.mixer_hidden_dim2*self.args.mixer_hidden_dim, device=device)
        self.biasesL2_net = nn.Sequential(
            nn.Linear(total_state_dim, self.args.mixer_hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(self.args.mixer_hidden_dim, self.args.mixer_hidden_dim2, device=device)
        )
        
        if self.args.COMMON_AGENTS_NETWORK:
            self.net_params = list(self.agents_net.net.parameters()) + list(self.weightsL1_net.parameters()) + list(self.biasesL1_net.parameters())  +list(self.weightsL2_net.parameters()) + list(self.biasesL2_net.parameters())
        else:
            agent_params = list()
            for agent_net in self.agents_nets.values():
                agent_params += list(agent_net.net.parameters())
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
        
    def get_Q_values(self, agent, obs):
        obs_t = torch.as_tensor(obs['obs'], dtype=torch.float32,device=device)
        q_values = self.get_agent_nets(agent)(obs_t)
        return q_values

    def get_Q_max(self, masked_q_values, obs, all_q_values=None):
        if len(masked_q_values) == 0:
            return -1, torch.tensor([0],device=device)
        max_q_index = torch.argmax(masked_q_values, dim=-1).detach().item()
        max_q = masked_q_values[max_q_index] 
        if all_q_values != None: #only in case a mask is given
            max_q_index = ((all_q_values == max_q.item()).cpu() * (obs['action_mask'] == 1)).nonzero(as_tuple=True)[0][0].item()
        return max_q_index, max_q


    def act(self, agent, obs):
        with torch.no_grad():
            q_values = self.get_Q_values(agent, obs)
            #taking only masked q values to choose action to take
            masked_q_val = torch.masked_select(q_values, torch.as_tensor(obs['action_mask'], dtype=torch.bool,device=device))
            if masked_q_val.numel() == 0:
                return None
            action, _ = self.get_Q_max(masked_q_val, obs, q_values)
            return action
    
        
        
    
class AgentNet(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.nb_inputs_agent, args.dim_L1_agents_net, device=device),
            nn.ReLU(),
            #nn.Linear(args.dim_L1_agents_net,args.dim_L2_agents_net, device=device),
            #nn.ReLU(),
            nn.Linear(args.dim_L1_agents_net, args.n_actions, device=device)
        ).to(device)
    def forward(self,obs_t):
        return self.net(obs_t)

 

class Runner:
    def __init__(self, env, args):
        self.args = args
        self.env = env
        self.blue_team_buffers = Buffers(self.env, self.args, self.args.blue_agents, device)
        self.opposing_team_buffers = Buffers(self.env, self.args, self.args.opposing_agents, device)
        self.online_net = QMixer(self.env, self.args)
        self.target_net = copy.deepcopy(self.online_net) #QMixer(self.env, self.args)


        #display model graphs in tensorboard
        self.writer = SummaryWriter()
        args.log_params(self.writer, 'qmixlin', TERRAIN)
        self.writer.add_graph(self.online_net.get_agent_nets(self.args.blue_agents[0]),(torch.empty((self.args.observations_dim),device=device)) )
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
            reward = -0.01
            done = True
        else:
            reward = self.env._cumulative_rewards[agent]
            done = self.env.dones[agent]
            
        if self.is_opposing_team(agent):
            
            self.opposing_team_buffers.observation_next[agent] = self.env.observe(agent)
            self.opposing_team_buffers.episode_reward += reward
            
            self.opposing_team_buffers.observation[agent] = self.opposing_team_buffers.observation_next[agent]
        else:
            
            self.blue_team_buffers.observation_next[agent] = self.env.observe(agent)
            
            self.blue_team_buffers.episode_reward += reward
            self.transition[agent] = [self.blue_team_buffers.observation[agent], action,reward,done,self.blue_team_buffers.observation_next[agent]]
            self.blue_team_buffers.observation[agent] = self.blue_team_buffers.observation_next[agent]
            
        
    def step_buffer(self):
        self.blue_team_buffers.nb_transitions += 1
        self.opposing_team_buffers.nb_transitions +=1
    def reset_buffer(self, agent):
        if self.is_opposing_team(agent):
            self.opposing_team_buffers.observation[agent] = self.env.observe(agent)

        else:
            self.blue_team_buffers.observation[agent] = self.env.observe(agent)

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

    def load_model(self, dir):
        mixer_model = dir + '/qmix_net_params.pt'
        target_model = dir + '/qmix_target_net_params.pt'
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
          
        transitions = random.sample(self.blue_team_buffers.replay_buffer, self.args.BATCH_SIZE)
        obses_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents*self.args.observations_dim), device=device)
        actions_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents), device=device)
        Q_ins_target_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents), device=device)
        Q_action_online_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents), device=device)
        rewards_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents), device=device)
        dones_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents), device=device)
        next_obses_t = torch.empty((self.args.BATCH_SIZE,self.args.n_blue_agents*self.args.observations_dim), device=device)
        transition_nb = 0
        for t in transitions:
            
            agent_nb = 0
            for agent in self.args.blue_agents:

                obses_t[transition_nb][self.args.observations_dim*agent_nb:(self.args.observations_dim*(agent_nb+1))] = torch.as_tensor(t[agent][0]['obs'], dtype=torch.float32, device=device)
                actions_t[transition_nb][agent_nb] = t[agent][1]
                rewards_t[transition_nb][agent_nb] = t[agent][2]
                dones_t[transition_nb][agent_nb] = t[agent][3]
                next_obses_t[transition_nb][self.args.observations_dim*agent_nb:(self.args.observations_dim*(agent_nb+1))] = torch.as_tensor(t[agent][4]['obs'], dtype=torch.float32, device=device).detach()
                if t[agent][1] == -1:
                    Q_action_online_t[transition_nb][agent_nb] = 0
                else:
                    Q_action_online_t[transition_nb][agent_nb] = torch.gather(self.online_net.get_Q_values(agent, t[agent][0]).squeeze(0), 0,torch.tensor([t[agent][1]], device=device))
                target_q_values = self.target_net.get_Q_values(agent, t[agent][4])
                masked_target_q = torch.masked_select(target_q_values.squeeze(0), torch.as_tensor(t[agent][4]['action_mask'], dtype=torch.bool,device=device))
                Q_ins_target_t[transition_nb][agent_nb] = self.target_net.get_Q_max(masked_target_q, t[agent][4], target_q_values)[1].detach()
                
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
        y_tot = rewards_t + self.args.GAMMA*(1 + (-1)*all_agents_done_t)*Qtot_max_target

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
                        self.opposing_team_buffers.observation[agent], _, done, _ = self.env.last()
                        action = self.adversary_tactic(agent, self.opposing_team_buffers.observation[agent])
                        if done:
                            action = None
                    
                    else:
                        self.blue_team_buffers.observation[agent], _, done, _ = self.env.last()
                        action = self.random_action(agent, self.blue_team_buffers.observation[agent])
                        if done:
                            action = None
                        
                    self.env.step(action)
                    self.update_buffer(agent, action)
                    self.visualize()

                    #update buffer here
                    
                
                
                self.complete_transition()
                if all(x == True for x in self.env.dones.values()):  #if all agents are done, episode is done, -> reset the environment
                    self.give_global_reward()
                    self.blue_team_buffers.episode_reward = self.blue_team_buffers.episode_reward/(self.args.n_blue_agents) #
                    self.blue_team_buffers.rew_buffer.append(self.blue_team_buffers.episode_reward)
                    self.env.reset()
                    
                    self.reset_buffers()
                self.blue_team_buffers.replay_buffer.append(self.transition)
        # trainingoptim

        self.env.reset()
        self.reset_buffers()

        transitions_counter = 0
        for step in itertools.count(start=self.args.ITER_START_STEP):
            if transitions_counter > self.args.STOP_TRAINING:
                break
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
                    action = self.online_net.act(agent, self.blue_team_buffers.observation[agent])
                    if rnd_sample <= epsilon and self.args.GREEDY:
                        action = self.random_action(agent, self.blue_team_buffers.observation[agent])
                    if done:
                        action = None
                    
                self.env.step(action)
                self.update_buffer(agent, action)
                self.visualize()
            
            self.complete_transition()
            if all(x == True for x in self.env.dones.values()):  #if all agents are done, episode is done, -> reset the environment
                self.give_global_reward()
                self.blue_team_buffers.episode_reward = self.blue_team_buffers.episode_reward/(self.args.n_blue_agents)
                self.blue_team_buffers.rew_buffer.append(self.blue_team_buffers.episode_reward)
                self.env.reset()
                self.writer.add_scalar("Reward", self.blue_team_buffers.episode_reward,step  )
                self.reset_buffers()
                #self.train(step) #training only after each episode
            self.blue_team_buffers.replay_buffer.append(self.transition)
            self.train(step)
            
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
                    action = self.online_net.act(agent, self.blue_team_buffers.observation[agent])
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



def main(argv):
    env = defense_v0.env(terrain=TERRAIN, max_cycles=constants.EPISODE_MAX_LENGTH, max_distance=constants.MAX_DISTANCE )
    env.reset()
    args = Args(env)
    args.MODEL_DIR = MODEL_DIR
    args.RUN_NAME = RUN_NAME
    args.ADVERSARY_TACTIC = ADVERSARY_TACTIC
    runner = Runner(env, args)
    try:
        opts, args = getopt.getopt(argv,"hl:e:",["load_model=","eval_model="])
    except getopt.GetoptError:
        print('error')
    if len(argv) == 0:
        runner.run()
    for opt, arg in opts:
        if opt == '-h':
            print('q_mix.py')
            print ('q_mix.py -l <model_folder_to_load>')
            print('OR')
            print('q_mix.py  -e <model_folder_to_eval>')
            sys.exit()
        elif opt in ("-l", "--load_model"):
            runner.args.MODEL_TO_LOAD = arg
            runner.run()
        elif opt in ("-e", "--eval_model"):
            runner.eval(arg)



if __name__ == "__main__":
    main(sys.argv[1:])
        

