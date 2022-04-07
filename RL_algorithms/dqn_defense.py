
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

#setting up TensorBoard
writer = SummaryWriter()

#environment constants
EPISODE_MAX_LENGTH = 200
MAX_DISTANCE = 5
TERRAIN = 'central_5x5'
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
        self.EPSILON_END = 0.02
        self.EPSILON_DECAY = 500000
        self.SYNC_TARGET_FRAMES = 200
        #visualization parameters
        self.PRINT_LOGS = False
        self.VISUALIZE_WHEN_LEARNED = True
        self.VISUALIZE_AFTER = 5000000
        self.VISUALIZE = False
        self.WAIT_BETWEEN_STEPS = 0.01
        self.GREEDY = True
        #save and reload model
        self.SAVE_CYCLE = 50000
        self.MODEL_DIR = 'defense_params_dqn'
        self.RUN_NAME = ''
        self.ITER_START_STEP = 0 #when starting training with an already trained model
        self.MODEL_TO_LOAD = ''
        #agent network parameters
        self.dim_L1_agents_net = 32
        self.dim_L2_agents_net = 32
        self.hidden_layer1_dim = 64
        #environment specific parameters calculation
        self.TEAM_TO_TRAIN = 'blue'
        self.OPPOSING_TEAM = 'red'
        self.ADVERSARY_TACTIC = 'passive'
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
        hparams = {'envparam/terrrain': TERRAIN,'Learning rate': self.LEARNING_RATE, 'Batch size': self.BATCH_SIZE, 'Buffer size': self.BUFFER_SIZE, 'Min buffer length': self.MIN_BUFFER_LENGTH, '\gamma': self.GAMMA, 'Epsilon range': f'{self.EPSILON_START} - {self.EPSILON_END}', 'Epsilon decay': self.EPSILON_DECAY, 'Synchronisation rate': self.SYNC_TARGET_FRAMES, 'Timestamp': int(datetime.timestamp(datetime.now()) - datetime.timestamp(datetime(2022, 2, 1, 11, 26, 31,0)))}
        metric_dict = { 'hparam/dim L1 agent net': self.dim_L1_agents_net, 'hparam/dim L2 agent net': self.dim_L2_agents_net}
        writer.add_hparams(hparams, metric_dict)

def mask_array(array, mask):
    int = np.ma.compressed(np.ma.masked_where(mask==0, array) )

    return np.ma.compressed(np.ma.masked_where(mask==0, array) )

class DQN(nn.Module):
    def __init__(self, env, args, observation_space_shape, action_space_n):
        super().__init__()
        #params
        args.hidden_layer1_dim = 32
        #hidden_layer2_dim = 60
        self.net = nn.Sequential(
            nn.Linear(np.prod(observation_space_shape) , args.hidden_layer1_dim),
            nn.ReLU(),
            #nn.Linear(hidden_layer1_dim,hidden_layer2_dim),
            #nn.ReLU(),
            nn.Linear(args.hidden_layer1_dim, action_space_n)
        ).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = args.LEARNING_RATE)
        
    def forward(self, obs):
        return self.net(torch.as_tensor(obs, dtype=torch.float32,device=device))
        
    def get_Q_values(self, obs):
        obs_t = torch.as_tensor(obs['obs'], dtype=torch.float32,device=device)
        q_values = self.net(obs_t.unsqueeze(0))
        return q_values

    def get_Q_max(self, q_values, obs, all_q_values=None):
        max_q_index = torch.argmax(q_values, dim=-1).detach().item()
        max_q = q_values[max_q_index]
        if all_q_values != None:
            max_q_index = ((all_q_values == max_q.item()) * (obs['action_mask'] == 1)).nonzero(as_tuple=True)[1][0].item()
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
    
        
        
    

class buffers:
    def __init__(self, env, args, agents):
        self.observation = dict()
        self.observation_next = dict()
        self.episode_reward = 0.0
        self.nb_transitions = 0
        self.replay_buffer = deque(maxlen=args.BUFFER_SIZE)
        self.rew_buffer = deque([0.0], maxlen=args.REW_BUFFER_SIZE)
        self.loss_buffer = deque([0.0], maxlen=args.REW_BUFFER_SIZE)

        for agent in agents:
            self.observation[agent] = env.observe(agent)

class runner:
    def __init__(self, env, args):
        self.args = args
        self.env = env
        self.blue_team_buffers = buffers(self.env, self.args, self.args.blue_agents)
        self.opposing_team_buffers = buffers(self.env, self.args, self.args.opposing_agents)
        self.online_nets = {}
        self.target_nets = {}
        self.online_nets_params = list()
        for agent in self.args.blue_agents:
            self.online_nets[agent] = DQN(self.env, self.args, self.env.observation_space(agent).spaces['obs'].shape, self.env.action_space(agent).n)

        self.target_nets = copy.deepcopy(self.online_nets)
        #display model graphs in tensorboard
        writer.add_graph(self.online_nets[self.args.blue_agents[0]],torch.empty((self.args.observations_dim),device=device) )
        
        self.sync_networks()
        
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
            self.opposing_team_buffers.observation_next[agent] = self.env.observe(agent)
            self.opposing_team_buffers.episode_reward += reward
            #store transition ?
            self.opposing_team_buffers.observation[agent] = self.opposing_team_buffers.observation_next[agent]
        else:
            self.blue_team_buffers.observation_next[agent] = self.env.observe(agent)
            self.blue_team_buffers.episode_reward += reward
            
            self.transition[agent] = (self.blue_team_buffers.observation[agent], action,reward,done,self.blue_team_buffers.observation_next[agent])
            self.blue_team_buffers.observation[agent] = self.blue_team_buffers.observation_next[agent]
            
        return done
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

    def sync_networks(self):
        for agent in self.args.blue_agents:
                self.target_nets[agent].load_state_dict(self.online_nets[agent].state_dict())

    def visualize(self):
          #evaluating the actual policy of the agents
        if self.args.VISUALIZE:
            self.args.GREEDY = False
            self.env.render()

    def save_model(self, train_step):  #taken from https://github.com/koenboeckx/qmix/blob/main/qmix.py to save learnt model
        num = str(train_step // self.args.SAVE_CYCLE)
        if self.args.RUN_NAME != '':
            dirname = self.args.MODEL_DIR + '/' + self.args.RUN_NAME + '/' +datetime.now().strftime("%d%H%M%b%Y") + '/agent_dqn_params/'
        else:
            dirname = self.args.MODEL_DIR + '/' +datetime.now().strftime("%d%H%M%b%Y")+ '/agent_dqn_params/'

        if not os.path.exists(dirname):
            os.makedirs(dirname)
        for agent in self.args.blue_agents:
                torch.save(self.online_nets[agent].state_dict(),  dirname   + agent + '.pt')

    def load_model(self, dir):
        for agent in self.args.blue_agents:
                agent_model = dir + '/agent_dqn_params/' + agent +'.pt'
                self.online_nets[agent].load_state_dict(torch.load(agent_model))

    def run(self):
        
        #Init replay buffer
        
        if self.args.MODEL_TO_LOAD != '':
            self.load_model(self.args.MODEL_TO_LOAD)
            print('ok')
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
                    
                self.env.step(action)
                self.visualize()
                self.update_buffer(agent, action)
            
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
                    action = self.online_nets[agent].act(self.blue_team_buffers.observation[agent])
                    if rnd_sample <= epsilon and self.args.GREEDY:
                        action = self.random_action(agent, self.blue_team_buffers.observation[agent])
                    if done:
                        action = None
                    
                self.env.step(action)
                self.visualize()
                self.update_buffer(agent, action)
            
            self.blue_team_buffers.replay_buffer.append(self.transition)
            
            if all(x == True for x in self.env.dones.values()):  #if all agents are done, episode is done, -> reset the environment
        
                self.blue_team_buffers.episode_reward = self.blue_team_buffers.episode_reward/(self.args.n_blue_agents)
                self.blue_team_buffers.rew_buffer.append(self.blue_team_buffers.episode_reward)
                self.env.reset()
                writer.add_scalar("Reward", self.blue_team_buffers.episode_reward,step  )
                self.reset_buffers()


            

            transitions = random.sample(self.blue_team_buffers.replay_buffer, self.args.BATCH_SIZE)
            loss_sum = 0
            for agent in self.args.blue_agents:
        

                obses = np.asarray([t[agent][0]['obs'] for t in transitions])
                actions = np.asarray([t[agent][1] for t in transitions])
                rews = np.asarray([ t[agent][2] for t in transitions])
                dones = np.asarray([t[agent][3] for t in transitions])
                new_obses = np.asarray([t[agent][4]['obs'] for t in transitions])

                obses_t = torch.as_tensor(obses, dtype=torch.float32)
                actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
                rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
                dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
                new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

                # targets

                target_q_values = self.target_nets[agent](new_obses_t)
                max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

                targets = rews_t + self.args.GAMMA*(1 - dones_t)*max_target_q_values


                # loss 
                
                q_values = self.online_nets[agent](obses_t)

                action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

                error = targets - action_q_values
                loss = error**2
                loss = loss.sum()
                # gradient descent
                self.online_nets[agent].optimizer.zero_grad()
                loss.backward()
                self.online_nets[agent].optimizer.step()
                loss_sum += loss.detach().item()


            loss_sum = loss_sum/self.args.n_blue_agents
            self.blue_team_buffers.loss_buffer.append(loss_sum)  # detach ?????????????????
            writer.add_scalar("Loss /agent", loss_sum, step)
            



            if step % args.SYNC_TARGET_FRAMES == 0:
                self.sync_networks()

            #save model

            if step % self.args.SAVE_CYCLE:
                self.save_model(step)

            #logging
            if step % self.args.REW_BUFFER_SIZE == 0 and self.args.PRINT_LOGS:
                print('\n Step', step )
                print('Avg Episode Reward /agent ', np.mean(self.blue_team_buffers.rew_buffer))
                print('Avg Loss over a batch', np.mean(self.blue_team_buffers.loss_buffer))
        writer.close() 

    
        ###

    def eval(self, params_directory):

        self.env.reset()
        self.reset_buffers()
        self.load_model(params_directory)
        self.transition = dict()
        for step in itertools.count(start=self.args.ITER_START_STEP):
            
            self.step_buffer()
            for agent in self.env.agent_iter(max_iter=len(self.env.agents)):
                
                if self.is_opposing_team(agent):
                    self.opposing_team_buffers.observation[agent], _, done, _ = self.env.last()
                    action = self.adversary_tactic(agent, self.opposing_team_buffers.observation[agent])
                    
                else:
                    self.blue_team_buffers.observation[agent], _, done, _ = self.env.last()
                    action = self.online_nets[agent].act(self.blue_team_buffers.observation[agent])
                if done:
                    self.env.step(None)
                    self.env.render()
                    time.sleep(self.args.WAIT_BETWEEN_STEPS)
                else:
                    self.env.step(action)
                    
                    time.sleep(self.args.WAIT_BETWEEN_STEPS)

                if action == None:
                    action = -1
                done = self.update_buffer(agent, action)

            if all(x == True for x in self.env.dones.values()):  #if all agents are done, episode is done, -> reset the environment
        
                self.blue_team_buffers.episode_reward = self.blue_team_buffers.episode_reward/(self.args.n_blue_agents)
                self.blue_team_buffers.rew_buffer.append(self.blue_team_buffers.episode_reward)
                self.env.reset()
                self.env.render()
                print(f'Episode reward /agent: {self.blue_team_buffers.episode_reward}')
                self.reset_buffers()




if __name__ == "__main__":
    env = defense_v0.env(terrain=TERRAIN, max_cycles=EPISODE_MAX_LENGTH, max_distance=MAX_DISTANCE )
    env.reset()
    args = Args(env)
    args.log_params(writer)
    runner = runner(env, args)
    if len(sys.argv) == 1:
        runner.run()
    else:
        runner.eval(sys.argv[1])

