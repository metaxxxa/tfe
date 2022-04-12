
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
os.chdir('/home/jack/Documents/ERM/Master thesis/tfe')
sys.path.insert(0, '/home/jack/Documents/ERM/Master thesis/tfe')
from env import defense_v0

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
dev = "cpu"  #if cpu faster than gpu 
device = torch.device(dev) 



#environment constants
EPISODE_MAX_LENGTH = 200
MAX_DISTANCE = 5
TERRAIN = 'benchmark_10x10_1v1'

#to log results
class Metrics:
    def __init__(self, nb_episodes):
        self.nb_episodes = nb_episodes
        self.rewards_buffer = deque(maxlen=nb_episodes)
        self.wins = deque(maxlen=nb_episodes)
        self.nb_steps = deque(maxlen=nb_episodes)
        self.env = ''
        
#parameters
class Args:
    def __init__(self, env):
            
        self.BUFFER_SIZE = 2000
        self.REW_BUFFER_SIZE = 10000
        self.LEARNING_RATE = 0.5e-4
        self.MIN_BUFFER_LENGTH = 300
        self.BATCH_SIZE = 64
        self.GAMMA = 0.95
        self.EPSILON_START = 1
        self.EPSILON_END = 0.01
        self.EPSILON_DECAY = 400000
        self.SYNC_TARGET_FRAMES = 200
        self.STOP_TRAINING = self.EPSILON_DECAY*2
        #visualization parameters
        self.PRINT_LOGS = False
        self.VISUALIZE_WHEN_LEARNED = True
        self.VISUALIZE_AFTER = 5000000
        self.VISUALIZE = False
        self.WAIT_BETWEEN_STEPS = 0.01
        self.GREEDY = True
        #logging
        self.TENSORBOARD = True
        #save and reload model
        self.SAVE_CYCLE = 50000
        self.MODEL_DIR = 'defense_params_dqn'
        self.RUN_NAME = 'benchmarking'
        self.ITER_START_STEP = 0 #when starting training with an already trained model
        self.MODEL_TO_LOAD = ''
        #agent network parameters
        self.dim_L1_agents_net = 32
        self.dim_L2_agents_net = 32
        self.hidden_layer1_dim = 64
        #environment specific parameters calculation
        self.WINNING_REWARD = 1
        self.LOSING_REWARD = -1
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
    def log_params(self, writer=None):
        hparams = {'envparam/terrrain': TERRAIN,'Adversary tactic':self.ADVERSARY_TACTIC , 'Algorithm': 'DQN' , 'Learning rate': self.LEARNING_RATE, 'Batch size': self.BATCH_SIZE, 'Buffer size': self.BUFFER_SIZE, 'Min buffer length': self.MIN_BUFFER_LENGTH, '/gamma': self.GAMMA, 'Epsilon range': f'{self.EPSILON_START} - {self.EPSILON_END}', 'Epsilon decay': self.EPSILON_DECAY, 'Synchronisation rate': self.SYNC_TARGET_FRAMES, 'Timestamp': int(datetime.timestamp(datetime.now()) - datetime.timestamp(datetime(2022, 2, 1, 11, 26, 31,0)))}
        metric_dict = { 'hparam/dim L1 agent net': self.dim_L1_agents_net, 'hparam/dim L2 agent net': self.dim_L2_agents_net}
        if self.TENSORBOARD:
            writer.add_hparams(hparams, metric_dict)

def mask_array(array, mask):
    int = np.ma.compressed(np.ma.masked_where(mask==0, array) )

    return np.ma.compressed(np.ma.masked_where(mask==0, array) )
class Params:
    def __init__(self, step):
        self.step = step
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
        if len(q_values) == 0:
            return -1, torch.tensor([0],device=device)
        max_q_index = torch.argmax(q_values, dim=-1).detach().item()
        max_q = q_values[max_q_index]
        if all_q_values != None:
            max_q_index = ((all_q_values == max_q.item()).cpu() * (obs['action_mask'] == 1)).nonzero(as_tuple=True)[1][0].item()
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
        if self.args.TENSORBOARD:
            #setting up TensorBoard
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter()
            args.log_params(self.writer)
                #display model graphs in tensorboard
            self.writer.add_graph(self.online_nets[self.args.blue_agents[0]],torch.empty((self.args.observations_dim),device=device) )
        
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
            
            self.transition[agent] = [self.blue_team_buffers.observation[agent], action,reward,done,self.blue_team_buffers.observation_next[agent]]
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
        params.blue_team_replay_buffer = self.blue_team_buffers.replay_buffer
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
        with open(f'{dirname}/loading_parameters.bin',"wb") as f:
            pickle.dump(params, f)

    def load_model(self, dir):
        for agent in self.args.blue_agents:
                agent_model = dir + '/agent_dqn_params/' + agent +'.pt'
                self.online_nets[agent].net.load_state_dict(torch.load(agent_model))
        with open(f'{dir}/loading_parameters.bin',"rb") as f:
            self.loading_parameters = pickle.load(f)
        self.blue_team_buffers.replay_buffer = self.loading_parameters.blue_team_replay_buffer


    def run(self):
        
        #Init replay buffer
        
        if self.args.MODEL_TO_LOAD != '':
            self.load_model(self.args.MODEL_TO_LOAD)
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
            
            
            #self.complete_transition()
            if all(x == True for x in self.env.dones.values()):  #if all agents are done, episode is done, -> reset the environment
                #self.give_global_reward()
                self.blue_team_buffers.episode_reward = self.blue_team_buffers.episode_reward/(self.args.n_blue_agents) #
                self.blue_team_buffers.rew_buffer.append(self.blue_team_buffers.episode_reward)
                self.env.reset()
                
                self.reset_buffers()
            self.blue_team_buffers.replay_buffer.append(self.transition)
        # trainingoptim

        self.env.reset()
        self.reset_buffers()
        transitions_counter = 0
        for step in itertools.count():
            if transitions_counter == self.args.STOP_TRAINING:
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
                    action = self.online_nets[agent].act(self.blue_team_buffers.observation[agent])
                    if rnd_sample <= epsilon and self.args.GREEDY:
                        action = self.random_action(agent, self.blue_team_buffers.observation[agent])
                    if done:
                        action = None
                    
                self.env.step(action)
                self.visualize()
                self.update_buffer(agent, action)
            
            #self.complete_transition()
            if all(x == True for x in self.env.dones.values()):  #if all agents are done, episode is done, -> reset the environment
                #self.give_global_reward()
                self.blue_team_buffers.episode_reward = self.blue_team_buffers.episode_reward/(self.args.n_blue_agents)
                self.blue_team_buffers.rew_buffer.append(self.blue_team_buffers.episode_reward)
                self.env.reset()
                if self.args.TENSORBOARD:
                    self.writer.add_scalar("Reward", self.blue_team_buffers.episode_reward,step  )
                self.reset_buffers()

            self.blue_team_buffers.replay_buffer.append(self.transition)
            
            

            transitions = random.sample(self.blue_team_buffers.replay_buffer, self.args.BATCH_SIZE)
            loss_sum = 0
            for agent in self.args.blue_agents:
        
                
                obses = np.asarray([t[agent][0]['obs'] for t in transitions])
                actions = np.asarray([t[agent][1] for t in transitions])
                rews = np.asarray([ t[agent][2] for t in transitions])
                dones = np.asarray([t[agent][3] for t in transitions])
                #new_obses = np.asarray([t[agent][4]['obs'] for t in transitions])
                max_target_q_values = np.asarray([self.target_nets[agent].get_Q_max(torch.masked_select(self.target_nets[agent].get_Q_values(t[agent][4]), torch.as_tensor(t[agent][4]['action_mask'], dtype=torch.bool,device=device)),t[agent][4],  self.target_nets[agent].get_Q_values(t[agent][4]))[1].detach().item() for t in transitions])

                obses_t = torch.as_tensor(obses, dtype=torch.float32,device=device)
                actions_t = torch.as_tensor(actions, dtype=torch.int64,device=device).unsqueeze(-1)
                rews_t = torch.as_tensor(rews, dtype=torch.float32,device=device).unsqueeze(-1)
                dones_t = torch.as_tensor(dones, dtype=torch.float32,device=device).unsqueeze(-1)
                #new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)
                max_target_q_values_t = torch.as_tensor(max_target_q_values, dtype=torch.float32,device=device)
                # targets
                
                #self.target_nets[agent].get_q_max(torch.masked_select(self.target_nets[agent](torch.as_tensor(t[agent][4]['obs'], dtype=torch.float32)), torch.as_tensor(t[agent][4]['action_mask'], dtype=torch.bool,device=device)),t[agent][4],  self.target_nets[agent](torch.as_tensor(t[agent][4]['obs'], dtype=torch.float32)))
                #masked_target_q_values = 
                #max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

                targets = rews_t + self.args.GAMMA*(1 - dones_t)*max_target_q_values_t


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

    def eval(self, params_directory, nb_episodes=10000, visualize = True, log = True):
        results = Metrics(nb_episodes)
        self.env.reset()
        self.reset_buffers()
        self.load_model(params_directory)
        self.transition = dict()
        ep_counter = 0
        for _ in itertools.count(start=self.args.ITER_START_STEP):
            if ep_counter == nb_episodes:
                break
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
                    if visualize:
                        self.env.render()
                    time.sleep(self.args.WAIT_BETWEEN_STEPS)
                else:
                    self.env.step(action)
                    
                    time.sleep(self.args.WAIT_BETWEEN_STEPS)

                self.update_buffer(agent, action)

            if all(x == True for x in self.env.dones.values()):  #if all agents are done, episode is done, -> reset the environment
        
                self.blue_team_buffers.episode_reward = self.blue_team_buffers.episode_reward/(self.args.n_blue_agents)
                results.rewards_buffer.append(self.blue_team_buffers.episode_reward)
                results.nb_steps.append(self.blue_team_buffers.nb_transitions)
                results.wins.append(self.winner_is_blue())
                self.env.reset()
                if visualize:
                    self.env.render()
                if log:
                    print(f'Episode reward /agent: {self.blue_team_buffers.episode_reward}')
                self.reset_buffers()
                ep_counter += 1

        return results

if __name__ == "__main__":
    env = defense_v0.env(terrain=TERRAIN, max_cycles=EPISODE_MAX_LENGTH, max_distance=MAX_DISTANCE )
    env.reset()
    args = Args(env)
    runner = runner(env, args)
    if len(sys.argv) == 1:
        runner.run()
    else:
        runner.eval(sys.argv[1])

def main(argv):
    env = defense_v0.env(terrain=TERRAIN, max_cycles=EPISODE_MAX_LENGTH, max_distance=MAX_DISTANCE )
    env.reset()
    args = Args(env)
    runner = runner(env, args)
    try:
        opts, args = getopt.getopt(argv,"hl:e:",["load_model=","eval_model="])
    except getopt.GetoptError:
        print('error')
    if len(argv) == 0:
        runner.run()
    for opt, arg in opts:
        if opt == '-h':
            print('dqn.py')
            print ('dqn.py -l <model_folder_to_load>')
            print('OR')
            print('dqn.py  -e <model_folder_to_eval> <folder_to_save_metrics>')
            sys.exit()
        elif opt in ("-l", "--load_model"):
            runner.args.MODEL_TO_LOAD = arg
            runner.run()
        elif opt in ("-e", "--eval_model"):
            runner.eval(arg)



if __name__ == "__main__":
    main(sys.argv[1:])