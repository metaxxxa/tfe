import torch
import numpy as np 
import re
from datetime import datetime
#parameters

class DQNArgs:
    def __init__(self, env):
            
        self.BUFFER_SIZE = 2000
        self.LEARNING_RATE = 1e-4
        self.MIN_BUFFER_LENGTH = 300
        self.BATCH_SIZE = 64
        self.GAMMA = 0.95
        self.EPSILON_START = 1
        self.EPSILON_END = 0.01
        self.EPSILON_DECAY = 20000
        self.SYNC_TARGET_FRAMES = 200
        self.STOP_TRAINING = self.EPSILON_DECAY*2
        self.USE_PER = False
        self.EPSILON_PER = 0.0001
        self.ALPHA_PER = 0.6
        self.B_PER = 0.4
        self.DOUBLE_DQN = False
        self.CONVOLUTIONAL_INPUT = True
        self.CONV_OUT_CHANNELS = 16
        self.KERNEL_SIZE = 3
        self.PADDING = 1
        #visualization parameters
        self.PRINT_LOGS = False
        self.VISUALIZE_WHEN_LEARNED = True
        self.VISUALIZE_AFTER = 5000000
        self.VISUALIZE = False
        self.WAIT_BETWEEN_STEPS = 0.01
        self.GREEDY = True
        #logging
        self.TENSORBOARD = True
        self.REW_BUFFER_SIZE = 10000
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
        self.hidden_layer2_dim = 64
        #environment specific parameters calculation
        self.WINNING_REWARD = 1
        self.LOSING_REWARD = -1
        self.TEAM_TO_TRAIN = 'blue'
        self.OPPOSING_TEAM = 'red'
        self.ADVERSARY_TACTIC = 'random'
        self.ADVERSARY_MODEL = ''
        self.params(env)

    def params(self, env):  #environment specific parameters calculation
        
        self.blue_agents = [key for key in env.agents if re.match(rf'^{self.TEAM_TO_TRAIN}',key)]
        self.all_agents = env.agents
        self.red_agents = [key for key in env.agents if re.match(rf'^{self.OPPOSING_TEAM}',key)]
        self.n_blue_agents = len(self.blue_agents)
        agent = self.blue_agents[0]
        self.nb_inputs_agent = np.prod(env.observation_space(agent).spaces['obs'].shape)
        self.observations_dim = np.prod(env.observation_space(agent).spaces['obs'].shape)
        self.n_actions = env.action_space(agent).n
    def log_params(self, terrain, writer=None):
        hparams = {'envparam/terrrain': terrain,'Adversary tactic':self.ADVERSARY_TACTIC , 'Algorithm': 'DQN' , 'Learning rate': self.LEARNING_RATE, 'Batch size': self.BATCH_SIZE, 'Buffer size': self.BUFFER_SIZE, 'Min buffer length': self.MIN_BUFFER_LENGTH, '/gamma': self.GAMMA, 'Epsilon range': f'{self.EPSILON_START} - {self.EPSILON_END}', 'Epsilon decay': self.EPSILON_DECAY, 'Synchronisation rate': self.SYNC_TARGET_FRAMES, 'Timestamp': int(datetime.timestamp(datetime.now()) - datetime.timestamp(datetime(2022, 2, 1, 11, 26, 31,0)))}
        metric_dict = { 'hparam/dim L1 agent net': self.dim_L1_agents_net, 'hparam/dim L2 agent net': self.dim_L2_agents_net}
        if self.TENSORBOARD:
            writer.add_hparams(hparams, metric_dict)


class QMIXArgs:
    def __init__(self, env):
            
        self.BUFFER_SIZE = 2000
        self.REW_BUFFER_SIZE = 1000
        self.LEARNING_RATE = 1e-4
        self.MIN_BUFFER_LENGTH = 300
        self.BATCH_SIZE = 64
        self.GAMMA = 0.95
        self.EPSILON_START = 1
        self.EPSILON_END = 0.01
        self.EPSILON_DECAY = 100000
        self.SYNC_TARGET_FRAMES = 200
        self.STOP_TRAINING = self.EPSILON_DECAY*5
        self.USE_PER = False
        self.EPSILON_PER = 0.01
        self.ALPHA_PER = 0.6
        self.B_PER = 0.4
        #visualization parameters
        self.VISUALIZE_WHEN_LEARNED = True
        self.VISUALIZE_AFTER = 20000000
        self.VISUALIZE = False
        self.WAIT_BETWEEN_STEPS = 0.0001
        self.GREEDY = True
        #logging
        self.TENSORBOARD = True
        #saving models
        self.ITER_START_STEP = 0 #when starting training with an already trained model, 0 by default without model to load
        self.MODEL_TO_LOAD = ''
        self.SAVE_CYCLE = 100000
        self.MODEL_DIR = 'defense_params_qmix'
        self.RUN_NAME = 'benchmarking'
        #agent network parameters
        self.COMMON_AGENTS_NETWORK = True
        self.dim_L1_agents_net = 32
        self.dim_L2_agents_net = 32
        #mixing network parameters
        self.mixer_hidden_dim = 32
        self.mixer_hidden_dim2 = 32
        #environment specific parameters calculation
        self.WINNING_REWARD = 1
        self.LOSING_REWARD = -1
        self.TEAM_TO_TRAIN = 'blue'
        self.OPPOSING_TEAM = 'red'
        self.ADVERSARY_TACTIC = 'random'
        self.params(env)

    def params(self, env):  #environment specific parameters calculation
        
        self.blue_agents = [key for key in env.agents if re.match(rf'^{self.TEAM_TO_TRAIN}',key)]
        self.red_agents = [key for key in env.agents if re.match(rf'^{self.OPPOSING_TEAM}',key)]
        self.all_agents = env.agents
        self.opposing_agents = [key for key in env.agents if re.match(rf'^{self.OPPOSING_TEAM}',key)]
        self.n_blue_agents = len(self.blue_agents)
        agent = self.blue_agents[0]
        self.nb_inputs_agent = np.prod(env.observation_space(agent).spaces['obs'].shape)
        self.observations_dim = np.prod(env.observation_space(agent).spaces['obs'].shape)
        self.n_actions = env.action_space(agent).n
    def log_params(self, writer, algorithm, terrain):
        hparams = {'envparam/terrrain': terrain, 'Adversary tactic' : self.ADVERSARY_TACTIC, 'Algorithm': algorithm , 'Learning rate': self.LEARNING_RATE, 'Batch size': self.BATCH_SIZE, 'Buffer size': self.BUFFER_SIZE, 'Min buffer length': self.MIN_BUFFER_LENGTH, '/gamma': self.GAMMA, 'Epsilon range': f'{self.EPSILON_START} - {self.EPSILON_END}', 'Epsilon decay': self.EPSILON_DECAY, 'Synchronisation rate': self.SYNC_TARGET_FRAMES, 'Timestamp': int(datetime.timestamp(datetime.now()) - datetime.timestamp(datetime(2022, 2, 1, 11, 26, 31,0))), 'Common agent network': int(self.COMMON_AGENTS_NETWORK)}
        metric_dict = { 'hparam/dim L1 agent net': self.dim_L1_agents_net, 'hparam/dim L2 agent net': self.dim_L2_agents_net, 'hparam/mixer hidden dim 1': self.mixer_hidden_dim, 'hparam/mixer hidden dim 2': self.mixer_hidden_dim2}
        writer.add_hparams(hparams, metric_dict)
