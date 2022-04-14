from collections import deque
import numpy as np
from datetime import datetime
import re
import torch 
import os, sys


class Constants:
    def __init__(self):
        self.EPISODE_MAX_LENGTH = 200
        self.MAX_DISTANCE = 5

#to log results
class Metrics:
    def __init__(self, nb_episodes):
        self.nb_episodes = nb_episodes
        self.rewards_buffer = deque(maxlen=nb_episodes)
        self.wins = deque(maxlen=nb_episodes)
        self.nb_steps = deque(maxlen=nb_episodes)
        self.env = ''

class Buffers:
    def __init__(self, env, args, agents, device):
        
        self.observation = dict()
        self.observation_next = dict()
        self.hidden_state = dict()
        self.hidden_state_next = dict()
        self.episode_reward = 0.0
        self.nb_transitions = 0
        self.replay_buffer = deque(maxlen=args.BUFFER_SIZE)
        self.rew_buffer = deque([0.0], maxlen=args.REW_BUFFER_SIZE)
        self.loss_buffer = deque([0.0], maxlen=args.REW_BUFFER_SIZE)

        for agent in agents:
            self.observation[agent] = env.observe(agent)
            self.hidden_state[agent] = torch.zeros(args.dim_L2_agents_net, device=device).unsqueeze(0)
       

class Params:
    def __init__(self, step):
        self.step = step
        
def get_device():
    if torch.cuda.is_available():
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    
    return torch.device(dev) 

def mask_array(array, mask):
    int = np.ma.compressed(np.ma.masked_where(mask==0, array) )

    return np.ma.compressed(np.ma.masked_where(mask==0, array) )


