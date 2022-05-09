import torch
import torch.nn as nn
import numpy as np
import os
import sys, getopt

#importing the defense environment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)
from env import defense_v0



    
from Utils.helper import Buffers, Params, Metrics, Constants, mask_array, get_device
from Utils.params import VDNArgs as Args
from Utils.agent import AgentNet

from runners import Runner

device = get_device()

#environment constants
constants = Constants()
TERRAIN = 'benchmark_10x10_2v2'

MODEL_DIR = 'defense_params_vdn'
RUN_NAME = 'benchmarking'
ADVERSARY_TACTIC = 'random'
ENV_SIZE = 10 #todo : calculate 


class VDNMixer(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.to(device)
        #params
        self.args = args
        total_state_dim = 0
        if self.args.COMMON_AGENTS_NETWORK:
            self.agents_net = AgentNet(self.args)
            for agent in self.args.blue_agents:
                #self.agent_nets[agent] = self.agents_net
                total_state_dim += np.prod(env.observation_space(agent).spaces['obs'].shape)               

        else:
            for agent in args.blue_agents:
                self.agents_nets = dict()
                self.agents_nets[agent] = AgentNet(args)
                total_state_dim += np.prod(env.observation_space(agent).shape)


        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.args.LEARNING_RATE) #adding all parameters to optimizer at once
        
    def forward(self, obs, Qin_t):
        
        
        return Qin_t.sum(1)
        
    def get_Q_values(self, agent, obs,hidden_state):
        obs_t = torch.as_tensor(obs['obs'], dtype=torch.float32,device=device)
        q_values, hidden_state_next = self.agents_nets[agent](obs_t.unsqueeze(0), hidden_state)
        return q_values, hidden_state_next

    def get_Q_max(self, masked_q_values, obs, all_q_values=None):
        if len(masked_q_values) == 0:
            return -1, torch.tensor([0],device=device)
        max_q_index = torch.argmax(masked_q_values, dim=-1).item()
        max_q = masked_q_values[max_q_index] 
        if all_q_values != None: #only in case a mask is given
            max_q_index = ((all_q_values == max_q.item()).cpu() * (obs['action_mask'] == 1)).nonzero(as_tuple=True)[-1][0].item()
        return max_q_index, max_q


    def act(self, agent, obs, hidden_state):
        with torch.no_grad():
            q_values, hidden_state_next = self.get_Q_values(agent, obs, hidden_state)
            #taking only masked q values to choose action to take
            masked_q_val = torch.masked_select(q_values, torch.as_tensor(obs['action_mask'], dtype=torch.bool,device=device))
            if masked_q_val.numel() == 0:
                return None, hidden_state_next
            action, _ = self.get_Q_max(masked_q_val, obs, q_values)
            return action, hidden_state_next
    
        
        


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
        runner = Runner(env, args_runner, TERRAIN)
        runner.run()
    for opt, arg in opts:
        if opt == '-h':
            print('vdn.py')
            print ('vdn.py -l <model_folder_to_load>')
            print('OR')
            print('vdn.py  -e <model_folder_to_eval>')
            sys.exit()
        elif opt in ('-a', "--load_adversary"):
            args_runner.ADVERSARY_MODEL = arg
            args_runner.ADVERSARY_TACTIC = 'vdn'
        elif opt in ("-l", "--load_model"):
            args_runner.MODEL_TO_LOAD = arg
            runner = Runner(env, args_runner, TERRAIN)
            runner.run()
        elif opt in ("-e", "--eval_model"):
            runner = Runner(env, args_runner, TERRAIN)
            runner.eval(arg)



if __name__ == "__main__":
    main(sys.argv[1:])
        

