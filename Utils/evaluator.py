
import pickle
import pandas as pd
import os, sys
import torch 
os.chdir('/home/jack/Documents/ERM/Master thesis/tfe')
sys.path.insert(0, '/home/jack/Documents/ERM/Master thesis/tfe')
from env import defense_v0

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
dev = "cpu"  #if cpu faster than gpu

#importing RL algorithm
from RL_algorithms import dqn_defense as dqn
from RL_algorithms.dqn_defense import Params, Metrics #for the pickle to work in the algorithm's model loading



#environment constants
EPISODE_MAX_LENGTH = 200
MAX_DISTANCE = 5


class Results:
    def __init__(self,algorithm, model_directory, environments_directory, nb_episodes, EPISODE_MAX_LENGTH, MAX_DISTANCE):
        self.nb_episodes = nb_episodes
        self.metrics = []
        self.model_directory = model_directory
        self.algorithm = algorithm
        self.environments_directory = environments_directory
        self.nb_episodes = nb_episodes
        self.EPISODE_MAX_LENGTH = EPISODE_MAX_LENGTH
        self.MAX_DISTANCE = MAX_DISTANCE

def test_model(algorithm, adversary_tactic, model_directory, environments_directory, nb_episodes, EPISODE_MAX_LENGTH, MAX_DISTANCE, save_directory = None): #test all environments of a directory with a given model

    results ={'algorithm': algorithm, 'model': model_directory, 'environments': environments_directory, 'nb episodes': nb_episodes, 'episode max length': EPISODE_MAX_LENGTH, 'max distance':  MAX_DISTANCE, 'envs': {}}

    for filename in os.listdir(f'env/terrains/{environments_directory}'):
        f = os.path.join(environments_directory, filename)
        
        #set up environmnent
        env = defense_v0.env(terrain=f'{environments_directory}/{filename[0:-4]}', max_cycles=EPISODE_MAX_LENGTH, max_distance=MAX_DISTANCE )
        env.reset()

        if algorithm == 'dqn':
            args = dqn.Args(env)
            args.ADVERSARY_TACTIC = adversary_tactic
            args.TENSORBOARD = False
            runner = dqn.Runner(env, args)

        res = runner.eval(model_directory, nb_episodes, False, False)
        res.env = f'{filename[0:-4]}'
        results['envs'][res.env] = { 'steps': res.nb_steps, 'rewards': res.rewards_buffer, 'wins': res.wins}
    
    if save_directory !=None:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        model_run = model_directory.split('/')[-1]
        
        with open(f'{save_directory}/results_{algorithm}_{environments_directory}_{model_run}.bin',"wb") as f:
            pickle.dump(results, f)
        
        
    return results



# plot results

#env = defense_v0.env(terrain='benchmark_10x10_2v2', max_cycles=20, max_distance=5 )
#env.render()
#print('ok')


if __name__ == "__main__":
    
    algo = 'dqn'
    model_dir = 'defense_params_dqn/benchmarking/071732mai2022step_0'
    env_dir = 'testgenlib'
    nb_ep = 3
    adversary_type = 'random'
    #model_dir = 'random'
    out = test_model(algo, adversary_type,  model_dir, env_dir, nb_ep, EPISODE_MAX_LENGTH, MAX_DISTANCE, 'evalstestt1')