
import pickle
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
import dqn_defense
from dqn_defense import Params, Metrics #for the pickle to work in the algorithm's model loading



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

def test_model(algorithm, model_directory, environments_directory, nb_episodes, EPISODE_MAX_LENGTH, MAX_DISTANCE, save_directory = None): #test all environments of a directory with a given model

    results = Results(algorithm, model_directory, environments_directory, nb_episodes, EPISODE_MAX_LENGTH, MAX_DISTANCE)
    
    for filename in os.listdir(f'env/terrains/{environments_directory}'):
        f = os.path.join(environments_directory, filename)
        
        #set up environmnent
        env = defense_v0.env(terrain=f'{environments_directory}/{filename[0:-4]}', max_cycles=EPISODE_MAX_LENGTH, max_distance=MAX_DISTANCE )
        env.reset()

        if algorithm == 'dqn':
            args = dqn_defense.Args(env)
            runner = dqn_defense.runner(env, args)

        res = runner.eval(model_directory, nb_episodes, False)
        res.env = f'{filename[0:-4]}'
        results.metrics.append(res)
    
    if save_directory !=None:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        model_run = model_directory.split('/')[-1]
        with open(f'{save_directory}/results_{algorithm}_{environments_directory}_{model_run}.bin',"wb") as f:
            pickle.dump(results, f)
    return results

algo = 'dqn'
model_dir = 'defense_params_dqn/111943avril2022step_0'
env_dir = 'testt'
nb_ep = 10

out = test_model(algo, model_dir, env_dir, nb_ep, EPISODE_MAX_LENGTH, MAX_DISTANCE, 'evals')


# plot results
result_file = 'evals/results_dqn_testt_111943avril2022step_0.bin'
with open(result_file,"rb") as f:
    results = pickle.load(f)

print('ok')
