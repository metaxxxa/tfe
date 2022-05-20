
import pickle
from pydoc import plain
import math
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
from RL_algorithms import q_mix_defense as qmix
from RL_algorithms import vdn_defense as vdn
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
    nb_env = len(os.listdir(f'env/terrains/{environments_directory}'))
    i = 0
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
        elif algorithm == 'qmix':
            args = qmix.Args(env)
            args.ADVERSARY_TACTIC = adversary_tactic
            args.TENSORBOARD = False
            runner = qmix.Runner(env, args)
        elif algorithm == 'vdn':
            args = qmix.Args(env)
            args.ADVERSARY_TACTIC = adversary_tactic
            args.TENSORBOARD = False
            runner = qmix.Runner(env, args)

        res = runner.eval(model_directory, nb_episodes, False, False)
        res.env = f'{filename[0:-4]}'
        results['envs'][res.env] = { 'steps': res.nb_steps, 'rewards': res.rewards_buffer, 'wins': res.wins}
        i+=1
        if i % math.ceil(nb_env/100) == 0:
            progress = i/nb_env
            print(f'{progress} %')
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
    plaindqn = 'results/plaindqn/100238mai2022step_300000'
    plaindqnconv = 'results/plaindqnconv/102226mai2022step_300000/'
    ddqn_conv = 'results/dqn_conv_double/122032mai2022step_300000/'
    ddqn_conv_annealing = 'results/dqn_conv_double_PER_annealing/131520mai2022step_100000'
    ddqn_conv_annealing = 'results/dqn_conv_double_PER_annealing/dqn_enhanced/190322May2022step_300000'
    CE_dqn_plain = 'results/change_env/plain/dqn/170118mai2022step_220000/'
    CE_dqn_features = 'results/change_env/with_features/dqn/180810May2022step_300000'
    env_dir = 'eval_lib_fixedObs3'
    small_evallib_lines = 'eval_lib_fixedObs_lines'
    final_eval_lib = 'eval_lib'
    final_eval_lib_lines = 'eval_lib_lines'

    nb_ep = 50
    adversary_random = 'random'
    #model_dir = 'random'
  #  out = test_model(algo, adversary_random,  plaindqn, small_evallib_lines, nb_ep, EPISODE_MAX_LENGTH, MAX_DISTANCE, 'results/plaindqn/small_evallib_lines')
 #   out = test_model(algo, adversary_random,  plaindqnconv, small_evallib_lines, nb_ep, EPISODE_MAX_LENGTH, MAX_DISTANCE, 'results/plaindqnconv/small_evallib_lines')
 #   out = test_model(algo, adversary_random,  plaindqnconv, final_eval_lib_lines, nb_ep, EPISODE_MAX_LENGTH, MAX_DISTANCE, 'results/plaindqnconv/evallib_lines')
  #  out = test_model(algo, adversary_random,  CE_dqn_plain, final_eval_lib_lines, nb_ep, EPISODE_MAX_LENGTH, MAX_DISTANCE, 'results/change_env/plain/dqn/dqn_plain_evallib_lines')
    out = test_model(algo, adversary_random,  ddqn_conv_annealing, final_eval_lib_lines, nb_ep, EPISODE_MAX_LENGTH, MAX_DISTANCE, 'results/dqn_conv_double_PER_annealing/dqn_enhanced/evallib_lines')
    #out = test_model(algo, adversary_random,  CE_dqn_features, final_eval_lib_lines, nb_ep, EPISODE_MAX_LENGTH, MAX_DISTANCE, 'results/change_env/with_features/dqn/evallib_lines')

