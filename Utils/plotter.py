from asyncio import events
from cgi import test
import os, sys
import csv
import json
from termios import XTABS
import numpy as np
from scipy.ndimage.filters import uniform_filter1d

os.chdir('/home/jack/Documents/ERM/Master thesis/tfe')
sys.path.insert(0, '/home/jack/Documents/ERM/Master thesis/tfe')
import matplotlib.pyplot as plt
plt.style.use('Utils/plot_style.txt')
import pickle

# plot results
result_file = 'evals/results_dqn_testt_111943avril2022step_0.bin'
def plot_eval(result_file):
    with open(result_file,"rb") as f:
        results = pickle.load(f)

    plt.close('all')

    nb_episodes = results['nb episodes']
    x = list(range(1,nb_episodes+1))
    for env, metrics in results['envs'].items():
        plt.plot(x, metrics['rewards'], label=env)
    plt.legend()
    plt.xlabel('Episode n°')
    plt.ylabel('Reward /agent')
    plt.title(f'Reward per agent for each environment ({nb_episodes} episodes)')
    plt.show(block=False)
    plt.pause(10)

def compute_results(result_file):

    with open(result_file,"rb") as f:
        results = pickle.load(f)
    plt.close('all')

    nb_episodes = results['nb episodes']
    nb_env = len(results['envs'].keys())
    mean_reward_per_env = {}
    mean_steps_per_env = {}
    mean_wins_per_env = {}
    mean_reward = 0
    mean_steps = 0
    mean_wins = 0
    print('---          ---          ---')
    for env, metrics in results['envs'].items():
        mean_reward_per_env[env] = np.mean(metrics['rewards'])
        mean_steps_per_env[env] = np.mean(metrics['steps'])
        mean_wins_per_env[env] = np.sum(metrics['wins'])/nb_episodes
        print(f'Environment : {env}\nMean reward: {mean_reward_per_env[env]} | Mean nb steps: {mean_steps_per_env[env]} | Win : {100*mean_wins_per_env[env]}%')
        mean_reward += np.mean(metrics['rewards']) 
        mean_steps += np.mean(metrics['steps'])
        mean_wins += np.sum(metrics['wins'])/nb_episodes
    mean_reward = mean_reward/nb_env
    mean_steps = mean_steps/nb_env
    mean_wins = mean_wins/nb_env
    print('---          ---          ---')
    print(f'Means over whole environment folder:\nReward {mean_reward} | Steps : {mean_steps} | Wins : {100*mean_wins}%')
    print('---          ---          ---')
    


def plot_nn(model, algo):
    from neuralnet_visualize import visualize as nnviz
    network = nnviz.visualizer(title=f"{algo} Neural Network")
    network.from_pytorch(model)
    network.visualize()

def plot_tensorboard_log(json_file, data_type, algo, env, name, size=(10,5)):
    filter_factor = 13
    f = open(json_file)
    data = json.load(f)
    step = np.asarray([element[1] for element in data])
    data = np.asarray([element[2] for element in data])

    fig = plt.figure(figsize=size)
    plt.plot(step, data)
    plt.xlabel('Step')
    if data_type == 'loss':
        plt.yscale("log") 
        plt.ylabel('Loss /agent /step')
    elif data_type == 'reward':
        plt.plot(step, uniform_filter1d(data, size=filter_factor), 'C2', label='Smoothed value')
        plt.ylabel('Reward /agent')
        plt.legend()
    plt.title(f'{algo} {data_type} on the {env} environment')

    fig.savefig(f'figures/{name}.png', bbox_inches='tight', dpi=300)
    plt.show(block=False)
    plt.pause(1)
    f.close()

test = 'evals/results_dqn_testt_random.bin'
compute_results(test)

qmixloss_benchmark = 'toplot/Apr12_16-18-44_qmix_log.json'
plot_tensorboard_log(qmixloss_benchmark, 'loss', 'QMIX', 'benchmark', 'benchmark/loss_qmix_benchmark')

qmixreward_benchmark = 'toplot/Apr12_16-18-44_milleniumfalcon_qmix_log1.json'
plot_tensorboard_log(qmixreward_benchmark, 'reward', 'QMIX', 'benchmark', 'benchmark/reward_qmix_benchmark')