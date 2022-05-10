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

def plot_eval_folder(resultfile, plot_folder, base_env, name):
    sim_index, reward, steps, wins, nb_episodes = compute_results(resultfile)
    folder_name = f'{plot_folder}/{base_env}_{name}'
    if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    x, y = zip(*sorted(zip(sim_index, reward)))
    plt.plot(x, y)
    plt.legend()
    plt.xlabel('Similarity Index')
    plt.ylabel('Reward /agent')
    plt.title(f'Reward for agent trained on {base_env} (averaged over {nb_episodes} episodes)')
    plt.show(block=False)
    plt.pause(10)
    plt.savefig(f'{folder_name}/reward_plot.png')
    plt.close()

    x, y = zip(*sorted(zip(sim_index, steps)))
    plt.plot(x, y)
    plt.legend()
    plt.xlabel('Similarity Index')
    plt.ylabel('Steps /episode')
    plt.title(f'Steps per episode for agent trained on {base_env} (averaged over {nb_episodes} episodes)')
    plt.show(block=False)
    plt.pause(10)
    plt.savefig(f'{folder_name}/steps_plot.png')
    plt.close()

    x, y = zip(*sorted(zip(sim_index, wins)))
    plt.plot(x, y)
    plt.legend()
    plt.xlabel('Similarity Index')
    plt.ylabel('Win rate')
    plt.title(f'Win rate for agent trained on {base_env} (averaged over {nb_episodes} episodes)')
    plt.show(block=False)
    plt.pause(10)
    plt.savefig(f'{folder_name}/wins_plot.png')
    plt.close()



def compute_results(result_file):

    with open(result_file,"rb") as f:
        results = pickle.load(f)
    plt.close('all')

    nb_episodes = results['nb episodes']
    nb_env = len(results['envs'].keys())
    mean_reward_per_env = {}
    mean_steps_per_env = {}
    mean_wins_per_env = {}
    sim_per_env = {}
    mean_reward = 0
    mean_steps = 0
    mean_wins = 0
    reward = []
    steps = []
    wins = []
    sim_index = []
    print('---          ---          ---')
    for env, metrics in results['envs'].items():
        sim_index.append(float(env.split('simIndex_')[-1].split('_')[0]))
        reward.append(np.mean(metrics['rewards']))
        steps.append(np.mean(metrics['steps']))
        wins.append( np.sum(metrics['wins'])/nb_episodes)
        mean_reward_per_env[env] = np.mean(metrics['rewards'])
        mean_steps_per_env[env] = np.mean(metrics['steps'])
        mean_wins_per_env[env] = np.sum(metrics['wins'])/nb_episodes
        #print(f'Environment : {env}\nMean reward: {mean_reward_per_env[env]} | Mean nb steps: {mean_steps_per_env[env]} | Win : {100*mean_wins_per_env[env]}%')
        mean_reward += np.mean(metrics['rewards']) 
        mean_steps += np.mean(metrics['steps'])
        mean_wins += np.sum(metrics['wins'])/nb_episodes
    mean_reward = mean_reward/nb_env
    mean_steps = mean_steps/nb_env
    mean_wins = mean_wins/nb_env
    print('---          ---          ---')
    print(f'Means over whole environment folder:\nReward {mean_reward} | Steps : {mean_steps} | Wins : {100*mean_wins}%')
    print('---          ---          ---')
    
    return sim_index, reward, steps, wins, nb_episodes


def plot_nn(model, algo):
    from neuralnet_visualize import visualize as nnviz
    network = nnviz.visualizer(title=f"{algo} Neural Network")
    network.from_pytorch(model)
    network.visualize()

def plot_tensorboard_log(json_file, data_type, algo, env, name, size=(10,5)):
    filter_factor = 200
    f = open(json_file)
    data = json.load(f)
    step = np.asarray([element[1] for element in data])
    data = np.asarray([element[2] for element in data])

    fig = plt.figure(figsize=size)
    plt.plot(step, data)
    plt.xlabel('Step')
    if data_type == 'loss':
        if algo == 'QMIX':
            plt.yscale("log") 
        plt.ylabel('Loss /agent /step')
        
    plt.plot(step, uniform_filter1d(data, size=filter_factor), 'C2', label='Smoothed value')
    plt.ylabel('Reward /agent')
    plt.legend()
    plt.title(f'{algo} {data_type} on the {env} environment')

    fig.savefig(f'figures/{name}.png', bbox_inches='tight', dpi=300)
    plt.show(block=False)
    plt.pause(1)
    f.close()




if __name__ == "__main__":

    test = 'evalstestt1/results_dqn_testgenlib_071732mai2022step_0.bin'
    compute_results(test)
    plot_eval(test)
    plot_eval_folder(test, 'testfolderplot', 'benchmark10_1v1', 'firstjet')
    qmixloss_benchmark = 'toplot/Apr12_16-18-44_qmix_log.json'
    plot_tensorboard_log(qmixloss_benchmark, 'loss', 'QMIX', 'benchmark', 'benchmark/loss_qmix_benchmark')

    qmixreward_benchmark = 'toplot/Apr12_16-18-44_milleniumfalcon_qmix_log1.json'
    plot_tensorboard_log(qmixreward_benchmark, 'reward', 'QMIX', 'benchmark', 'benchmark/reward_qmix_benchmark')

    dqnloss_benchmark = 'toplot/avril18_21-24-37_DeathStar_dqn_loss.json'
    plot_tensorboard_log(dqnloss_benchmark, 'loss', 'DQN', 'benchmark', 'benchmark/loss_dqn_benchmark')

    dqnrew_benchmark = 'toplot/avril18_21-24-37_DeathStar_dqn_rew.json'
    plot_tensorboard_log(dqnrew_benchmark, 'reward', 'DQN', 'benchmark', 'benchmark/reward_dqn_benchmark')