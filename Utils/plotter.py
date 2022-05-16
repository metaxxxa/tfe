from asyncio import events
from cgi import test
import os, sys
import math
import json
from termios import XTABS
import numpy as np
from scipy.ndimage.filters import uniform_filter1d


os.chdir('/home/jack/Documents/ERM/Master thesis/tfe')
sys.path.insert(0, '/home/jack/Documents/ERM/Master thesis/tfe')
import matplotlib.pyplot as plt
plt.style.use('Utils/plot_style.txt')
import pickle
from Utils.helper import similarity_index
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:cyan']
# plot results
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

def plot_eval_folder(resultfile, plot_folder, base_env, name, figure_name, names=[], size=(10,5)):
    filter_factor = 1


    
    folder_name = f'{plot_folder}/{base_env}_{name}'
    if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    i= 0
    
    for file in resultfile:
        similarity_index, reward, steps, wins, nb_episodes, reward_v, steps_v, wins_v = compute_results(file)

        x, y = zip(*sorted(zip(similarity_index, reward)))
        _, var = zip(*sorted(zip(similarity_index, reward_v)))
        
        if len(names) > 1:
            plt.plot(x,  uniform_filter1d(y, size=filter_factor), color= COLORS[i], label=names[i])
        else:
            plt.plot(x,  uniform_filter1d(y, size=filter_factor), color= COLORS[i])
        plt.errorbar(x, uniform_filter1d(y, size=filter_factor), yerr=var, fmt='o', ecolor=COLORS[i], color='white')
        i += 1
    plt.legend()
    plt.xlabel('Similarity Index')
    plt.ylabel('Reward /agent')
    plt.title(f'Reward for {figure_name} trained on {base_env}')
    plt.show(block=False)
    plt.pause(1)
    
    plt.savefig(f'{folder_name}/reward_plot.png')
    plt.close()
    i= 0
    for file in resultfile:
        similarity_index, reward, steps, wins, nb_episodes, reward_v, steps_v, wins_v = compute_results(file)

        x, y = zip(*sorted(zip(similarity_index, steps)))
        _, var = zip(*sorted(zip(similarity_index, steps_v)))
        if len(names) != 0:
            plt.plot(x,  uniform_filter1d(y, size=filter_factor), color=COLORS[i], label=names[i])
        else:
            plt.plot(x,  uniform_filter1d(y, size=filter_factor), color=COLORS[i])
        plt.errorbar(x, uniform_filter1d(y, size=filter_factor), yerr=var, fmt='o', ecolor=COLORS[i], color='white')
        i += 1
    plt.legend()
    plt.xlabel('Similarity Index')
    plt.ylabel('Steps /episode')
    plt.title(f'Steps per episode for {figure_name} trained on {base_env}')
    plt.show(block=False)
    plt.pause(1)
    plt.savefig(f'{folder_name}/steps_plot.png')
    plt.close()
    i= 0
    for file in resultfile:
        similarity_index, reward, steps, wins, nb_episodes, reward_v, steps_v, wins_v = compute_results(file)

        sim_index = similarity_index
        w = wins
        x, y = zip(*sorted(zip(similarity_index, wins)))
        _, var = zip(*sorted(zip(similarity_index, wins_v)))
        if len(names) != 0:
            plt.plot(x,  uniform_filter1d(y, size=filter_factor), label=names[i], color=COLORS[i])
        else:
            plt.plot(x,  uniform_filter1d(y, size=filter_factor), color=COLORS[i])
        plt.errorbar(x, uniform_filter1d(y, size=filter_factor), yerr=var, fmt='o', ecolor=COLORS[i], color='white')
        i +=1 
    plt.legend()
    plt.xlabel('Similarity Index')
    plt.ylabel('Win rate')
    plt.title(f'Win rate for {figure_name} trained on {base_env}')
    plt.show(block=False)
    plt.pause(1)
    plt.savefig(f'{folder_name}/wins_plot.png')
    plt.close()



def compute_results(result_file):

    with open(result_file,"rb") as f:
        results = pickle.load(f)
    f.close()

    nb_episodes = results['nb episodes']
    nb_env = len(results['envs'].keys())
    mean_reward_per_env = {}
    mean_steps_per_env = {}
    mean_wins_per_env = {}
    mean_reward = 0
    mean_steps = 0
    mean_wins = 0
    reward = []
    steps = []
    wins = []
    reward_var = []
    steps_var = []
    wins_var = []
    sim_index = []
    print('---          ---          ---')
    for env, metrics in results['envs'].items():
        sim_index.append(float(env.split('simIndex_')[-1].split('_')[0]))
        reward.append(np.mean(metrics['rewards']))
        reward_var.append(np.std(metrics['rewards']))
        steps.append(np.mean(metrics['steps']))
        steps_var.append(np.std(metrics['steps']))
        wins.append( np.mean(metrics['wins']))
        wins_var.append( np.std(metrics['wins']))
        mean_reward_per_env[env] = np.mean(metrics['rewards'])
        mean_steps_per_env[env] = np.mean(metrics['steps'])
        mean_wins_per_env[env] = np.mean(metrics['wins'])
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
    
    return sim_index, reward, steps, wins, nb_episodes, reward_var, steps_var, wins_var


def plot_nn(model, algo):
    from neuralnet_visualize import visualize as nnviz
    network = nnviz.visualizer(title=f"{algo} Neural Network")
    network.from_pytorch(model)
    network.visualize()

def plot_tensorboard_log(json_file, data_type, algo, env, name, size=(10,5)):
    filter_factor = 250
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

def plot_tensorboard_compare(json_files,names, data_type, algo, env, figure_name, size=(10,5)):
    filter_factor = 200
    data = {}
    step = {}
    i = 0
    for json_file in json_files:
        f = open(json_file)
        data[names[i]] = json.load(f)
        
        step[names[i]] = np.asarray([element[1] for element in data[names[i]]])
        data[names[i]] = np.asarray([element[2] for element in data[names[i]]])
        i += 1
    
    fig = plt.figure(figsize=size)
    i = 0
    for name in names:
        #plt.plot(step[name], data[name], label=name)
        plt.plot(step[name], uniform_filter1d(data[name], size=filter_factor), label=name, color=COLORS[i])
        new_step, std_up, std_down = compute_variance(step[name], uniform_filter1d(data[name], size=filter_factor), 20)
        plt.fill_between(new_step, std_down, std_up, color=COLORS[i] , alpha=0.1)

        i+=1
    plt.xlabel('Step')
    if data_type == 'loss':
        if algo == 'QMIX':
            plt.yscale("log") 
        plt.ylabel('Loss /agent /step')
        
    elif data_type == 'reward':
        plt.ylabel('Reward /agent /episode')
    elif data_type == 'steps /episode':
        plt.ylabel('Steps /episode')
    elif data_type == 'win rate':
        plt.axhline(0.8, c='r', ls=':')
        plt.ylabel('Win rate')
    plt.legend()
    plt.title(f'{algo.upper()} {data_type} on the {env} environment')

    fig.savefig(f'figures/{figure_name}.png', bbox_inches='tight', dpi=300)
    plt.show(block=False)
    plt.pause(1)
    f.close()

def compute_variance(steps, data, window=15):
    new_step = []
    std_up = []
    std_down = []
    Z = 10 #80 % confidence interval
    for i in range(int(np.floor((len(data)-1)/window))+1):
        segment = data[i*window:((i+1)*window -1)]
        ci = np.std(segment)*5
        std_up.append(data[i*window+int(np.floor((len(segment)-1)/2))] +ci )
        std_down.append(data[i*window+int(np.floor((len(segment)-1)/2))] - ci)
        new_step.append((steps[i*window+int(np.floor((len(segment)-1)/2))]))
    return new_step, std_up, std_down



if __name__ == "__main__":

    plain_dqn = 'eval_plaindqn3/results_dqn_eval_lib_fixedObs3_100238mai2022step_300000.bin'
    plain_dqn_conv = 'eval_plaindqnconv/results_dqn_eval_lib_fixedObs3_.bin'
    #plain_dqn = 'test/results_dqn_eval_lib_fixedObs3_100238mai2022step_300000.bin'

    x = 'eval_plaindqn/results_dqn_eval_lib_fixedObs2_100238mai2022step_300000.bin'
    plot_eval_folder([plain_dqn, plain_dqn_conv], 'results/plaindqn', 'benchmark10_1v1', 'test_trained_sameEnv', 'Plain DQN', ['Basic DQN', 'Conv DQN'])
 
 #   plot_eval_folder([x], 'results/plaindqn', 'benchmark10_1v1', 'test_trained_sameEnv', 'Plain DQN', ['Basic DQN'])
 
    # plot_eval_folder(test, 'testfolderplot', 'benchmark10_1v1', 'firstjet')
    # qmixloss_benchmark = 'toplot/Apr12_16-18-44_qmix_log.json'
    # plot_tensorboard_log(qmixloss_benchmark, 'loss', 'QMIX', 'benchmark', 'benchmark/loss_qmix_benchmark')

    # qmixreward_benchmark = 'toplot/Apr12_16-18-44_milleniumfalcon_qmix_log1.json'
    # plot_tensorboard_log(qmixreward_benchmark, 'reward', 'QMIX', 'benchmark', 'benchmark/reward_qmix_benchmark')

    # dqnloss_benchmark = 'toplot/avril18_21-24-37_DeathStar_dqn_loss.json'
    # plot_tensorboard_log(dqnloss_benchmark, 'loss', 'DQN', 'benchmark', 'benchmark/loss_dqn_benchmark')

    # dqnrew_benchmark = 'toplot/avril18_21-24-37_DeathStar_dqn_rew.json'
    # plot_tensorboard_log(dqnrew_benchmark, 'reward', 'DQN', 'benchmark', 'benchmark/reward_dqn_benchmark')

    json_dqn_loss = ['results/plaindqn/tensorboard_data/run-mai09_22-51-32_DeathStar-tag-Loss _agent.json', 'results/plaindqnconv/tensorboard_data/run-mai10_15-06-41_DeathStar-tag-Loss _agent.json',  'results/dqn_conv_double/tensorboard_data/run-mai12_10-43-27_DeathStar-tag-Loss _agent.json', 'results/dqnconv_PER_double/tensorboard_data/run-mai11_23-38-06_DeathStar-tag-Loss _agent.json'
, 'results/dqn_conv_double_PER_annealing/tensorboard_data/run-mai13_11-58-36_DeathStar-tag-Loss _agent.json']
    
    json_dqn_reward = ['results/plaindqn/tensorboard_data/run-mai09_22-51-32_DeathStar-tag-Reward.json', 'results/plaindqnconv/tensorboard_data/run-mai10_15-06-41_DeathStar-tag-Reward.json',  'results/dqn_conv_double/tensorboard_data/run-mai12_10-43-27_DeathStar-tag-Reward.json', 'results/dqnconv_PER_double/tensorboard_data/run-mai11_23-38-06_DeathStar-tag-Reward.json'
, 'results/dqn_conv_double_PER_annealing/tensorboard_data/run-mai13_11-58-36_DeathStar-tag-Reward.json']
    
    json_dqn_steps = ['results/plaindqn/tensorboard_data/run-mai09_22-51-32_DeathStar-tag-Steps.json', 'results/plaindqnconv/tensorboard_data/run-mai10_15-06-41_DeathStar-tag-Steps.json',  'results/dqn_conv_double/tensorboard_data/run-mai12_10-43-27_DeathStar-tag-Steps.json', 'results/dqnconv_PER_double/tensorboard_data/run-mai11_23-38-06_DeathStar-tag-Steps.json'
, 'results/dqn_conv_double_PER_annealing/tensorboard_data/run-mai13_11-58-36_DeathStar-tag-Steps.json']
    
    json_dqn_wins = ['results/plaindqn/tensorboard_data/run-mai09_22-51-32_DeathStar-tag-Win.json', 'results/plaindqnconv/tensorboard_data/run-mai10_15-06-41_DeathStar-tag-Win.json',  'results/dqn_conv_double/tensorboard_data/run-mai12_10-43-27_DeathStar-tag-Win.json', 'results/dqnconv_PER_double/tensorboard_data/run-mai11_23-38-06_DeathStar-tag-Win.json'
, 'results/dqn_conv_double_PER_annealing/tensorboard_data/run-mai13_11-58-36_DeathStar-tag-Win.json']
    

    
    names = ['Plain DQN', 'Conv DQN', 'Conv DDQN', 'Conv DDQN PER', 'Conv DDQN PER with annealing']
 #   plot_tensorboard_compare(json_dqn_loss, names, 'loss', 'dqn', 'benchmark', 'benchmark/dqn_compare_loss')


 #   plot_tensorboard_compare(json_dqn_reward, names, 'reward', 'dqn', 'benchmark', 'benchmark/dqn_compare_reward')

 #   plot_tensorboard_compare(json_dqn_steps, names, 'steps /episode', 'dqn', 'benchmark', 'benchmark/dqn_compare_steps')

  #  plot_tensorboard_compare(json_dqn_wins, names, 'win rate', 'dqn', 'benchmark', 'benchmark/dqn_compare_wins')