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

def plot_eval_folder(resultfile, plot_folder, base_env, name, figure_name, names=[], show_var='', filter_factor=1, training_plot=False, size=(10,5)):


    
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
        if show_var == 'error bars':
            plt.errorbar(x, uniform_filter1d(y, size=filter_factor), yerr=var, fmt='o', ecolor=COLORS[i], color='white')
        elif show_var == 'interval':
            new_step, std_up, std_down = compute_variance(x, uniform_filter1d(y, size=filter_factor), 20)
            plt.fill_between(new_step, std_down, std_up, color=COLORS[i] , alpha=0.1)

        i += 1
    plt.legend()
    plt.xlabel('Similarity Index')
    plt.ylabel('Reward /agent')
    if training_plot:
        plt.title(f'Reward per episode for {figure_name} trained on {base_env}')
    else:
        plt.title(f'Reward per episode for {figure_name}')
    plt.show(block=False)
    plt.pause(1)
    
    plt.savefig(f'{folder_name}/{figure_name}_reward_plot.png', bbox_inches='tight')
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
        if show_var == 'error bars':
            plt.errorbar(x, uniform_filter1d(y, size=filter_factor), yerr=var, fmt='o', ecolor=COLORS[i], color='white')
        elif show_var == 'interval':
            new_step, std_up, std_down = compute_variance(x, uniform_filter1d(y, size=filter_factor), 20)
            plt.fill_between(new_step, std_down, std_up, color=COLORS[i] , alpha=0.1)
        i += 1
    plt.legend()
    plt.xlabel('Similarity Index')
    plt.ylabel('Steps /episode')
    if training_plot:
        plt.title(f'Steps per episode for {figure_name} trained on {base_env}')
    else:
        plt.title(f'Steps per episode for {figure_name}')
    plt.show(block=False)
    plt.pause(1)
    plt.savefig(f'{folder_name}/{figure_name}_steps_plot.png', bbox_inches='tight')
    plt.close()
    i= 0
   # filter_factor = int(np.ceil(filter_factor/10))
    for file in resultfile:
        similarity_index, reward, steps, wins, nb_episodes, reward_v, steps_v, wins_v = compute_results(file)

        x, y = zip(*sorted(zip(similarity_index, wins)))
        _, var = zip(*sorted(zip(similarity_index, wins_v)))
        if len(names) != 0:
            plt.plot(x,  uniform_filter1d(y, size=filter_factor), label=names[i], color=COLORS[i])
        else:
            plt.plot(x,  uniform_filter1d(y, size=filter_factor), color=COLORS[i])
        if show_var == 'error bars':
            plt.errorbar(x, uniform_filter1d(y, size=filter_factor), yerr=var, fmt='o', ecolor=COLORS[i], color='white')
        elif show_var == 'interval':
            new_step, std_up, std_down = compute_variance(x, uniform_filter1d(y, size=filter_factor), 20)
            plt.fill_between(new_step, std_down, std_up, color=COLORS[i] , alpha=0.1)
        i +=1 
    plt.legend()
    plt.xlabel('Similarity Index')
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.ylabel('Win rate')
    if training_plot:
        plt.title(f'Win rate per episode for {figure_name} trained on {base_env}')
    else:
        plt.title(f'Win rate per episode for {figure_name}')
    plt.show(block=False)
    plt.pause(1)
    plt.savefig(f'{folder_name}/{figure_name}_wins_plot.png', bbox_inches='tight')
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
        ci = Z*np.std(segment)/math.sqrt(len(segment))
        std_up.append(data[i*window+int(np.floor((len(segment)-1)/2))] +ci )
        std_down.append(data[i*window+int(np.floor((len(segment)-1)/2))] - ci)
        new_step.append((steps[i*window+int(np.floor((len(segment)-1)/2))]))
    return new_step, std_up, std_down

def plot_sim_index_repartition(env_library, name, nb_bars=10, size=(10,5)):
    sim_index = []
    i = 0
    for filename in os.listdir(f'env/terrains/{env_library}'):
        i += 1
        sim_index_string = filename.split('simIndex_')[-1].split('_')[0]
        if sim_index_string == 'benchmark':
            sim_index_string = '1'
        sim_index.append(float(sim_index_string))

    fig = plt.figure(figsize=size)
    plt.hist(sim_index, bins = np.linspace(0,1,nb_bars))
    plt.xlabel('Similarity index')
    plt.ylabel("N° of environments") 
    
    plt.title(f'Similarity index distribution of environments {name} set')
    fig.savefig(f'figures/{name}sim_index_distrib.png', bbox_inches='tight', dpi=300)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

if __name__ == "__main__":
    
 #   plot_sim_index_repartition('training_lib', 'test', )

    plain_dqn_small_lib = 'eval_plaindqn3/results_dqn_eval_lib_fixedObs3_100238mai2022step_300000.bin'
    plain_dqn_lines_small_lib = 'results/plaindqn/small_evallib_lines/results_dqn_eval_lib_fixedObs_lines_100238mai2022step_300000.bin'
    plain_dqn_conv_small_lib = 'eval_plaindqnconv/results_dqn_eval_lib_fixedObs3_.bin'
    plain_dqn_conv_lines_small_lib = 'results/plaindqnconv/small_evallib_lines/results_dqn_eval_lib_fixedObs_lines_.bin'
    #plain_dqn = 'test/results_dqn_eval_lib_fixedObs3_100238mai2022step_300000.bin'
    plaindqnconv = 'results/plaindqnconv/evallib_plaindqnconv/results_dqn_eval_lib_.bin'
    plaindqnconvLines = 'results/plaindqnconv/evallib_lines/results_dqn_eval_lib_lines_.bin'
    CE_plaindqnconvLines = 'results/change_env/plain/dqn/dqn_plain_evallib_lines/results_dqn_eval_lib_.bin'
    ddqn_conv_PERanneal = 'results/dqn_conv_double_PER_annealing/evallib_ddqnconv_anneal/results_dqn_eval_lib_131520mai2022step_100000.bin'
    ddqn_conv_PERanneal_lines = 'results/dqn_conv_double_PER_annealing/evallib_lines/results_dqn_eval_lib_lines_131520mai2022step_100000.bin'
    ddqn_conv_PERanneal_lines = 'results/dqn_conv_double_PER_annealing/dqn_enhanced/evallib_lines/results_dqn_eval_lib_lines_190322May2022step_300000.bin'
    CE_ddqn_conv_PERanneal_lines = 'results/change_env/with_features/dqn/evallib_lines/results_dqn_eval_lib_lines_180810May2022step_300000.bin'
    
    test_dqn_evallib = [plaindqnconvLines, ddqn_conv_PERanneal_lines, CE_plaindqnconvLines, CE_ddqn_conv_PERanneal_lines]
  #  plot_eval_folder([plain_dqn_small_lib, plain_dqn_conv_small_lib], 'results/plaindqn', 'benchmark terrain', 'test_trained_sameEnv', 'Plain DQN', ['Basic DQN', 'Conv DQN'], 'error bars', 1)
 #   plot_eval_folder([plain_dqn_lines_small_lib, plain_dqn_conv_lines_small_lib], 'results/plaindqn', 'benchmark terrain', 'test_trained_sameEnv_lines', 'Plain DQN', ['Basic DQN', 'Conv DQN'], 'error bars', 1)
    
    names = ['DQN trained on benchmark', 'PER DDQN trained on benchmark']
    plot_eval_folder([plaindqnconvLines, ddqn_conv_PERanneal_lines], 'figures', 'benchmark', 'res_comp_dqn_benchmark', 'convolutional DQN', names, 'interval', 10)

    names = ['Plain DQN trained on benchmark', 'PER DDQN trained on benchmark','Plain DQN trained on training set', 'PER DDQN trained on training set']
    plot_eval_folder(test_dqn_evallib, 'figures', 'benchmark', 'res_comp_all_dqn', 'convolutional DQN', names, 'interval', 50)
    names = ['Plain Conv DQN trained on training set', 'PER Conv DDQN trained on training set']
    plot_eval_folder([CE_plaindqnconvLines, CE_ddqn_conv_PERanneal_lines], 'figures/CE_conv', 'changing', 'res_comp_dqn_CE', 'convolutional DQN', names, 'interval', 50)
 

    names = ['DQN fully observable', 'VDN fully observable', 'QMIX fully observable', 'DQN partially observable', 'VDN partially observable', 'QMIX partially observable']
    plot_eval_folder([ddqn_conv_PERanneal_lines], 'figures/observability', 'changing', 'res_comp_observability', 'convolutional DQN', names, 'interval', 50)
 

    # plot_eval_folder(test, 'testfolderplot', 'benchmark10_1v1', 'firstjet')
    # qmixloss_benchmark = 'toplot/Apr12_16-18-44_qmix_log.json'
    # plot_tensorboard_log(qmixloss_benchmark, 'loss', 'QMIX', 'benchmark', 'benchmark/loss_qmix_benchmark')

    # qmixreward_benchmark = 'toplot/Apr12_16-18-44_milleniumfalcon_qmix_log1.json'
    # plot_tensorboard_log(qmixreward_benchmark, 'reward', 'QMIX', 'benchmark', 'benchmark/reward_qmix_benchmark')

    # dqnloss_benchmark = 'toplot/avril18_21-24-37_DeathStar_dqn_loss.json'
    # plot_tensorboard_log(dqnloss_benchmark, 'loss', 'DQN', 'benchmark', 'benchmark/loss_dqn_benchmark')

    # dqnrew_benchmark = 'toplot/avril18_21-24-37_DeathStar_dqn_rew.json'
    # plot_tensorboard_log(dqnrew_benchmark, 'reward', 'DQN', 'benchmark', 'benchmark/reward_dqn_benchmark')
    dqn_f_Loss = 'results/dqn_conv_double_PER_annealing/tensorboard_data/run-mai13_11-58-36_DeathStar-tag-Loss _agent.json'
    dqn_f_Loss = 'results/dqn_conv_double_PER_annealing/dqn_enhanced/run-May18_15-53-51_milleniumfalcon-tag-Loss _agent.json'

    dqn_f_Steps = 'results/dqn_conv_double_PER_annealing/tensorboard_data/run-mai13_11-58-36_DeathStar-tag-Steps.json'
    dqn_f_Steps = 'results/dqn_conv_double_PER_annealing/dqn_enhanced/run-May18_15-53-51_milleniumfalcon-tag-Steps.json'


    dqn_f_Reward = 'results/dqn_conv_double_PER_annealing/tensorboard_data/run-mai13_11-58-36_DeathStar-tag-Reward.json'
    dqn_f_Reward = 'results/dqn_conv_double_PER_annealing/dqn_enhanced/run-May18_15-53-51_milleniumfalcon-tag-Reward.json'


    dqn_f_Win = 'results/dqn_conv_double_PER_annealing/tensorboard_data/run-mai13_11-58-36_DeathStar-tag-Win.json'
    dqn_f_Win = 'results/dqn_conv_double_PER_annealing/dqn_enhanced/run-May18_15-53-51_milleniumfalcon-tag-Win.json'


    json_dqn_loss = ['results/plaindqn/tensorboard_data/run-mai09_22-51-32_DeathStar-tag-Loss _agent.json', 'results/plaindqnconv/tensorboard_data/run-mai10_15-06-41_DeathStar-tag-Loss _agent.json',  'results/dqn_conv_double/tensorboard_data/run-mai12_10-43-27_DeathStar-tag-Loss _agent.json', 'results/dqnconv_PER_double/tensorboard_data/run-mai11_23-38-06_DeathStar-tag-Loss _agent.json'
, dqn_f_Loss]
    
    json_dqn_reward = ['results/plaindqn/tensorboard_data/run-mai09_22-51-32_DeathStar-tag-Reward.json', 'results/plaindqnconv/tensorboard_data/run-mai10_15-06-41_DeathStar-tag-Reward.json',  'results/dqn_conv_double/tensorboard_data/run-mai12_10-43-27_DeathStar-tag-Reward.json', 'results/dqnconv_PER_double/tensorboard_data/run-mai11_23-38-06_DeathStar-tag-Reward.json'
, dqn_f_Reward]
    
    json_dqn_steps = ['results/plaindqn/tensorboard_data/run-mai09_22-51-32_DeathStar-tag-Steps.json', 'results/plaindqnconv/tensorboard_data/run-mai10_15-06-41_DeathStar-tag-Steps.json',  'results/dqn_conv_double/tensorboard_data/run-mai12_10-43-27_DeathStar-tag-Steps.json', 'results/dqnconv_PER_double/tensorboard_data/run-mai11_23-38-06_DeathStar-tag-Steps.json'
, dqn_f_Steps]
    
    json_dqn_wins = ['results/plaindqn/tensorboard_data/run-mai09_22-51-32_DeathStar-tag-Win.json', 'results/plaindqnconv/tensorboard_data/run-mai10_15-06-41_DeathStar-tag-Win.json',  'results/dqn_conv_double/tensorboard_data/run-mai12_10-43-27_DeathStar-tag-Win.json', 'results/dqnconv_PER_double/tensorboard_data/run-mai11_23-38-06_DeathStar-tag-Win.json'
, dqn_f_Win]
    
    dqn_cE_Loss = ['results/change_env/plain/dqn/tensorboard_data/run-May16_00-17-42_DeathStar-tag-Loss _agent.json']
    
    dqn_cE_Steps = ['results/change_env/plain/dqn/tensorboard_data/run-May16_00-17-42_DeathStar-tag-Steps.json']
    
    dqn_cE_Reward = ['results/change_env/plain/dqn/tensorboard_data/run-May16_00-17-42_DeathStar-tag-Reward.json']
    
    dqn_cE_Win = ['results/change_env/plain/dqn/tensorboard_data/run-May16_00-17-42_DeathStar-tag-Win.json']
    

    qmix_plain_Loss =  'results/qmix/plain/tensorboard_data/run-mai10_22-59-13_DeathStar-tag-Loss.json'
    qmix_plain_Steps =  'results/qmix/plain/tensorboard_data/run-mai10_22-59-13_DeathStar-tag-Steps.json'
    qmix_plain_Wins =  'results/qmix/plain/tensorboard_data/run-mai10_22-59-13_DeathStar-tag-Win.json'
    qmix_plain_Reward =  'results/qmix/plain/tensorboard_data/run-mai10_22-59-13_DeathStar-tag-Reward.json'

    qmixvdn_plain_Loss = [qmix_plain_Loss]
    qmixvdn_plain_Steps = [qmix_plain_Steps]
    qmixvdn_plain_Reward= [qmix_plain_Reward]
    qmixvdn_plain_Wins = [qmix_plain_Wins]

    names = ['Plain QMIX']
  #  plot_tensorboard_compare(qmixvdn_plain_Loss, names, 'loss', 'qmix', 'changing', 'base/qmix_vdn_compare_Loss')
#
 #   plot_tensorboard_compare(qmixvdn_plain_Steps, names, 'steps /episode', 'qmix', 'changing', 'base/qmix_vdn_compare_Steps')

  #  plot_tensorboard_compare(qmixvdn_plain_Reward, names, 'reward', 'qmix', 'changing', 'base/qmix_vdn_compare_Reward')

   # plot_tensorboard_compare(qmixvdn_plain_Wins, names, 'win rate', 'qmix', 'changing', 'base/qmix_vdn_compare_Wins')


    names = ['Plain DQN', 'Conv DQN', 'Conv DDQN', 'Conv DDQN PER', 'Conv DDQN PER with annealing']
    plot_tensorboard_compare(json_dqn_loss, names, 'loss', 'dqn', 'benchmark', 'benchmark/dqn_compare_loss')


    plot_tensorboard_compare(json_dqn_reward, names, 'reward', 'dqn', 'benchmark', 'benchmark/dqn_compare_reward')

    plot_tensorboard_compare(json_dqn_steps, names, 'steps /episode', 'dqn', 'benchmark', 'benchmark/dqn_compare_steps')

    plot_tensorboard_compare(json_dqn_wins, names, 'win rate', 'dqn', 'benchmark', 'benchmark/dqn_compare_wins')

    names = ['Plain DQN']
    plot_tensorboard_compare(dqn_cE_Loss, names, 'loss', 'dqn', 'changing', 'changing_environments/dqnCE_compare_loss')

    plot_tensorboard_compare(dqn_cE_Steps, names, 'steps /episode', 'dqn', 'changing', 'changing_environments/dqnCE_compare_steps')

    plot_tensorboard_compare(dqn_cE_Reward, names, 'reward', 'dqn', 'changing', 'changing_environments/dqnCE_compare_reward')

    plot_tensorboard_compare(dqn_cE_Win, names, 'win rate', 'dqn', 'changing', 'changing_environments/dqnCE_compare_wins')