import os, sys
import csv
import json
from termios import XTABS
import numpy as np

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



result_file = 'evals/avril12_16-19-03_DeathStar.csv'
jason = 'evals/avril11_19-43-44_DeathStar.json'
with open(result_file) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    csvReader

def plot_tensorboard_log(json_file, data_type, algo, env, name, size=(10,5)):
    f = open(json_file)
    data = json.load(f)
    step = np.asarray([element[1] for element in data])
    loss = np.asarray([element[2] for element in data])

    fig = plt.figure(figsize=size)
    plt.plot(step, loss)
    plt.xlabel('Step')
    if data_type == 'loss':
        plt.ylabel('Loss /agent /step')
    elif data_type == 'reward':
        plt.ylabel('Reward /agent')
    plt.title(f'Training the {algo} on the {env} environment')

    fig.savefig(f'figures/{name}.png', bbox_inches='tight', dpi=300)
    plt.show(block=False)
    plt.pause(1)
    f.close()
# Closing file

plot_tensorboard_log(jason, 'loss', 'DQN', 'random env', 'testok')