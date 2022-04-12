import os, sys

os.chdir('/home/jack/Documents/ERM/Master thesis/tfe')
sys.path.insert(0, '/home/jack/Documents/ERM/Master thesis/tfe')
import matplotlib.pyplot as plt
plt.style.use('Utils/plot_style.txt')
import pickle

# plot results
result_file = 'evals/results_dqn_testt_111943avril2022step_0.bin'
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