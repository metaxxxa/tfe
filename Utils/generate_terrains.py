
from ctypes import util
import os, sys
import pickle
import numpy as np
os.chdir('/home/jack/Documents/ERM/Master thesis/tfe')
sys.path.insert(0, '/home/jack/Documents/ERM/Master thesis/tfe')
from env import utilities

size = 7

terr = utilities.make_terrain(size)

dir = 'random'
utilities.load_terrain('central_5x5')
a = utilities.render_terrain(terr)


def generate_random(size, nb_agents, obstacles_random_factor):

    random_obstacles = np.transpose(np.where(np.random.rand(size, size) < obstacles_random_factor)) #placing obstacles randomly
    obstacles = [tuple(obstacle) for obstacle in random_obstacles]

    blue_coord = [(np.random.randint(size), np.random.randint(size)) for i in range(nb_agents)] #placing agents randomly
    red_coord = [(np.random.randint(size), np.random.randint(size)) for i in range(nb_agents)]

    obstacles = [i for i in obstacles if (i not in blue_coord and i not in red_coord)] #removing obstacles from agents, positions
    terrain = {'size': size, 'obstacles': obstacles, 'blue': blue_coord, 'red':red_coord}
    return terrain

rand = generate_random(8,1, 0.1)
print(rand)

for i in range(5):
    utilities.write_terrain('testt', f'test_terrain{i}', generate_random(8,1, 0.1))
