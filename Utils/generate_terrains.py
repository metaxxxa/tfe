
import os, sys
import numpy as np
import random
import string
os.chdir('/home/jack/Documents/ERM/Master thesis/tfe')
sys.path.insert(0, '/home/jack/Documents/ERM/Master thesis/tfe')
from env import utilities
import helper


def generate_random(size, nb_agents, obstacles_random_factor):
    nb_obstacles = round(obstacles_random_factor*size**2)
    random_obstacles = np.transpose(np.where(np.random.rand(size, size) < obstacles_random_factor)) #placing obstacles randomly
    obstacles = [tuple(obstacle) for obstacle in random_obstacles]

    blue_coord = [(np.random.randint(size), np.random.randint(size)) for _ in range(nb_agents)] #placing agents randomly at different locations
    red_coord = [(np.random.randint(size), np.random.randint(size)) for _ in range(nb_agents)]
    while red_coord ==  blue_coord:
        red_coord = [(np.random.randint(size), np.random.randint(size)) for _ in range(nb_agents)]
    obstacles = [i for i in obstacles if (i not in blue_coord and i not in red_coord)] #removing obstacles from agents, positions
    while len(obstacles) != nb_obstacles:
        if len(obstacles)!=0:
            obstacles.pop()
        while len(obstacles) < nb_obstacles:
            while True:
                coord = (np.random.randint(size), np.random.randint(size))
                if ((coord not in blue_coord) and (coord not in red_coord)):
                    obstacles.append(coord) 
                    break 
            
    terrain = {'size': size, 'obstacles': obstacles, 'blue': blue_coord, 'red':red_coord}
    return terrain


def generate_on_diff(size, nb_agents, base_terrain, similarity_index, tolerance, window=3, lines=False):
    error = tolerance*1.1
    steps = 3
    while error > tolerance:
        #for i in np.linspace(0,1, steps):
        for i in 0.3**np.linspace(5,1,steps):  #high number of obstacles directly lead to more environments in the small similarity index range
            terrain_generated = generate_random(size, nb_agents, i)
            error = abs(helper.similarity_index(utilities.load_terrain(base_terrain), terrain_generated, lines, window) - similarity_index)
            if error < tolerance:
                break
        steps = steps *2

    return terrain_generated

def generate_library(size, nb_agents, base_terrain, nb_env, folder, window=3, lines=False):
    for i in np.linspace(0,0.25, nb_env):
        terrain = generate_random(size, nb_agents, i)
        similarity = helper.similarity_index(utilities.load_terrain(base_terrain), terrain, lines, window)
        id = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(6)]) + f'_simIndex_{similarity}'
        utilities.write_terrain(folder, id, terrain)

def generate_eval_lib(size, nb_agents, base_terrain, nb_env, folder, tolerance=0.005, window=3, lines=False):
    similarity_index = list(np.linspace(0,1,nb_env))
    if os.path.exists(f'env/terrains/{folder}'):  #if continuing on existing folder
        for filename in os.listdir(f'env/terrains/{folder}'):
            corresponding_index = np.nonzero(abs((np.asarray(similarity_index) - float(filename.split('_')[2].split('_')[0]))) < tolerance)
            if len(corresponding_index[0]) != 0:
                similarity_index.pop(corresponding_index[0][0])


    steps = 3
    
    while len(similarity_index) > 0:
        
        for i in 0.3**np.linspace(5,1,steps):  #high number of obstacles directly lead to more environments in the small similarity index range
            terrain_generated = generate_random(size, nb_agents, i)
            error = abs(helper.similarity_index(utilities.load_terrain(base_terrain), terrain_generated, lines, window) - similarity_index)
            if len((error<tolerance).nonzero()[0]) != 0:
                index = (error<tolerance).nonzero()[0][0]
                similarity_index.pop(index)
                similarity = helper.similarity_index(utilities.load_terrain(base_terrain), terrain_generated, lines, window)
                id = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(6)]) + f'_simIndex_{similarity}'
                utilities.write_terrain(folder, id, terrain_generated)
                if len(similarity_index) != 0:
                    break
        steps = steps +4




        




if __name__ == "__main__":

    os.chdir('/home/jack/Documents/ERM/Master thesis/tfe')
    sys.path.insert(0, '/home/jack/Documents/ERM/Master thesis/tfe')
    for i in range(5):
        utilities.write_terrain('testt', f'test_terrain{i}', generate_random(8,1, 0.1))
    base= 'benchmark_10x10_1v1'
    index = 3
    lines = False
    window = 3
    for i in range(1,6):
        print(helper.similarity_index(base, f'test/sim{i}_', lines, window))
    
    #generate_library(10, 1, 'benchmark_10x10_1v1', 10, 'training_lib')
    generate_eval_lib(10, 1,base, 500, 'eval_lib')
  #  utilities.write_terrain('test', f'ondiff_ind={index}', t)
 #   helper.show_terrain(f'test/ondiff_ind={index}')