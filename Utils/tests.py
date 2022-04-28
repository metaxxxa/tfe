import os
from collections import deque
from distutils.command.build import build
from matplotlib.font_manager import findSystemFonts
import numpy as np
from datetime import datetime
import re
import torch 
import os, sys
from PIL import Image, ImageDraw

os.chdir('/home/jack/Documents/ERM/Master thesis/tfe')
sys.path.insert(0, '/home/jack/Documents/ERM/Master thesis/tfe')

from Utils import helper
from RL_algorithms import dqn_defense as dqn
from env import defense_v0

LINES = False
WINDOW = 3

def eval_library(base_terrain, library_folder, adversary_tactic, params_directory, nb_episodes):
    constants = helper.Constants()
    for terrain_name in os.listdir(f'env/terrains/{library_folder}'):
        terrain = os.path.join(f'env/terrains/{library_folder}', terrain_name)

        if os.path.isfile(terrain):
            terrain_name = f'{library_folder}/{terrain_name}'.split('.ter')[0]
            #sim_index = int(terrain_name.split('_')[-1].split('.ter')[0])
            sim_index = helper.similarity_index(f'{library_folder}/{base_terrain}', terrain_name, LINES, WINDOW)
            
            env = defense_v0.env(terrain=terrain_name, max_cycles=constants.EPISODE_MAX_LENGTH, max_distance=constants.MAX_DISTANCE )
            env.reset()
            args_runner = dqn.Args(env)
            #args_runner.MODEL_DIR = MODEL_DIR
            #args_runner.RUN_NAME = RUN_NAME
            args_runner.ADVERSARY_TACTIC = adversary_tactic
            runner = dqn.Runner(env, args_runner)
            results = runner.eval(params_directory, nb_episodes=nb_episodes, visualize = False, log = False)
            print(results)


if __name__ == "__main__":

    lib_f = 'test'
    params = 'defense_params_dqn/benchmarking/281507avril2022step_0'
    base = 'benchmark_10x10_1v1_1'
    eval_library(base, lib_f, 'random', params , 100)