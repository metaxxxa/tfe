from collections import deque
from distutils.command.build import build
from matplotlib.font_manager import findSystemFonts
import numpy as np
from datetime import datetime
import re
import torch 
import os, sys
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim

class Constants:
    def __init__(self):
        self.EPISODE_MAX_LENGTH = 200
        self.MAX_DISTANCE = 5

#to log results
class Metrics:
    def __init__(self, nb_episodes):
        self.nb_episodes = nb_episodes
        self.rewards_buffer = deque(maxlen=nb_episodes)
        self.wins = deque(maxlen=nb_episodes)
        self.nb_steps = deque(maxlen=nb_episodes)
        self.env = ''

class Buffers:
    def __init__(self, env, args, agents, device):
        self.action = dict()
        self.observation = dict()
        self.observation_next = dict()
        self.hidden_state = dict()
        self.hidden_state_next = dict()
        self.episode_reward = 0.0
        self.nb_transitions = 0
        self.replay_buffer = deque(maxlen=args.BUFFER_SIZE)
        if args.USE_PER:
            self.priority = deque(maxlen=args.BUFFER_SIZE)
            self.weights = deque(maxlen=args.BUFFER_SIZE)
        self.rew_buffer = deque([0.0], maxlen=args.REW_BUFFER_SIZE)
        self.loss_buffer = deque([0.0], maxlen=args.REW_BUFFER_SIZE)
        self.steps_buffer = deque([0.0], maxlen=args.REW_BUFFER_SIZE)
        self.wins_buffer = deque([0.0], maxlen=args.REW_BUFFER_SIZE)

        for agent in agents:
            self.observation[agent] = env.observe(agent)
            self.hidden_state[agent] = torch.zeros(args.dim_L2_agents_net, device=device).unsqueeze(0)
       

class Params:
    def __init__(self, step):
        self.step = step
        
def get_device():
    if torch.cuda.is_available():
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    
    return torch.device(dev) 

def mask_array(array, mask):
    int = np.ma.compressed(np.ma.masked_where(mask==0, array) )

    return np.ma.compressed(np.ma.masked_where(mask==0, array) )

def similarity_index(ter1, ter2, lines, window):
    img1 = build_terrain(utilities.load_terrain(ter1), lines)[1]
    img2 = build_terrain(utilities.load_terrain(ter2), lines)[1]

    return ssim(img1, img2, win_size=window, )

def build_terrain(terrain, lines=False):
    ter = np.ones([terrain['size'], terrain['size'], 3], dtype='uint8')*255
    
    #making an image
    
    if lines:
        image = Image.fromarray(ter, mode='RGB')
        imageD = ImageDraw.Draw(image)
        for coord_blue in terrain['blue']:
            for coord_red in terrain['red']:
                imageD.line([coord_blue[1], coord_blue[0], coord_red[1], coord_red[0]], fill ="green", width = 0)

        ter = np.array(image)

    for x, y in terrain['obstacles']:
        ter[x,y,0], ter[x,y,1], ter[x,y,2] = 0, 0, 0
    for x, y in terrain['blue']:
        ter[x,y,0], ter[x,y,1], ter[x,y,2] = 0, 0, 255
    for x, y in terrain['red']:
        ter[x,y,0], ter[x,y,1], ter[x,y,2] = 255, 0, 0
    
    image = Image.fromarray(ter, mode='RGB')

    image.show()

    return image, np.array(image)


if __name__ == "__main__":
        
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    os.chdir(BASE_DIR)
    sys.path.insert(0, BASE_DIR)
    from env import defense_v0

    
    from env import utilities



    ter = 'benchmark_10x10_1v1'
    ter2 = 'benchmark_10x10_1v1_sym'
    t = 'central_10x10'

    i = similarity_index(ter, t, True, 3)

    env = defense_v0.env(terrain=ter, max_cycles=200, max_distance=5)
    env.reset()
    im = env.render(mode='rgb_array')
    print(f'Similarity index: {i}')
    

