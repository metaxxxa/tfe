import torch
import torch.nn as nn
import numpy as np
import gym
import random
from collections import deque
import itertools
import random
from torch.utils.tensorboard import SummaryWriter


if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev) 

#setting up TensorBoard

writer = SummaryWriter()

class args:
    BUFFER_SIZE = 10000
    REW_BUFFER_SIZE = 100
    LEARNING_RATE = 1e-4
    MIN_BUFFER_LENGTH = 1000
    BATCH_SIZE = 200
    GAMMA = 0.9
    EPSILON_START = 1
    EPSILON_END = 0.001
    EPSILON_DECAY = 100000
    SYNC_TARGET_FRAMES = 100
    VISUALIZE_WHEN_LEARNED = True

class DQN(nn.Module):
    def __init__(self, env):
        super().__init__()
        #params
        hidden_layer1_dim = 32
        #hidden_layer2_dim = 60
        self.net = nn.Sequential(
            nn.Linear(np.prod(env.observation_space.shape) , hidden_layer1_dim),
            nn.ReLU(),
            #nn.Linear(hidden_layer1_dim,hidden_layer2_dim),
            #nn.ReLU(),
            nn.Linear(hidden_layer1_dim, env.action_space.n)
        ).to(device)
        

        self.to(device)
    
    def forward(self, obs_t):
        
        return self.net(obs_t)
        
    def get_Q_values(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
        q_values = self.net(obs_t.unsqueeze(0))
        return q_values

    def get_Q_max(self, q_values):
        max_q_index = torch.argmax(q_values, dim=1)[0]
        max_q_index = max_q_index.detach().item()
        max_q = q_values[0,max_q_index]
        return max_q_index, max_q

    def act(self, obs):
        action, _ = self.get_Q_max(self.get_Q_values(obs))
        return action
    
    
class runner_DQN:
    def __init__(self,args):
        self.args = args
        self.env = gym.make('CartPole-v0')

        self.replay_buffer = deque(maxlen=self.args.BUFFER_SIZE)
        self.rew_buffer = deque([0.0], maxlen=self.args.REW_BUFFER_SIZE)
        self.loss_buffer = deque([0.0], maxlen=self.args.REW_BUFFER_SIZE)

        self.online_net = DQN(self.env)
        self.target_net = DQN(self.env)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.net.parameters(), lr = self.args.LEARNING_RATE)

  

    def run(self):
        observation = self.env.reset()
        episode_reward = 0.0
        #Init replay buffer
        for _ in range(self.args.MIN_BUFFER_LENGTH):
            transition = []
            action = self.env.action_space.sample()
            
            new_observation, reward, done, info = self.env.step(action)
            episode_reward += reward
            transition = [observation, action,reward,done, new_observation]
            observation = new_observation
            if done:
                observation = self.env.reset()
                self.rew_buffer.append(episode_reward)
                episode_reward = 0.0
                
            
            self.replay_buffer.append(transition)

        # training and optimization

        observation = self.env.reset()
        episode_reward = 0.0

                
        for step in itertools.count():
            epsilon = np.interp(step, [0, self.args.EPSILON_DECAY], [self.args.EPSILON_START, self.args.EPSILON_END])
            rnd_sample = random.random()
            transition = []
            if rnd_sample <= epsilon:
                action = self.env.action_space.sample()
            else:
                action = self.online_net.act(observation)
            
            new_observation, reward, done, info = self.env.step(action)
            episode_reward += reward
            transition = [observation, action,reward,done, new_observation]
            observation = new_observation
            if done:
                observation = self.env.reset()
                self.rew_buffer.append(episode_reward)
                writer.add_scalar("Reward", episode_reward,step  )
                episode_reward = 0.0
            
            self.replay_buffer.append(transition)
            

            #gradient step
            transitions = random.sample(self.replay_buffer, self.args.BATCH_SIZE)

            obses = np.asarray([t[0] for t in transitions])
            actions = np.asarray([t[1] for t in transitions])
            rews = np.asarray([ t[2] for t in transitions])
            dones = np.asarray([t[3] for t in transitions])
            new_obses = np.asarray([t[4] for t in transitions])

            obses_t = torch.as_tensor(obses, dtype=torch.float32).to(device)
            actions_t = torch.as_tensor(actions, dtype=torch.int64).to(device).unsqueeze(-1)
            rews_t = torch.as_tensor(rews, dtype=torch.float32).to(device).unsqueeze(-1)
            dones_t = torch.as_tensor(dones, dtype=torch.float32).to(device).unsqueeze(-1)
            new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32).to(device)


            target_q_values = self.target_net(new_obses_t)
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

            targets = rews_t + self.args.GAMMA*(1 - dones_t)*max_target_q_values
            q_values_online = torch.gather(input=self.online_net.get_Q_values(obses_t).squeeze(0), dim=1, index=actions_t)

            loss_fun = nn.MSELoss()
            loss = loss_fun(q_values_online, targets)
            self.loss_buffer.append(loss.detach().item())
            writer.add_scalar("Loss", loss, step)
            


            # gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #update target network

            if step % self.args.SYNC_TARGET_FRAMES == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            #logging
            if step % 1000 == 0:
                print('\n Step', step )
                print('Avg Rew', np.mean(self.rew_buffer))
                print('Avg Loss', np.mean(self.loss_buffer))


            if np.mean(self.rew_buffer) > 185:
                        
                while self.args.VISUALIZE_WHEN_LEARNED:
                    obs = self.env.reset()
                    self.env.render()
                    obs, _, done, _ = self.env.step(self.online_net.act(obs))
                    if done:
                        obs = self.env.reset()
        writer.close() 

if __name__ == "__main__":
    runner = runner_DQN(args)
    runner.run()
