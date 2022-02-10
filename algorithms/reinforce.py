"""Independet REINFORCE for DEFENSE (with action mask)
We implement the simplest policy gradient method: REINFORCE.

A policy gradient algorithm immediately optimizes the policy itself
(we don't need the notion of a value-action function) by computing
the gradient of the policy (via the policy-gradient theorem) and 
performing gradient ascent.

WITH parameter (weight) sharing => same policy network for each team

Williams, R.J. Simple statistical gradient-following algorithms
for connectionist reinforcement learning. Mach Learn 8, 229–256 
(1992). https://doi.org/10.1007/BF00992696
"""

import numpy as np 
import torch
from torch.distributions import Categorical

from utilities import build_network, EpisodeStep

# hack to allow import of env
import sys; sys.path.insert(0, '.')
from env import defense_v0 

class Agent:
    """A Reinforce Agent"""
    def __init__(self, name, net, gamma) -> None:
        self.name = name
        self.policy = net
        self.gamma = gamma
    
    def get_action(self, observation):
        action_mask = observation['action_mask']
        with torch.no_grad():
            logits = self.policy(torch.from_numpy(observation['obs']).float())
            logits[action_mask == 0.0] = -1e6 # set logits of not-allowed actions to very low value
            action = Categorical(logits=logits).sample()
        return action.item()

    def update(self, batch):
        pass

class Runner:
    def __init__(self, terrain='flat_5x5') -> None:
        self.env = defense_v0.env(terrain=terrain)
        self.env.reset()
        self.gamma = 0.99

        nets = {}
        for team in ['blue', 'red']:
            nets[team] = build_network([env.observation_space(team+'_0')['obs'].shape[0], 128, 
                                        env.action_space(team+'_0').n])

        self.agents = {}
        for agent in self.env.agents:
            self.agents[agent] = Agent(name=agent, net=nets[agent[:-2]], gamma=self.gamma)

    def run(self, n_iters=10):
        for ii in range(n_iters):
            episode = self.generate_episode()
            rtgs = self.rewards_to_go(episode)
            print('....')

    def generate_episode(self):
        self.env.reset()
        episode = {agent: [] for agent in self.env.agents}
        done = False
        for agent in self.env.agent_iter():
            self.env.render()
            observation, reward, done, _ =  self.env.last() 
            # set observation, done and reward as next_obs of previous step
            if episode[agent]:
                episode[agent][-1].next_obs = observation['obs']
                episode[agent][-1].next_mask = observation['action_mask']
                episode[agent][-1].reward = reward
                episode[agent][-1].done = done
            
            if done:
                action = None
            else:
                action = self.agents[agent].get_action(observation)
            
            self.env.step(action)
            episode[agent].append(EpisodeStep(observation['obs'], observation['action_mask'], action,
                                              None, None, None, None))
        return episode
     
    def rewards_to_go(self, batch):
        rtgs = {agent : [] for agent in batch}
        for agent in batch:
            R = 0
            for step in reversed(batch[agent][:-1]):
                if step.done:
                    R = 0
                R = self.gamma*R +  step.reward
                rtgs[agent].insert(0, R)
        return rtgs


def train():
    pass

if __name__ == '__main__':
    env = defense_v0.env(terrain='flat_5x5')
    agent = 'blue_0'
    net = build_network([env.observation_space(agent)['obs'].shape[0], 128, env.action_space(agent).n])
    print('....')

    env.reset()
    obs, _, _, _ = env.last()
    """"
    agent = Agent('blue_0', env.observation_space(agent)['obs'].shape[0], env.action_space(agent).n,
                  gamma=0.99, lr=0.01, layers=[128])
    print(agent.get_action(obs))
    """
    runner = Runner()
    runner.run(n_iters=10)