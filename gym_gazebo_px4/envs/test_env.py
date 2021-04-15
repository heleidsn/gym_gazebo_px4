'''
Author: Lei He
Date: 2021-04-14 16:48:15
LastEditTime: 2021-04-14 16:49:57
Description: test gym env
Github: https://github.com/heleidsn
'''

import gym

class TestEnv(gym.Env):
        
    def __init__(self):
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Discrete(2)

    def step(self, action):
        state = 1
    
        if action == 2:
            reward = 1
        else:
            reward = -1
            
        done = True
        info = {}
        return state, reward, done, info

    def reset(self):

        state = 0
        print('reset')

        return state