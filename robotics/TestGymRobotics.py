import sys
import os
print(sys.path)
sys.path.insert(0, 'C:\\Users\\DID1TV\\.mujoco\\mjpro150\\bin')
os.system('echo %PATH%')
os.system('set PATH=C:\\Users\\DID1TV\\.mujoco\\mjpro150\\bin;%PATH%')
os.system('echo %PATH%')
print(sys.path)

import gym
#import mujoco_py

#env = gym.make('FetchPickAndPlace-v1')
#env = gym.make('FetchPush-v1')
env = gym.make('Reacher-v2')
env.reset()
env.render()

print("OK!")