
# import random
# from collections import deque
import time
import numpy as np
import os
import shelve

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from Model import Model
from Agent import Agent
from env import Env, SHUTDOWN

from setup import *

import sys
stdout = sys.stdout
# sys.stdout = open("log.txt", "w")

np.random.seed(0)
random.seed(0)

import datetime
print(datetime.datetime.now())


USE_CUDA = torch.cuda.is_available()

env = Env()
n_episode = 30_000
state_size = 6 # env.observation_space.shape[0]
action_size = 2 #env.action_space.n
batch_size = 64

output_dir = './data/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
 
# agent = DQNAgent(state_size,action_size)
agent = Agent(state_size, action_size)
# agent = Agent(state_size, action_size, learning_rate=0.00048)
# agent.load(f'saved6/agent_0005600.hdf5')
# agent.load(f'saved8/best1136_opening290.hdf5')
# Column.OPENING_SIZE = 210
Column.OPENING_SIZE = 450
# agent.epsilon = agent.epsilon_min

# Save starting
agent.save('{}agent_{:07d}.hdf5'.format(output_dir,0))
 
episode = 0
count   = 0

f = open('loss.txt','w')
while episode < n_episode:
    episode+=1

    if SHUTDOWN: # for pygame
        break

    # Learning Loop
    state=env.reset() 
    state=np.reshape(state,[1, state_size])
    for t in range(5000):
        # time.sleep(0.5)
        if window is not None:
            SHUTDOWN = env.render()
            if SHUTDOWN: 
                break
        action = agent.act(state)
        next_state, reward, done = env.runframe(action)
        next_state=np.reshape(next_state,[1, state_size])
        agent.remember(state,action,reward,next_state,done)
        if done:
            # Let agent remember a few more times of it's failure
            for _ in range(5):
                agent.remember(state,action,reward,next_state,done)
        state=next_state
        if done:
            if episode%20 == 1:
            # if episode%50 ==1 or 5<env.count<15:
                print('episode: {}/{},\ttime: {},\tscore: {},\tepsilon: {:.2} \tLR: {:.05}'.format(
                    episode,
                    n_episode,
                    t,
                    env.count,
                    agent.epsilon,
                    agent.optimizer.param_groups[0]["lr"])
                )
            break

    # Learning Algorithm in Replay, updates agent.model
    if len(agent.memory)>batch_size:
        agent.replay(batch_size)


    if episode%5 == 4:
        agent.update_target_model() # Update target to same as Q

    if episode % 3 == 2:
        # agent.replay_last_n(10) # Replay most recent deaths
        # agent.replay_last(lr=0.003) # Replay most recent death
        # agent.replay_last() # Replay most recent death
        print(f"{agent.loss.item():.10f}", file=f)
 
    #To save file every 100 episodes into hdf5     
    if episode%200==0 and episode>100:
        
        print(episode,'epsilon:',round(agent.epsilon,3))
        agent.save('{}agent_{:07d}.hdf5'.format(output_dir,episode))
        #with shelve.open(f'{output_dir}memory.pickle') as file:
        #    file[f'{episode}'] = agent.memory  

    #To save best file
    count+= 1 if env.count>=20 else -1 if count>10 else -count
    if count>10:
        agent.epsilon = agent.epsilon_min
        print(f'Episode {episode}. Count is at {count}...')
        # agent.learning_rate_decay = 1
    if count>30:
        print(f'saving best {episode} with opening {Column.OPENING_SIZE}')
        agent.save(f'{output_dir}best{episode}_opening{Column.OPENING_SIZE}.hdf5')
        print('clearing memory')
        agent.memory.clear()
        print('reducing opening size')
        Column.OPENING_SIZE -= 60
        count = 0
    if Column.OPENING_SIZE < 150:
        break

if RENDERING:
    pygame.quit()

print(datetime.datetime.now())




