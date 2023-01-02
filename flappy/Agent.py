import random, math, copy
import numpy as np
from collections import deque
from itertools import islice
import shelve

import torch
import torch.optim as optim
from Model import Model

output_dir = './data/'

# GAMMA = 0.993 
GAMMA = 0.988 # 0.5 after 57 frames
LEARNING_RATE = 3e-3 # 0.003 is optimal?

class Agent:
    def __init__(self,state_size,action_size,learning_rate=LEARNING_RATE, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)

        # adjusting epsilon
        self.epsilon = 0.999 # exploitation=0 vs exploration=1
        self.epsilon_decay = 0.995 # less and less each time
        self.epsilon_min = 0.01 # 1% exploration

        # Adjusting learning rate
        self.gamma = GAMMA # focus on future rewards, high tends to overfit

        self.learning_rate = learning_rate
        # self.learning_rate_max = 0.003
        # self.learning_rate_min = 0.0003 # minimum at 0.003
        self.learning_rate_min = 1e-3 # minimum at 0.003
        self.learning_rate_decay = 0.999


        # self.learning_rate = 0.0003
        # self.learning_rate_max = 0.00005
        # self.learning_rate_min = 0.0000005
        # self.learning_rate_decay = 2 - 0.9995

        self.model = self._build_model()
        self.model_tar = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        # self.optimizer = optim.SGD(self.model.parameters(), lr = self.learning_rate, momentum=self.learning_rate_decay)

        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: self.learning_rate_decay**epoch) # Decay
        # self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)                             # No change
        # self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.learning_rate_min, max_lr=self.learning_rate_max)

        self.batch_size = batch_size
        self.O = np.ones((batch_size,self.state_size))
        self.Next_O = np.ones((batch_size,self.state_size))
        self.n_epochs = 5

    def _build_model(self, hidden_dims = 64):
        model = Model(self.state_size,hidden_dims,self.action_size)
        model = model.cuda()
        return model
        
        # Old tf keras code
        # model=Sequential()
        # model.add(Dense(128,input_dim = self.state_size, activation = 'relu'))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(24, activation='relu'))
        # model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse',optimizer=Adam(learning_rate=self.learning_rate))
    
        return model
    
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
        
    def act(self,state):
        if np.random.rand()<=self.epsilon:
            return int(np.random.rand() < 0.055)
        act_values = self.model(torch.Tensor(state).cuda())
        return np.argmax(act_values.cpu().data[0])
        
    def replay_every_action(self,batch_size):
        """Old code"""
        if len(self.memory) < batch_size:
            return
        # self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate) ## CANNOT CHANGE OPTIMIZER!!! will die
        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            q_values = self.model(torch.Tensor(state).cuda())
            target_q_values = self.model_tar(torch.Tensor(next_state).cuda()).max(dim = 1)[0]
            target_q_values = np.array(target_q_values.cpu().data)

            expected_q = np.array(q_values.cpu().data)
            expected_q[0][action] = reward + (1-done)*self.gamma*target_q_values
            
            self.loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean() # Mean square error loss
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay

        # LEARNING RATE decay with LR scheduler
        if self.learning_rate>self.learning_rate_min:
            self.learning_rate*=self.learning_rate_decay
            self.lr_scheduler.step()
            # print('lr', self.optimizer.param_groups[0]["lr"])

        # Learning rate cyclic (doesn't work!!! need change lr_scheduler)
        # if self.learning_rate_decay>1 and self.learning_rate > self.learning_rate_max:
        #     self.learning_rate_decay = 1/self.learning_rate_decay
        # if self.learning_rate_decay<1 and self.learning_rate < self.learning_rate_min:
        #     self.learning_rate_decay = 1/self.learning_rate_decay
        # self.learning_rate*=self.learning_rate_decay
        
        # return 

    def replay(self, batch_size):
        # Buff is:
        # buff.add(np.array(obs),action,reward,np.array(next_obs),adj,next_adj,terminated)

        if len(self.memory) < batch_size:
            return

        for _ in range(self.n_epochs):
            batch = random.sample(self.memory, batch_size)

            for i, sample in enumerate(batch):
                state, action, reward, next_state, done = sample
                self.O[i] = state
                self.Next_O[i] = next_state

            q_values = self.model(torch.Tensor(self.O).cuda())
            target_q_values = self.model_tar(torch.Tensor(self.Next_O).cuda()).max(dim = 1)[0]
            target_q_values = np.array(target_q_values.cpu().data)
            expected_q = np.array(q_values.cpu().data)
            
            for j, sample in enumerate(batch):
                state, action, reward, next_state, done = sample
                expected_q[j][action] = reward + (1-done)*self.gamma*target_q_values[j]
            
            self.loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
            
        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay

        # LEARNING RATE decay with lr_scheduler
        if self.learning_rate>self.learning_rate_min:
            self.learning_rate*=self.learning_rate_decay
            self.lr_scheduler.step()
            

    def replay_old(self, batch_size):
        # Reference DGN Code

        # for j in range(batch_size):
        #     sample = batch[j]
        #     self.O[j] = sample[0]
        #     self.Next_O[j] = sample[3]

        # q_values = self.model(torch.Tensor(self.O).cuda())
        # target_q_values = self.model_tar(torch.Tensor(self.Next_O).cuda())[0]
        # target_q_values = np.array(target_q_values.cpu().data)
        # expected_q = np.array(q_values.cpu().data)
        
        # for j in range(batch_size):
        #     sample = batch[j]
        #     action = sample[1]
        #     reward = sample[2]
        #     done = sample[4]
        #     expected_q[j][action] = reward + (1-done)*GAMMA*target_q_values[j]
        
        # loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean() # Mean square error loss
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # Old TF code
        # if len(self.memory) < batch_size:
        #     return
        # minibatch = random.sample(self.memory,batch_size)
        # for state,action,reward,next_state,done in minibatch:
        #     target=reward
        #     if not done:
        #         target = reward + self.gamma * np.amax(self.model_tar(next_state)[0])
        #     target_f = self.model(state)
        #     target_f[0][action]=target
        #     self.model.fit(state,target_f,epochs=1,verbose=0)
            
        # if self.epsilon>self.epsilon_min:
        #     self.epsilon*=self.epsilon_decay
        
        #Cyclic learning rate
        # if self.learning_rate > self.learning_rate_max:
        #     #decay 2 times faster if learning rate starts high
        #     self.learning_rate *= self.learning_rate_decay

        #reverse learning rate for cyclic (more efficient)
        # if self.learning_rate_decay>1 and self.learning_rate > self.learning_rate_max:
        #     self.learning_rate_decay = 1/self.learning_rate_decay
        # if self.learning_rate_decay<1 and self.learning_rate < self.learning_rate_min:
        #     self.learning_rate_decay = 1/self.learning_rate_decay
        # self.learning_rate*=self.learning_rate_decay
        pass
 
    def replay_last_n(self, n=10):
        # self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        # if len(self.memory) < n:
        #     return
        batch = random.sample(list(islice(self.memory, len(self.memory) - n, len(self.memory))),2)
        for state, action, reward, next_state, done in batch:
            q_values = self.model(torch.Tensor(state).cuda())
            target_q_values = self.model_tar(torch.Tensor(next_state).cuda()).max(dim = 1)[0]
            target_q_values = np.array(target_q_values.cpu().data)

            expected_q = np.array(q_values.cpu().data)
            expected_q[0][action] = reward + (1-done)*self.gamma*target_q_values
            
            self.loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean() # Mean square error loss
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

        return 
        
    def replay_last(self, lr=0):
        if lr:
            # self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate) # FAILS! cannot change optimizer
            self.optimizer.param_groups[0]["lr"] = lr

        batch = (self.memory[-1],)
        for state, action, reward, next_state, done in batch:
            # print(reward)
            q_values = self.model(torch.Tensor(state).cuda())
            target_q_values = self.model_tar(torch.Tensor(next_state).cuda()).max(dim = 1)[0]
            target_q_values = np.array(target_q_values.cpu().data)

            expected_q = np.array(q_values.cpu().data)
            expected_q[0][action] = reward + (1-done)*self.gamma*target_q_values
            
            self.loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean() # Mean square error loss
            # print('lr', self.optimizer.param_groups[0]["lr"])
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

        self.optimizer.param_groups[0]["lr"] = current_lr

        return 
            

    def update_target_model(self):
        self.model_tar.load_state_dict(self.model.state_dict())

    def save(self,name):
        torch.save(self.model,name)
 
    def load(self,name):
        self.model = torch.load(name) 

    def load_reset(self, filename):
        self.load(filename)
        self.epsilon = 0.0
 
    def load_memory(self, episode_num):
        with shelve.open(f'{output_dir}memory.pickle') as file:
            self.memory = file.get(f'{episode_num}') or self.memory

    def load_all(self, episode_num):
        self.load_reset('{}agent_{:05d}.hdf5'.format(output_dir,episode_num))
        self.load_memory(episode_num)
 