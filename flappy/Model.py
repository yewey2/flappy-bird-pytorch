import math, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class Encoder(nn.Module):
    def __init__(self, din, hidden_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(din, hidden_dim)

    def forward(self, x):
        embedding = F.relu(self.fc(x))
        return embedding

class Hidden(nn.Module):
	def __init__(self, hidden_dim, dout):
		super(Hidden, self).__init__()
		self.fc = nn.Linear(hidden_dim, dout)

	def forward(self, x):
		return F.relu(self.fc(x))

class Q_Net(nn.Module):
    def __init__(self, hidden_dim, dout):
        super(Q_Net, self).__init__()
        self.fc = nn.Linear(hidden_dim, dout)

    def forward(self, x):
        q = self.fc(x)
        return q

class Model(nn.Module):
	def __init__(self,num_inputs,hidden_dim,num_actions):
		super(Model, self).__init__()
		self.encoder = Encoder(num_inputs,hidden_dim)
		self.hidden = Hidden(hidden_dim,hidden_dim)
		# self.hidden1 = Hidden(hidden_dim, hidden_dim*2)
		# self.hidden2 = Hidden(hidden_dim*2, hidden_dim*2)
		# self.hidden3 = Hidden(hidden_dim*2, hidden_dim)
		self.q_net = Q_Net(hidden_dim,num_actions)
        
	def forward(self, x):
		h1 = self.encoder(x)
		# h2 = self.hidden1(h1)
		# h3 = self.hidden2(h2)
		# h4 = self.hidden3(h3)
		# q = self.q_net(h4)
		q = self.q_net(self.hidden(h1))
		return q 

        
if __name__ == "__main__":
	print(Model(5, 128, 3))












