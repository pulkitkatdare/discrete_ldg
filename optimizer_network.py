import sys
import random, os
import copy
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from misc import layer_init, seed_everything

import matplotlib.pyplot as plt
import pickle 
import argparse
from torch.utils.data import DataLoader


class Fetaw(nn.Module):
    def __init__(self, input,output, hidden_size=200, learning_rate=1e-2):
        super(Fetaw, self).__init__()
        # original input -> 256 -> output
        self.f = nn.Sequential(nn.Linear(input-1, 300),
                                       nn.ReLU(),
                                       layer_init(nn.Linear(300, output), 1e-3),
                                       )
        
        
        self.w = nn.Sequential(nn.Linear(input-1, 300),
                                       nn.ReLU(),
                                      layer_init(nn.Linear(300, output), 1e-3)
        )
        #self.w_layer4 = layer_init(nn.Linear(600, output), 1e-3)
        
        self.eta = nn.Parameter(torch.normal(0.0, 1.0, (1, output)))
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.10)
        self.double()
    
    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        out_f = torch.tanh((self.f(s)))
        out_w =torch.tanh((self.w(s)))
        out_eta = self.eta
        
        return out_f, out_w.pow(2), out_eta
