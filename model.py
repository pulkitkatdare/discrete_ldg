
import copy
import torch  
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from misc import seed_everything, layer_init
from optimizer_network import Fetaw
from misc import layer_init, seed_everything

CUDA = False #torch.cuda.is_available()

softmax = nn.Softmax(dim=0)


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, learning_rate=1e-4, seed=0):
        super(PolicyNetwork, self).__init__()
        self.learning_rate = learning_rate
        self.num_actions = num_actions
        seed_everything(seed)
        torch.manual_seed(seed)
        self.linear1 = nn.Sequential(nn.Linear(num_inputs, num_actions))
        self.lr = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0)
        self.total_num_params = 0
        for params in self.parameters():
            self.total_num_params += params.numel()
            
            

    def forward(self, state):
        x = F.softmax(self.linear1(state), dim=1)
        return x 
    
    def get_action(self, state, action = None, train_time=True):
        #state = torch.from_numpy(state).float().unsqueeze(0)
       #state = state['agent']
        state = torch.tensor(state.reshape(-1, 2)).type(torch.DoubleTensor)
        probs = self.forward((state))
        highest_prob_action = torch.multinomial(probs.detach(), num_samples=1, replacement=True)
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        #print (probs.squeeze(0)[highest_prob_action])
        #print (log_prob)
        
        if train_time is False:
            self.optimizer.zero_grad()
            gradients_values = (torch.autograd.grad(log_prob, self.parameters(), create_graph=True, retain_graph=True))
            #log_prob.backward(retain_graph=True)
            gradients = []
            for count, params in enumerate(self.parameters()):
                gradients.append(copy.deepcopy(gradients_values[count].detach().cpu().numpy().reshape(-1)))
            self.optimizer.zero_grad()
            return highest_prob_action[0, 0].item(), log_prob , copy.deepcopy(gradients)
        else:
            return highest_prob_action, probs
        
    def get_action_gradient(self, state, action):
        #state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward((state))
        log_prob = torch.log(probs[0, action])
        #print (probs.squeeze(0)[highest_prob_action])
        #print (log_prob)
        
        self.optimizer.zero_grad()
        gradients_values = (torch.autograd.grad(log_prob, self.parameters(), create_graph=True, retain_graph=True))
        #log_prob.backward(retain_graph=True)
        gradients = []
        for count, params in enumerate(self.parameters()):
            gradients.append(copy.deepcopy(gradients_values[count].detach().cpu().numpy().reshape(-1)))
        self.optimizer.zero_grad()
        return probs[0, action] , copy.deepcopy(np.concatenate(gradients, axis=0))


class LDGNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, learning_rate=1e-1, seed=0):
        super(LDGNetwork, self).__init__()
        self.learning_rate = learning_rate
        self.num_actions = num_actions
        seed_everything(seed)
        torch.manual_seed(seed)
        self.linear1 = nn.Linear(num_inputs, num_actions, bias=False)
        torch.nn.init.orthogonal_(self.linear1.weight)
        #self.linear1.weight.data = torch.zeros(num_actions, num_inputs)
        self.lr = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0)
        self.total_num_params = 0
        for params in self.parameters():
            self.total_num_params += params.numel()
            
            

    def forward(self, state_features):
        x = torch.tanh(self.linear1(state_features))
        return x 

class FDGNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, learning_rate=1e-2, seed=0):
        super(FDGNetwork, self).__init__()
        self.learning_rate = learning_rate
        self.num_actions = num_actions
        seed_everything(seed)
        torch.manual_seed(seed)
        self.linear1 = nn.Sequential(nn.Linear(num_inputs, num_actions, bias=False))
        self.lr = learning_rate
        
        self.total_num_params = 0
        self.eta = nn.Parameter(torch.normal(0.0, 1.0, (1, num_actions)))
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.00)
        for params in self.parameters():
            self.total_num_params += params.numel()
            
            

    def forward(self, state_features):
        x = self.linear1(state_features)
        return x 
       
