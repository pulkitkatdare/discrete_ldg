from misc import layer_init, seed_everything
from model import PolicyNetwork
from optimizer_network import Fetaw
import sys
import matplotlib.pyplot as plt
import pickle 
import argparse
import torch.nn as nn 
import gym 
import numpy as np 
import gym_examples
import torch 
import numpy as np
from scipy.linalg import null_space

CUDA = False #torch.cuda.is_available()

def reward_fn(state, size):
    reward = (-(np.abs(state[0]-(size-1)) + np.abs(state[1]-((size-1)))))
    return reward

softmax = nn.Softmax(dim=0)

parser = argparse.ArgumentParser()
parser.add_argument(
        "--seed", type=int, default=0, help="seeding of the code"
    )
parser.add_argument('--batch_size', type=int, default=128, help='batch size of the code')
parser.add_argument('--k_batch_size', type=int, default=1, help='k_batch_size')
parser.add_argument('--my_implementation', type=bool, default=False, help='which version of the algorithm to use')
parser.add_argument('--gamma', type=float, default=1.0, help='discount factor')
parser.add_argument('--max_episodes', type=int, default=3000, help='number of training episodes')
parser.add_argument('--max_time_steps', type=int, default=200, help='maximum number of time step per episode')
parser.add_argument('--lam', type=float, default=0.1, help='regularization factor 1')
parser.add_argument('--lam2', type=float, default=0.1, help='regularization factor 2')
parser.add_argument('--size', type=int, default=5, help='regularization factor 2')


args = parser.parse_args()
seed_everything(args.seed)   


def decode_state(pos, state_size):
    y = pos%(state_size)
    x = pos // state_size
    #print (x, y)
    return [x, y]

def encode_state(state, state_size):
    return state_size*state[0] + state[1]

def calc_perf(d):
    R = args.size#np.shape(d)[0]
    perf = 0.0
    for i in range(R*R):
        state = decode_state(i, R)
        reward = (-(np.abs(state[0]-(R-1)) + np.abs(state[1]-((R-1)))))
        perf += d[i, 0]*(reward)
    return perf
def calculate_gradient(d, g):
    gradient = np.zeros((np.shape(g)[1], 1))
    R = args.size
    for i in range(R*R):
        state = decode_state(i, R)
        reward = reward_fn(state, size)
        gradient  += d[i, 0]*g[i, :].reshape(-1, 1)*reward
    return gradient
            
        
    
    
    



if __name__ == "__main__":
    seed_everything(args.seed)   
    policy = PolicyNetwork(2, 4, 3e-4, args.seed).double()
    if CUDA:
        policy = policy.to('cuda:0')
    #agent = LogDensityGradient(policy_net=policy)
    #run_steps(agent)
    size = args.size
    env = gym.make('gym_examples/GridWorld-v0', size=size)
    gamma = args.gamma
    action_space = 4
    perf_history = []
    timestep_history = []
    density_perf = []
    for iteration in range(400):
        A = np.zeros((size*size, size*size), dtype=np.float64)
        b = np.zeros((size*size, 1), dtype=np.float64)
        for i in range(size*size):
            if i == 0:
                b[i, 0] = (1 - gamma)
            A[i, i] = 1.0
            state = decode_state(i, size)
            possible_states = []
            possible_actions = []
            
            poss_state = state.copy()
            poss_state[0] = min(max(state[0]-1, 0), size-1)
            possible_states.append(poss_state)
            j = encode_state(poss_state, size) #possible_states.append((poss_state, encode_state(poss_state, size)))
            with torch.no_grad():
                action = policy(torch.tensor(poss_state).view(1, -1).type(torch.DoubleTensor))
            if j != i:
                act_prob = action[0, 0]
            else:
                act_prob = action[0, 2]
            A[i, j] += -gamma*act_prob.item()
            
            poss_state = state.copy()
            poss_state[0] = min(max(state[0]+1, 0), size-1)
            possible_states.append(poss_state)
            j = encode_state(poss_state, size) #possible_states.append((poss_state, encode_state(poss_state, size)))
            with torch.no_grad():
                action = policy(torch.tensor(poss_state).view(1, -1).type(torch.DoubleTensor))
            if j != i:
                act_prob = action[0, 2]
            else:
                act_prob = action[0, 0]
            A[i, j] += -gamma*act_prob.item()
            
            poss_state = state.copy()
            poss_state[1] = min(max(state[1]-1, 0), size-1)
            possible_states.append(poss_state)
            j = encode_state(poss_state, size) #possible_states.append((poss_state, encode_state(poss_state, size)))
            with torch.no_grad():
                action = policy(torch.tensor(poss_state).view(1, -1).type(torch.DoubleTensor))
            if j != i:
                act_prob = action[0, 1]
            else:
                act_prob = action[0, 3]
            A[i, j] += -gamma*act_prob.item()
            
            poss_state = state.copy()
            poss_state[1] = min(max(state[1]+1, 0), size-1)
            possible_states.append(poss_state)
            j = encode_state(poss_state, size) #possible_states.append((poss_state, encode_state(poss_state, size)))
            with torch.no_grad():
                action = policy(torch.tensor(poss_state).view(1, -1).type(torch.DoubleTensor))
            if j != i:
                act_prob = action[0, 3]
            else:
                act_prob = action[0, 1]
            A[i, j] += -gamma*act_prob.item()
            
        #print (np.linalg.det(A + 0.01*np.eye(size*size)))
        
        if gamma < 1.0:
            A_inv = np.linalg.inv(A)
            d = np.matmul(A_inv, b)
            d = d/np.sum(d)
        else:
            ns = null_space(A)
            d = ns/np.sum(ns)
        #d = d/np.sum(d)
        #print (A[12, 12], A[17, 17])
        
        A = np.zeros((size*size, size*size), dtype=np.float64)
        b = np.zeros((size*size,policy.total_num_params), dtype=np.float64)
        
        for i in range(size*size):
            A[i, i] = d[i, 0]
            state = decode_state(i, size)
            
            poss_state = state.copy()
            poss_state[0] = min(max(state[0]-1, 0), size-1)
            j = encode_state(poss_state, size) #possible_states.append((poss_state, encode_state(poss_state, size)))
            if j != i:
                act_prob, gradient = policy.get_action_gradient(torch.tensor(poss_state).view(1, -1).type(torch.DoubleTensor), 0)
            else:
                act_prob, gradient = policy.get_action_gradient(torch.tensor(poss_state).view(1, -1).type(torch.DoubleTensor), 2)
            
            A[i, j] += -gamma*act_prob.item()*d[j, 0]
            b[i, :] += gamma*act_prob.item()*d[j, 0]*gradient
            
            poss_state = state.copy()
            poss_state[0] = min(max(state[0]+1, 0), size-1)
            j = encode_state(poss_state, size) #possible_states.append((poss_state, encode_state(poss_state, size)))
            if j != i:
                act_prob, gradient = policy.get_action_gradient(torch.tensor(poss_state).view(1, -1).type(torch.DoubleTensor), 2)
            else:
                act_prob, gradient = policy.get_action_gradient(torch.tensor(poss_state).view(1, -1).type(torch.DoubleTensor), 0)
            A[i, j] += -gamma*act_prob.item()*d[j, 0]
            b[i, :] += gamma*act_prob.item()*d[j, 0]*gradient
            
            poss_state = state.copy()
            poss_state[1] = min(max(state[1]-1, 0), size-1)
            j = encode_state(poss_state, size) #possible_states.append((poss_state, encode_state(poss_state, size)))
            if j != i:
                act_prob, gradient = policy.get_action_gradient(torch.tensor(poss_state).view(1, -1).type(torch.DoubleTensor), 1)
            else:
                act_prob, gradient = policy.get_action_gradient(torch.tensor(poss_state).view(1, -1).type(torch.DoubleTensor), 3)
            A[i, j] += -gamma*act_prob.item()*d[j, 0]
            b[i, :] += gamma*act_prob.item()*d[j, 0]*gradient
            
            poss_state = state.copy()
            poss_state[1] = min(max(state[1]+1, 0), size-1)
            j = encode_state(poss_state, size) #possible_states.append((poss_state, encode_state(poss_state, size)))
            if j != i:
                act_prob, gradient = policy.get_action_gradient(torch.tensor(poss_state).view(1, -1).type(torch.DoubleTensor), 3)
            else:
                act_prob, gradient = policy.get_action_gradient(torch.tensor(poss_state).view(1, -1).type(torch.DoubleTensor), 1)
                
            A[i, j] += -gamma*act_prob.item()*d[j, 0]
            b[i, :] += gamma*act_prob.item()*d[j, 0]*gradient
    
        if gamma < 1.0:
            A_inv = np.linalg.inv(A)
            g = np.matmul(A_inv, b)
            g_sum = np.matmul(d.T, g)
        else:
            A_inv = np.linalg.pinv(A)
            g = np.matmul(A_inv, b)
            g_sum = np.matmul(d.T, g)
            g = g - g_sum
        perf = calc_perf(d)
        #print (perf, np.sum(d))
        gradient = calculate_gradient(d, g)
        total_params = 0
        if (iteration%20) == 0:
            average_rewards = []
            average_timesteps = []
            for i in range(5):
                seed_everything(i)   
                state = env.reset()
                done = False
                reward_sum = 0
                time_step = 0
                while (time_step < 200):
                    action = policy.get_action(state)
                    state, reward, done, info = env.step(action[0].item())
                    reward = reward
                    #print (action[1])
                    
                    reward_sum += (gamma**time_step)*reward
                    time_step += 1
                if gamma == 1:
                    average_rewards.append(reward_sum/200)
                else:
                    average_rewards.append((1-gamma)*reward_sum)
                average_timesteps.append(time_step)
            print (np.mean(average_rewards), np.mean(average_timesteps), perf)
            perf_history.append(np.mean(average_rewards))
            timestep_history.append(np.mean(average_timesteps))
            density_perf.append(perf)

        for params in policy.parameters():
            params.data += 0.1*(torch.tensor(gradient[total_params:total_params + params.numel(), 0]).type(torch.DoubleTensor)).view(params.size())# - 0.01*params.data
            total_params += params.numel()
import matplotlib.pyplot as plt 
plt.plot(range(0, 20*len(perf_history), 20), perf_history, color='#2ca25f')
plt.xlabel('training steps')
plt.xticks(np.arange(0, 20*len(perf_history), step=20))
plt.ylabel('Average rewards')
plt.grid(True)
plt.savefig('./experiment_data/ldg_performance.png')
with open('./eval_theory_performance/ldg_discrete_theoretical_' + str(args.seed) + '.pkl', 'wb') as f:
    pickle.dump({'perf_history': perf_history, 'timestep_history': timestep_history, 'density_perf': density_perf}, f)


    
    
    
    
    
    
    