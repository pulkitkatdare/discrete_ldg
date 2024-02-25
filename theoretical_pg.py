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

def reward_fn(state, size):
    reward = (-(np.abs(state[0]-(size-1)) + np.abs(state[1]-((size-1)))))
    return reward


_action_to_direction = {0: np.array([1, 0]),
        1: np.array([0, 1]),
        2: np.array([-1, 0]),
        3: np.array([0, -1]),
        }


def decode_state(pos, state_size):
    y = pos%(state_size)
    x = pos // state_size
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
def calculate_gradient(d, q, policy):
    gradient_average = np.zeros((policy.total_num_params, 1))
    R = args.size#np.shape(d)[0]
    for i in range(R*R):
        state = decode_state(i, size)
        with torch.no_grad():
            action = policy(torch.tensor(state).view(1, -1).type(torch.DoubleTensor))
        for j in range(4):
            act_prob, gradient = policy.get_action_gradient(torch.tensor(state).view(1, -1).type(torch.DoubleTensor), j)
            gradient_average += q[i, j]*d[i]*act_prob.item()*gradient.reshape(-1, 1)

    return gradient_average


def calculate_d(size, gamma, policy):
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
    if gamma < 1.0:
        A_inv = np.linalg.pinv(A)
        d = np.matmul(A_inv, b)
        d = d/np.sum(d)
    else:
        ns = null_space(A)
        print (np.shape(ns))
        alpha = 1/(np.sum(ns))
        d = alpha * ns#np.matmul(A_inv, b)#
    return d

def calculate_v(size, gamma, policy):
    A = np.zeros((size*size, size*size), dtype=np.float64)
    b = np.zeros((size*size, 1), dtype=np.float64)
    R = size
    
    for i in range(size*size):
        A[i, i] = 1.0
        state = decode_state(i, size)
        with torch.no_grad():
            action = policy(torch.tensor(state).view(1, -1).type(torch.DoubleTensor))
            for j in [0, 1, 2, 3]:
                action_prob = action[0, j].item()
                next_state = np.clip(state + _action_to_direction[j], 0, R-1)
                i_prime = encode_state(next_state, size)
                A[i, i_prime] += -gamma*action_prob
        b[i, 0] = reward_fn(state, size)
    
    A_inv = np.linalg.inv(A)
    v = np.matmul(A_inv, b)
    
    return v


CUDA = False #torch.cuda.is_available()

softmax = nn.Softmax(dim=0)

parser = argparse.ArgumentParser()
parser.add_argument(
        "--seed", type=int, default=0, help="seeding of the code"
    )
parser.add_argument('--batch_size', type=int, default=128, help='batch size of the code')
parser.add_argument('--k_batch_size', type=int, default=1, help='k_batch_size')
parser.add_argument('--my_implementation', type=bool, default=False, help='which version of the algorithm to use')
parser.add_argument('--gamma', type=float, default=1.0, help='discount factor')
parser.add_argument('--gamma2', type=float, default=0.95, help='discount factor')
parser.add_argument('--max_episodes', type=int, default=3000, help='number of training episodes')
parser.add_argument('--max_time_steps', type=int, default=200, help='maximum number of time step per episode')
parser.add_argument('--lam', type=float, default=0.1, help='regularization factor 1')
parser.add_argument('--lam2', type=float, default=0.1, help='regularization factor 2')
parser.add_argument('--size', type=int, default=5, help='regularization factor 2')


args = parser.parse_args()
seed_everything(args.seed)   

if __name__ == "__main__":
    seed_everything(args.seed)
    policy = PolicyNetwork(2, 4, 0.05, args.seed).double()
    if CUDA:
        policy = policy.to('cuda:0')
    #agent = LogDensityGradient(policy_net=policy)
    #run_steps(agent)
    size = args.size
    env = gym.make('gym_examples/GridWorld-v0', size=size)
    gamma = args.gamma
    size = args.size#env.size
    action_space = 4
    perf_history = []
    timestep_history = []
    density_perf = []
    R = args.size
    gamma_sensitivity = []
    gamma_sensitivity.append([])
    gamma_sensitivity.append([])
    gamma_sensitivity.append([])
    for iteration in range(200):
        d = calculate_d(size, 1.0, policy)
        gamma = args.gamma2
        v = calculate_v(size, gamma, policy)
        q = np.zeros((size*size, 4))
        for i in range(size*size):
            state = decode_state(i, size)
            reward = reward_fn(state, size)
            for action in [0, 1, 2, 3]:
                next_state = np.clip(state + _action_to_direction[action], 0, R-1)
                j = encode_state(next_state, size)
                q[i, action] = reward + gamma*v[j, 0]

        perf = calc_perf(d)
        gradient = calculate_gradient(d, q, policy)
        total_params = 0
        if (iteration%1) == 0:
            average_rewards = []
            average_timesteps = []
            for i in range(1):
                seed_everything(i)   
                state = env.reset()
                done = False
                reward_sum = 0
                time_step = 0
                while (time_step < 200):
                    action, _, _ = policy.get_action(state['agent'], train_time=False)
                    state, reward, done, info = env.step(action)
                    if done: 
                        break 

                    #print (action[1])
                    
                    reward_sum += reward
                    time_step += 1
                
                average_rewards.append(reward_sum)
                average_timesteps.append(time_step)
            print (iteration, np.mean(average_rewards), np.mean(average_timesteps), perf)
            gamma_sensitivity[1].append( np.mean(average_timesteps))
            gamma_sensitivity[0].append( np.mean(average_rewards))
            gamma_sensitivity[2].append(perf)
            perf_history.append(np.mean(average_rewards))
            timestep_history.append(np.mean(average_timesteps))
            density_perf.append(perf)
        for params in policy.parameters():
            params.data += 0.05*(torch.tensor(gradient[total_params:total_params + params.numel(), 0]).type(torch.DoubleTensor)).view(params.size())# - 0.01*params.data
            total_params += params.numel()
    with open('./runs/theoretical_pg_' + str(args.size) + '_times_' + str(args.size) + '_' + str(args.seed) + '.pkl', 'wb') as f:
        pickle.dump(gamma_sensitivity, f)
'''
import matplotlib.pyplot as plt 
plt.plot(range(0, 20*len(perf_history), 20), perf_history, color='#2ca25f')
plt.xlabel('training steps')
plt.xticks(np.arange(0, 20*len(perf_history), step=20))
plt.ylabel('Average rewards')
plt.grid(True)
plt.savefig('./experiment_data/ldg_performance.png')

with open('./eval_theory_performance/pg_theoretical_' + str(args.seed) + '.pkl', 'wb') as f:
    pickle.dump({'perf_history': perf_history, 'timestep_history': timestep_history, 'density_perf': density_perf}, f)

'''
    
    
    
    
    
    
    