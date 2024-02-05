import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from misc import layer_init, seed_everything
import argparse
import gym_examples
import pickle 
from model import PolicyNetwork, LDGNetwork, FDGNetwork
# Constants
GAMMA = 1.0

parser = argparse.ArgumentParser()
parser.add_argument(
        "--seed", type=int, default=0, help="seeding of the code"
    )
parser.add_argument('--batch_size', type=int, default=128, help='batch size of the code')
parser.add_argument('--k_batch_size', type=int, default=1, help='k_batch_size')
parser.add_argument('--my_implementation', type=bool, default=False, help='which version of the algorithm to use')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--max_episodes', type=int, default=3000, help='number of training episodes')
parser.add_argument('--max_time_steps', type=int, default=200, help='maximum number of time step per episode')
parser.add_argument('--lam', type=float, default=0.1, help='regularization factor 1')
parser.add_argument('--lam2', type=float, default=0.1, help='regularization factor 2')

args = parser.parse_args()
seed_everything(args.seed)

def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0 
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)
        
    discounted_rewards = torch.tensor(discounted_rewards)
    #discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)
    
    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()



env_name = 'gym_examples/GridWorld-v0'
env = gym.make(env_name, size=5)
#torch.manual_seed(args.seed)
if env_name == 'Taxi-v3':
    policy_net = PolicyNetwork(4, 6, 128, seed=args.seed)
else:
    policy_net = PolicyNetwork(2, 4, 0.05, args.seed).double()

max_episode_num = 200
max_steps = 200
gamma_sensitivity = []
gamma_sensitivity.append([])
gamma_sensitivity.append([])

for episode in range(max_episode_num):
    #print (episode)
    state = env.reset()
    if env_name == 'Taxi-v3':
        state = np.asarray(list(env.decode(state)))
    log_probs = []
    rewards = []
    reward_sum = 0.0

    for steps in range(max_steps):
        #env.render()
        if env_name == 'Taxi-v3':

            action, log_prob, _ = policy_net.get_action(state)
        else:
            action, log_prob, _ = policy_net.get_action(state['agent'], train_time=False)

        new_state, reward, done, _ = env.step(action)
        if env_name == 'Taxi-v3':
            new_state = np.asarray(list(env.decode(new_state)))
        log_probs.append(log_prob)
        rewards.append(reward)
        reward_sum += (GAMMA**steps)*reward

        if done or (steps == max_steps):
            update_policy(policy_net, rewards, log_probs)
            break
            
    if episode % 1 == 0:
        gamma_sensitivity[1].append(steps)
        gamma_sensitivity[0].append(reward_sum)
        sys.stdout.write("episode: {}, average_reward: {}, length: {}\n".format(episode,  reward_sum, steps))

        state = new_state
with open('./runs/pg_5_times_5_' + str(args.seed) + '.pkl', 'wb') as f:
    pickle.dump(gamma_sensitivity, f)
        