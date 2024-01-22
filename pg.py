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
# Constants
GAMMA = 0.99

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

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=1e-2, seed=0):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        torch.manual_seed(seed)
        self.linear1 = nn.Linear(num_inputs, num_actions)
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.softmax(self.linear1(state), dim=1)
        #x = F.softmax(self.linear2(x), dim=1)
        return x 
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob


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
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)
    
    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()


env_name = 'Taxi-v3'
env_name = 'gym_examples/GridWorld-v0'
env = gym.make(env_name)
torch.manual_seed(args.seed)
if env_name == 'Taxi-v3':
    policy_net = PolicyNetwork(4, 6, 128, seed=args.seed)
else:
    policy_net = PolicyNetwork(2, 4, 128, seed=args.seed)

max_episode_num = 200
max_steps = 200
numsteps = []
avg_numsteps = []
all_rewards = []

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
            action, log_prob = policy_net.get_action(state)
        else:
            action, log_prob = policy_net.get_action(state['agent'])

        new_state, reward, done, _ = env.step(action)
        if env_name == 'Taxi-v3':
            new_state = np.asarray(list(env.decode(new_state)))
        log_probs.append(log_prob)
        rewards.append(reward)
        reward_sum += (GAMMA**steps)*reward

        if done or (steps == max_steps):
            update_policy(policy_net, rewards, log_probs)
            break
            
    if episode % 10 == 0:
        numsteps.append(steps)
        avg_numsteps.append(np.mean(numsteps[-10:]))
        all_rewards.append(reward_sum)
        sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3),  np.round(np.mean(all_rewards[-10:]), decimals = 3), steps))
            
        
        state = new_state
with open('./experiment_data/pg_' + str(args.seed) + '.pkl', 'wb') as f:
    pickle.dump({'perf_history': all_rewards, 'timestep_history': numsteps}, f)
        