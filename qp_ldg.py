from misc import layer_init, seed_everything
from model import PolicyNetwork, LDGNetwork, FDGNetwork
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
from deep_rl.component.replay import Replay


CUDA = False #torch.cuda.is_available()

def reward_fn(state, size):
    reward = (-(np.abs(state[0]-(size-1)) + np.abs(state[1]-((size-1)))))
    return reward

softmax = nn.Softmax(dim=0)

parser = argparse.ArgumentParser()
parser.add_argument(
        "--seed", type=int, default=0, help="seeding of the code"
    )
parser.add_argument('--batch_size', type=int, default=256, help='batch size of the code')
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


def collect_data(env, replay, policy, state_size=5, num_episodes=10, num_steps=200):
    average_rewards = []
    average_timesteps = []
    for i in range(num_episodes):
            #print('Sampling trajectory %s' % (i))
            env.seed(i)
            states = env.reset()
            reward_sum = 0.0
            for j in range(num_steps):
                action, log_prob , gradients = policy.get_action(states['agent'], train_time=False)
                next_states, rewards, done, info = env.step(action)
                rewards = reward_fn(next_states['agent'], state_size)
                #print (rewards)
                
                #rewards = 10*rewards
                reward_sum += rewards
                if done or j == num_steps-1:
                    average_rewards.append(reward_sum)
                    average_timesteps.append(j)
                    break
                input_states = np.reshape(states['agent'], (1, 2))
                input_rewards = np.zeros((1, 1))
                input_rewards[0, 0] = (gamma**j)*rewards
                input_actions = np.reshape(action, (1, 1))#
                #input_actions[0, :] = action
                #next_states = next_states['agent'].copy()
                input_next_states = np.reshape(next_states['agent'], (1, 2))
                input_done = np.zeros((1, 1))
                input_done[0, 0] = done
                
                input_log_probs = np.zeros((1, 1))
                input_log_probs[0, 0] = log_prob.item()
                experiences = list(zip(input_states, input_actions, input_log_probs, input_rewards, input_next_states, input_done))
                replay.feed_batch(experiences, gradients)
                states = next_states

    return replay, np.mean(average_timesteps)




def decode_state(pos, state_size, action_size):
    action = pos % action_size
    y = pos%(state_size)
    x = pos // state_size
    #print (x, y)
    return [x, y], action

def encode_state_action(state, action, state_size, action_size):
    return (action_size*(state_size*state[0] + state[1]) + action)

if __name__ == "__main__":
    seed_everything(args.seed)   
    size = args.size
    state_size = size
    env = gym.make('gym_examples/GridWorld-v0', size=size)

    gamma = args.gamma
    batch_size = args.batch_size
    action_space = 4
    action_size  = action_space
    regularization = 0.1
    policy = PolicyNetwork(2, 4, 3e-4, args.seed).double()
    if CUDA:
        policy = policy.to('cuda:0')
    
    feature_matrix = torch.eye(state_size*state_size*action_space)
    w_ldg = LDGNetwork(num_inputs=state_size*state_size*action_space, num_actions=3*4)
    f_ldg = FDGNetwork(num_inputs=state_size*state_size*action_space, num_actions=3*4)
    for step in range(500):
        
        replay = Replay(memory_size=int(1e6), batch_size=batch_size)
        replay, average_timesteps = collect_data(env, replay, policy)
        for _ in range(10):
            
            batch, gradients = replay.sample()
            state, action, log_prob, rewards, next_state, done = batch
            next_action = policy.get_action(torch.tensor(next_state).type(torch.FloatTensor), train_time=True)[0]
            gradients = [torch.tensor(grad).type(torch.FloatTensor).view(batch_size, -1, 1) for grad in gradients]
            gradients = torch.cat(gradients, dim=1)

            encoded_state_action = action_size*(state_size * state[:, 0] + state[:, 1]) + action.reshape(-1)
            encoded_state_action_feature = feature_matrix[encoded_state_action].detach()
            encoded_next_state_action = action_size*(state_size * next_state[:, 0] + next_state[:, 1]) + next_action.detach().numpy().reshape(-1)
            encoded_next_state_action_feature = feature_matrix[encoded_next_state_action].detach()
            
            w_sa = w_ldg(encoded_state_action_feature).view(batch_size, -1, 1)
            f_sa = f_ldg(encoded_state_action_feature).view(batch_size, 1, -1)
            f_sa_prime = f_ldg(encoded_next_state_action_feature).view(batch_size, 1, -1)
            eta = f_ldg.eta 

            w_loss_t1 = torch.mean(torch.bmm(f_sa.detach(), w_sa)) 
            w_loss_t2 = - torch.mean(torch.bmm(f_sa.detach(), gradients.detach()))
            w_loss_t3 = - gamma * torch.mean(torch.bmm(f_sa_prime.detach(), w_sa))
            w_loss_t4 = - 0.5 * torch.mean((torch.bmm(f_sa.detach(), f_sa.detach().permute(0, 2, 1))))
            w_loss_t5 = regularization * (torch.matmul(eta.detach(), torch.mean(w_sa, dim=0)) - 0.5 * torch.matmul(eta.detach(), eta.detach().transpose(1, 0).detach().view(-1, 1)))
            w_loss = w_loss_t1 + w_loss_t2 + w_loss_t3 + w_loss_t4 + w_loss_t5

            f_loss_t1 = torch.mean(torch.bmm(f_sa, w_sa.detach())) 
            f_loss_t2 = - torch.mean(torch.bmm(f_sa, gradients))
            f_loss_t3 = - gamma * torch.mean(torch.bmm(f_sa_prime, w_sa.detach()))
            f_loss_t4 = - 0.5 * torch.mean((torch.bmm(f_sa, f_sa.permute(0, 2, 1))))
            f_loss_t5 = regularization * (torch.matmul(eta, torch.mean(w_sa.detach(), dim=0)) - 0.5 * torch.matmul(eta, eta.transpose(1, 0).view(-1, 1)))
            f_loss = -(f_loss_t1 + f_loss_t2 + f_loss_t3 + f_loss_t4 + f_loss_t5)

            total_loss = w_loss + f_loss

            f_ldg.optimizer.zero_grad()
            w_ldg.optimizer.zero_grad()
            f_loss.backward()
            w_loss.backward()
            f_ldg.optimizer.step()
            w_ldg.optimizer.step()
            #print (f_ldg.eta.detach())
            print (w_loss.item())
        print (average_timesteps, w_loss.item(), )
        log_density_gradient = torch.mean(torch.bmm(w_sa, torch.tensor(rewards).view(-1, 1, 1).type(torch.FloatTensor)), dim=0)
        I = 0
        for params in policy.parameters():
            num_params = params.numel()
            grad = log_density_gradient[I:I + num_params]
            params.data += 0.1*(grad.view(params.size())) #- 0.01*params.data
            I = I + num_params












        


    
    
    
    
    
    
    