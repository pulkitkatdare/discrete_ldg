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
parser.add_argument('--batch_size', type=int, default=128, help='batch size of the code')
parser.add_argument('--k_batch_size', type=int, default=1, help='k_batch_size')
parser.add_argument('--my_implementation', type=bool, default=False, help='which version of the algorithm to use')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--max_episodes', type=int, default=3000, help='number of training episodes')
parser.add_argument('--max_time_steps', type=int, default=200, help='maximum number of time step per episode')
parser.add_argument('--lam', type=float, default=0.1, help='regularization factor 1')
parser.add_argument('--lam2', type=float, default=0.1, help='regularization factor 2')
parser.add_argument('--size', type=int, default=5, help='grid size')


args = parser.parse_args()
seed_everything(args.seed)   


def collect_data(env, replay, policy, state_size=args.size, num_episodes=5, num_steps=100):
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
                #rewards = reward_fn(next_states['agent'], state_size)
                #print (rewards)
                
                #rewards = 10*rewards
                reward_sum += ((args.gamma)**j) * rewards
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

    return replay, np.mean(average_rewards), np.mean(average_timesteps)




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
    regularization = 0.0
    alpha = 0.99
    policy = PolicyNetwork(2, 4, 0.009, args.seed).double()
    if CUDA:
        policy = policy.to('cuda:0')
    
    feature_matrix = torch.eye(state_size*state_size*action_space)
    
    #f_ldg = FDGNetwork(num_inputs=state_size*state_size*action_space, num_actions=3*4)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    gamma_sensitivity = []
    gamma_sensitivity.append([])
    gamma_sensitivity.append([])
    
    for steps in range(500):
        args.gamma = gamma
        replay = Replay(memory_size=int(1e6), batch_size=batch_size)
        replay, average_rewards, average_timesteps = collect_data(env, replay, policy, state_size=state_size)
        w_ldg = LDGNetwork(num_inputs=state_size*state_size*action_space, num_actions=3*4)
        for k in range(500):
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
            #f=_sa = f_ldg(encoded_state_action_feature).view(batch_size, 1, -1)
            #f_sa_prime = f_ldg(encoded_next_state_action_feature).view(batch_size, 1, -1)
            #eta = f_ldg.eta 
            encoded_state_action_feature_transpose = encoded_state_action_feature.view(batch_size, -1, 1)
            encoded_state_action_feature = encoded_state_action_feature.view(batch_size, 1, -1)

            encoded_next_state_action_feature = encoded_next_state_action_feature.view(batch_size, 1, -1)
            encoded_next_state_action_feature_transpose = encoded_next_state_action_feature.view(batch_size, -1, 1)

            ## Maximization Term beta 
            B = torch.bmm(encoded_state_action_feature_transpose, encoded_state_action_feature)
            B = torch.mean(B, dim=0)
            A_term1 = torch.bmm(w_sa, encoded_state_action_feature)
            A_term1 = torch.mean(A_term1, dim=0)

            A_term2 = torch.bmm(gradients, encoded_state_action_feature)
            A_term2 = -torch.mean(A_term2, dim=0)

            A_term3 = torch.bmm(w_sa, encoded_next_state_action_feature)
            A_term3 = -gamma * torch.mean(A_term3, dim=0)

            beta_transpose = torch.matmul(A_term1 + A_term2 + A_term3, torch.linalg.pinv(B + 0.0* torch.eye(4 * state_size**2, 4 * state_size**2)))
            beta = torch.transpose(beta_transpose, 1, 0)
            ## Maximization Term tau 

            tau = torch.mean(w_sa, dim=0)
            ### Minimization Step
            f_sa = torch.matmul(encoded_state_action_feature_transpose.view(batch_size, -1), beta).view(batch_size, 1, -1)
            f_sa_prime = torch.matmul(encoded_next_state_action_feature_transpose.view(batch_size, -1), beta).view(batch_size, 1, -1)

            loss_term1 = torch.bmm(f_sa, w_sa)
            loss_term1 = torch.mean(loss_term1, dim=0)

            loss_term2 = torch.bmm(f_sa, gradients)
            loss_term2 = -torch.mean(loss_term2, dim=0)

            loss_term3 = torch.bmm(f_sa_prime, w_sa)
            loss_term3 = - gamma * torch.mean(loss_term3, dim=0)

            loss_term4 = torch.bmm(f_sa, f_sa.view(batch_size, -1, 1))
            loss_term4 = - 0.5 * torch.mean(loss_term4)


            loss_term5 = regularization * 0.5 * torch.matmul(torch.transpose(tau, 1, 0), tau)

            loss_w = loss_term1 + loss_term2 + loss_term3 + loss_term4 + loss_term5

            w_ldg.optimizer.zero_grad()
            loss_w.backward()
            w_ldg.optimizer.step()
            #### Baseline Function 
            
            w_sa_transpose = w_sa.view(batch_size, 1, -1)
            w_norm_squared = torch.bmm(w_sa_transpose, w_sa)

            phi_sa_next = encoded_state_action_feature_transpose - gamma * encoded_next_state_action_feature_transpose
            S = phi_sa_next.size(1)
            psi = torch.ones(batch_size, S + 1, 1)
            psi[:, :phi_sa_next.size(1), :] = phi_sa_next
            psi_transpose = psi.view(batch_size, 1, -1)

            psi_t_t = torch.bmm(psi, psi_transpose)
            rewards_tensor = torch.tensor(rewards).view(batch_size, 1, 1).type(torch.FloatTensor)

            X_temp = torch.bmm(psi_t_t.view(batch_size, -1, 1), w_norm_squared).view(batch_size, S + 1, S + 1)
            
            y_temp = torch.bmm(psi, rewards_tensor)
            y_temp = torch.bmm(y_temp, w_norm_squared)
            if k == 0:
                X = torch.mean(X_temp, dim=0)
                y = torch.mean(y_temp, dim=0)
            else:
                X = alpha * X + torch.mean(X_temp, dim=0)
                y = alpha * y + torch.mean(y_temp, dim=0)
            v = torch.matmul(torch.linalg.pinv(X), y)
            
            rho = torch.matmul(psi.view(batch_size, -1), v)
            #print (rho.size())




        

            #print (loss_w.item(), torch.mean(w_sa, dim=0).detach())
            w_ldg.linear1.weight.data -= torch.mean(w_ldg.linear1.weight.data, dim=1).view(-1, 1)





        
        gamma_sensitivity[0].append(average_rewards)
        gamma_sensitivity[1].append(average_timesteps)
        #
        rewards = rewards #- rho.detach().numpy() #- (1-args.gamma) * average_rewards

        
        
        log_density_gradient = torch.mean(torch.bmm(w_sa, torch.tensor(rewards).view(-1, 1, 1).type(torch.FloatTensor)), dim=0)
        std = torch.std(torch.bmm(w_sa, torch.tensor(rewards).view(-1, 1, 1).type(torch.FloatTensor)), dim=0)
        print (steps, average_timesteps, average_rewards)
        #policy.optimizer.zero_grad(set_to_none=False)
        #print (np.shape(state))
        #loss = torch.mean(policy(torch.tensor(state).type(torch.DoubleTensor)))
        #loss.backward()

        I = 0
        for params in policy.parameters():
            num_params = params.numel()
            grad = log_density_gradient[I:I + num_params]
            params.grad = torch.zeros_like(params).type(torch.DoubleTensor)
            #print (params.grad.size(), params.size())
            #params.grad = -grad.view(params.size()).type(torch.DoubleTensor)
            params.data += 0.05*(grad.view(params.size())) #- 0.01*params.data
            I = I + num_params
        #policy.optimizer.step() # Changes 
        #policy.optimizer.zero_grad()
    ax[0].plot(gamma_sensitivity[0], label='gamma=' + str(args.gamma))
    ax[0].set_ylabel('Discounted Rewards')
    ax[0].set_xlabel('gradient steps')
    ax[0].grid(visible=True)

    ax[1].plot(gamma_sensitivity[1], label='gamma=' + str(args.gamma))
    ax[1].set_ylabel('Average Timesteps')
    ax[1].set_xlabel('gradient steps')
    ax[1].grid(visible=True)

    plt.savefig('perf') 

    with open('./runs/ldg_' + str(args.seed) + '_' + str(args.gamma) + '_' + str(args.size) + '.pkl', 'wb') as f:
        pickle.dump(gamma_sensitivity, f)
        
        





        
        #print (f_ldg.eta.detach())
        #'''
        #print (w_loss.item())
        #print (average_timesteps, w_loss.item(), )
        #log_density_gradient = torch.mean(torch.bmm(w_sa, torch.tensor(rewards).view(-1, 1, 1).type(torch.FloatTensor)), dim=0)
        #I = 0
        #for params in policy.parameters():
        #   num_params = params.numel()
        #    grad = log_density_gradient[I:I + num_params]
        #    params.data += 0.1*(grad.view(params.size())) #- 0.01*params.data
        #    I = I + num_params
        












        


    
    
    
    
    
    
    