import matplotlib.pyplot as plt 
import pickle as pkl 
import numpy as np 


gamma = 0.99
size = 5

timedata = np.zeros((5, 500))
rewarddata = np.zeros((5, 500))

for i in range(5):
	with open('./runs/ldg_' + str(i) + '_' + str(gamma) + '_' + str(size) + '.pkl', 'rb') as f:
		data = pkl.load(f)
	temp_time_d = np.asarray(data[1])
	temp_reward_d = np.asarray(data[0])
	timedata[i, :] = temp_time_d
	rewarddata[i, :] = temp_reward_d
fig, ax = plt.subplots(1, 2)

mean_timedata = np.mean(timedata, axis=0)
print (np.min(timedata))
std_timedata = np.std(timedata, axis=0)
print (np.shape(std_timedata))

ax[0].plot(mean_timedata, color='#006d2c')
ax[0].fill_between(np.arange(500), mean_timedata-std_timedata, mean_timedata + std_timedata, color='#006d2c', alpha=0.1)

mean_rewardsdata = np.mean(rewarddata, axis=0)

std_rewarddata = np.std(rewarddata, axis=0)

ax[1].plot(mean_rewardsdata, color='#006d2c')
ax[1].fill_between(np.arange(500), mean_rewardsdata-std_rewarddata, mean_rewardsdata + std_rewarddata, color='#006d2c', alpha=0.1)

ax[0].set_ylabel('Average Timesteps')
ax[1].set_ylabel('Discounted Cumulative Rewards')

ax[0].set_xlabel('Gradient Steps')
ax[1].set_xlabel('Gradient Steps')

ax[0].grid(visible=True)
ax[1].grid(visible=True)

fig.tight_layout()

plt.savefig('performance_' + str(size) + '_times_' + str(size) + '.png') 











