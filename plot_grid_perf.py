import matplotlib.pyplot as plt 
import pickle as pkl 
import numpy as np 


gamma = 1.0
size = 5 
fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=500, constrained_layout=False)

L = 200
key_name = str(size) + '_times_' + str(size) + '_'

timedata = np.zeros((5, L))
rewarddata = np.zeros((5, L))
index = 0 
for i in [0, 1, 2, 3, 4]:
	with open('./runs/pg_' + key_name + str(i)  + '.pkl', 'rb') as f:
		data = pkl.load(f)
	temp_time_d = np.asarray(data[1])[0:L]
	print (temp_time_d[-1], min(temp_time_d))
	temp_reward_d = np.asarray(data[0])[0:L]
	timedata[index, :] = temp_time_d
	rewarddata[index, :] = temp_reward_d
	index += 1

mean_timedata = np.mean(timedata, axis=0)
print (np.min(timedata))
std_timedata = np.std(timedata, axis=0)/np.sqrt(5)
print (np.shape(std_timedata))

ax[0].plot(mean_timedata, label='PG', color='#de2d26', linewidth=2, linestyle='dashed')
ax[0].fill_between(np.arange(L), mean_timedata-std_timedata, mean_timedata + std_timedata, color='#de2d26', alpha=0.1)

mean_rewardsdata = np.mean(rewarddata, axis=0)

std_rewarddata = np.std(rewarddata, axis=0)/np.sqrt(5)

ax[1].plot(mean_rewardsdata, color='#de2d26', linewidth=2, linestyle='dashed')
ax[1].fill_between(np.arange(L), mean_rewardsdata-std_rewarddata, mean_rewardsdata + std_rewarddata, color='#de2d26', alpha=0.1)

timedata = np.zeros((5, L))
rewarddata = np.zeros((5, L))
index = 0 
for i in [0, 1, 2, 3, 4]:
	with open('./runs/theoretical_pg_' + key_name + str(i)  + '.pkl', 'rb') as f:
		data = pkl.load(f)
	temp_time_d = np.asarray(data[1])[0:L]
	print (temp_time_d[-1], min(temp_time_d))
	temp_reward_d = np.asarray(data[0])[0:L]
	timedata[index, :] = temp_time_d
	rewarddata[index, :] = temp_reward_d
	index += 1

mean_timedata = np.mean(timedata, axis=0)
print (np.min(timedata))
std_timedata = np.std(timedata, axis=0)/np.sqrt(5)
print (np.shape(std_timedata))

ax[0].plot(mean_timedata, label='Theoretical PG', color='#a50f15', linewidth=2, linestyle='dotted')
ax[0].fill_between(np.arange(L), mean_timedata-std_timedata, mean_timedata + std_timedata, color='#a50f15', alpha=0.1)

mean_rewardsdata = np.mean(rewarddata, axis=0)

std_rewarddata = np.std(rewarddata, axis=0)/np.sqrt(5)

ax[1].plot(mean_rewardsdata, color='#a50f15', linewidth=2, linestyle='dotted')
ax[1].fill_between(np.arange(L), mean_rewardsdata-std_rewarddata, mean_rewardsdata + std_rewarddata, color='#a50f15', alpha=0.1)

timedata = np.zeros((5, L))
rewarddata = np.zeros((5, L))
index = 0 
for i in [0, 1, 2, 3, 4]:
	with open('./runs/ldg_' + str(i) + '_' + str(gamma) + '_' + str(size) + '_adam_v2.pkl', 'rb') as f:
		data = pkl.load(f)
	temp_time_d = np.asarray(data[1])[0:L]
	print (temp_time_d[-1], min(temp_time_d))
	temp_reward_d = np.asarray(data[0])[0:L]
	timedata[index, :] = temp_time_d
	rewarddata[index, :] = temp_reward_d
	index += 1


mean_timedata = np.mean(timedata, axis=0)
print (np.min(timedata))
std_timedata = np.std(timedata, axis=0)/np.sqrt(5)
print (np.shape(std_timedata))

ax[0].plot(mean_timedata, label='LDPG (ours)', color='#006d2c', linewidth=2, linestyle='solid')
ax[0].fill_between(np.arange(L), mean_timedata-std_timedata, mean_timedata + std_timedata, color='#006d2c', alpha=0.1)

mean_rewardsdata = np.mean(rewarddata, axis=0)

std_rewarddata = np.std(rewarddata, axis=0)/np.sqrt(5)

ax[1].plot(mean_rewardsdata, color='#006d2c', linewidth=2, linestyle='solid')
ax[1].fill_between(np.arange(L), mean_rewardsdata-std_rewarddata, mean_rewardsdata + std_rewarddata, color='#006d2c', alpha=0.1)







timedata = np.zeros((5, L))
rewarddata = np.zeros((5, L))
index = 0 
for i in [0, 1, 2, 3, 4]:
	with open('./runs/ldg_pg_' + key_name + str(i)  + '.pkl', 'rb') as f:
		data = pkl.load(f)
	temp_time_d = np.asarray(data[1])[0:L]
	print (temp_time_d[-1], min(temp_time_d))
	temp_reward_d = np.asarray(data[0])[0:L]
	timedata[index, :] = temp_time_d
	rewarddata[index, :] = temp_reward_d
	index += 1

mean_timedata = np.mean(timedata, axis=0)
print (np.min(timedata))
std_timedata = np.std(timedata, axis=0)/np.sqrt(5)
print (np.shape(std_timedata))

ax[0].plot(mean_timedata, label='Theoretical LDG', color='#00441b', linewidth=2, linestyle='dashdot')
ax[0].fill_between(np.arange(L), mean_timedata-std_timedata, mean_timedata + std_timedata, color='#00441b', alpha=0.1)

mean_rewardsdata = np.mean(rewarddata, axis=0)

std_rewarddata = np.std(rewarddata, axis=0)/np.sqrt(5)

ax[1].plot(mean_rewardsdata, color='#00441b', linewidth=2, linestyle='dashdot')
ax[1].fill_between(np.arange(L), mean_rewardsdata-std_rewarddata, mean_rewardsdata + std_rewarddata, color='#00441b', alpha=0.1)















ax[0].set_ylabel('Average Timesteps',fontsize=15)
ax[1].set_ylabel('Average Rewards', fontsize=15)
ax[0].set_ylim(-5, 40)
ax[1].set_ylim(-400, 0)

ax[0].set_xlim(-1, L + 1 )
ax[1].set_xlim(-1, L + 1)



ax[0].set_xlabel('Episodes', fontsize=15)
ax[1].set_xlabel('Epsiodes', fontsize=15)

ax[0].grid(visible=True)
ax[1].grid(visible=True)

fig.legend(loc=(0.3, 0.9), ncol=4)
plt.subplots_adjust(wspace=0.3)
#fig.subplots_adjust(right=0.8)
#fig.tight_layout() 

plt.savefig('performance_' + str(size) + '_times_' + str(size) + '.pdf') 











