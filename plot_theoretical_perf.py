import matplotlib.pyplot as plt 
import pickle as pkl 
import numpy as np 


gamma = 1.0
size = 10 
fig, ax = plt.subplots(1, 1, figsize=(9, 5), dpi=500)#, constrained_layout=False)

L = 200
key_name = str(size) + '_times_' + str(size) + '_'

timedata = np.zeros((5, L))
rewarddata = np.zeros((5, L))
index = 0 
for i in [0, 1, 2, 3, 4]:
	with open('./runs/theoretical_pg_' + key_name + str(i)  + '.pkl', 'rb') as f:
		data = pkl.load(f)
	print (len(data))
	temp_time_d = np.asarray(data[0])[0:L]
	print (temp_time_d[-1], min(temp_time_d))
	temp_reward_d = np.asarray(data[0])[0:L]
	timedata[index, :] = temp_time_d
	rewarddata[index, :] = temp_reward_d
	index += 1

mean_timedata = np.mean(timedata, axis=0)
print (np.min(timedata))
std_timedata = np.std(timedata, axis=0)/np.sqrt(5)
print (np.shape(std_timedata))

ax.plot(mean_timedata, label='Theoretical PG (10 x 10)', color='#a50f15', linewidth=2, linestyle='dotted')
ax.fill_between(np.arange(L), mean_timedata-std_timedata, mean_timedata + std_timedata, color='#a50f15', alpha=0.1)






timedata = np.zeros((5, L))
rewarddata = np.zeros((5, L))
index = 0 
for i in [0, 1, 2, 3, 4]:
	with open('./runs/ldg_pg_' + key_name + str(i)  + '.pkl', 'rb') as f:
		data = pkl.load(f)
	temp_time_d = np.asarray(data[0])[0:L]
	print (temp_time_d[-1], min(temp_time_d))
	temp_reward_d = np.asarray(data[0])[0:L]
	timedata[index, :] = temp_time_d
	rewarddata[index, :] = temp_reward_d
	index += 1

mean_timedata = np.mean(timedata, axis=0)
print (np.min(timedata))
std_timedata = np.std(timedata, axis=0)/np.sqrt(5)
print (np.shape(std_timedata))

ax.plot(mean_timedata, label='Theoretical LDG (10 x 10)', color='#00441b', linewidth=2, linestyle=(0, (5, 1)))
ax.fill_between(np.arange(L), mean_timedata-std_timedata, mean_timedata + std_timedata, color='#00441b', alpha=0.1)

key_name = str(5) + '_times_' + str(5) + '_'


timedata = np.zeros((5, L))
rewarddata = np.zeros((5, L))
index = 0 
for i in [0, 1, 2, 3, 4]:
	with open('./runs/theoretical_pg_' + key_name + str(i)  + '.pkl', 'rb') as f:
		data = pkl.load(f)
	print (len(data))
	temp_time_d = np.asarray(data[0])[0:L]
	print (temp_time_d[-1], min(temp_time_d))
	temp_reward_d = np.asarray(data[0])[0:L]
	timedata[index, :] = temp_time_d
	rewarddata[index, :] = temp_reward_d
	index += 1

mean_timedata = np.mean(timedata, axis=0)
print (np.min(timedata))
std_timedata = np.std(timedata, axis=0)/np.sqrt(5)
print (np.shape(std_timedata))

ax.plot(mean_timedata, label='Theoretical PG (5 x 5)', color='#a50f15', linewidth=2, linestyle=(5, (10, 3)))
ax.fill_between(np.arange(L), mean_timedata-std_timedata, mean_timedata + std_timedata, color='#a50f15', alpha=0.1)






timedata = np.zeros((5, L))
rewarddata = np.zeros((5, L))
index = 0 
for i in [0, 1, 2, 3, 4]:
	with open('./runs/ldg_pg_' + key_name + str(i)  + '.pkl', 'rb') as f:
		data = pkl.load(f)
	temp_time_d = np.asarray(data[0])[0:L]
	print (temp_time_d[-1], min(temp_time_d))
	temp_reward_d = np.asarray(data[0])[0:L]
	timedata[index, :] = temp_time_d
	rewarddata[index, :] = temp_reward_d
	index += 1

mean_timedata = np.mean(timedata, axis=0)
print (np.min(timedata))
std_timedata = np.std(timedata, axis=0)/np.sqrt(5)
print (np.shape(std_timedata))

ax.plot(mean_timedata, label='Theoretical LDG (5 x 5)', color='#00441b', linewidth=2, linestyle=(0, (3, 1, 1, 1)))
ax.fill_between(np.arange(L), mean_timedata-std_timedata, mean_timedata + std_timedata, color='#00441b', alpha=0.1)










ax.set_ylabel('Average Rewards',fontsize=15)
#ax[1].set_ylabel('Average Rewards', fontsize=15)
#ax.set_ylim(-1.5, 0)
#ax[1].set_ylim(-400, 0)

#ax.set_xlim(0.5, L + 1 )
#ax[1].set_xlim(-1, L + 1)



ax.set_xlabel('Episodes', fontsize=15)
#ax[1].set_xlabel('Epsiodes', fontsize=15)

ax.grid(visible=True)
#ax[1].grid(visible=True)

fig.legend(loc=(0.613, 0.15), ncol=1)
#plt.subplots_adjust(right=0.7)

#plt.subplots_adjust(wspace=0.3)
#fig.subplots_adjust(right=0.8)
#fig.tight_layout() 

plt.savefig('theory_performance.pdf') 











