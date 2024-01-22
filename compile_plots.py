import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
ldg = {}
ldg['timesteps'] = np.zeros((10, 20))
ldg['rewards'] = np.zeros((10, 20))
pg = {}
pg['timesteps'] = np.zeros((10, 20))
pg['rewards'] = np.zeros((10, 20))

for i in range(10):
	print (i)
	with open('./experiment_data/gridworld/ldg_discrete_' + str(i) + '.pkl', 'rb') as f:
		data = pickle.load(f)
	ldg['timesteps'][i, :] = data['timestep_history']
	ldg['rewards'][i, :] = data['perf_history']
	with open('./experiment_data/gridworld/pg_' + str(i) + '.pkl', 'rb') as f:
		data = pickle.load(f)

	pg['timesteps'][i, :] = data['timestep_history']
	pg['rewards'][i, :] = data['perf_history']
colors = ['#2ca25f', '#de2d26']
plt.figure(figsize=(12, 5), dpi=80)
x_axis = 10*np.range(20)
ldg_mean = np.mean(ldg['rewards'], axis=0)
ldg_std = np.std(ldg['rewards'], axis=0)
plt.plot(x_axis, ldg_mean, color=colors[0])
plt.fill_between(x_axis, ldg_mean-ldg_std, ldg_mean + ldg_std)
plt.show()