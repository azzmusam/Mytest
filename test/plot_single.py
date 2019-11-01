import matplotlib.pyplot as plt
import csv
import numpy
import math
from pandas import read_csv

'''with open('time_step.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(row[0])'''

plt.close()

#data = numpy.memmap('time_step.csv', mode='r')
#y = [i for i in range(len(data))]

#fig, ax = plt.subplots()
#ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
#1001686
num = 500
data = read_csv('time_step.csv')
data = data.rename(columns={'0.0':'reward'})
data['smooth_path'] = data['reward'].rolling(num, min_periods=1).mean()
data['path_deviation'] = data['reward'].rolling(num, min_periods=1).std()
data = data.fillna(0)
print(len(data['reward']))

plt.plot(data['smooth_path'], linewidth=0.1, linestyle='-', marker=',', label="Reward per Time Step")
plt.xlabel('Time Step')
plt.ylabel('Total Reward')
plt.legend(loc='lower right')

plt.grid(color='k', alpha=.1)
#plt.yticks([0, -1500, -2000, -2500, -3000, -3500, -4000, -4500])
plt.fill_between(data['path_deviation'].index, (data['smooth_path'] -2*data['path_deviation']/math.sqrt(num)),(data['smooth_path'] +2*data['path_deviation']/math.sqrt(num)), color='b', alpha=.1)

plt.show()



#data.plot(kind='line', linestyle='-', linewidth=0.01, marker=',')
# plt.show()
#plt.switch_backend('TkAgg')
#plt.get_backend()
#plt.plot(data, y, label='Reward/time_step', linewidth = 0.001, linestyle='-', marker=",")
#mng = plt.get_current_fig_manager()
#mng.resize(*mng.window.maxsize())

