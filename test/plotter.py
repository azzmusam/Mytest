import pandas as pd 
import csv
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import os
import glob
import pdb

def Reward_errplotter(data1, data2, data3, name):
    path = os.getcwd()
    file1 = os.path.join(path, 'test_result', str(data1))
    file2 = os.path.join(path, 'results', str(data2))
    file3 = os.path.join(path, 'results', str(data3))
    #file4 = os.path.join(path, 'results', str(data4))

    data1 = pd.read_csv(file1, header=None, nrows=100000)
    data2 = pd.read_csv(file2, header=None, nrows=1000000)
    data3 = pd.read_csv(file3, header=None, nrows=1000000)
    #data4 = pd.read_csv(file4, header=None,  nrows=1000000)

    data1['reward'] = data1.rolling(10000, min_periods=1).mean()
    data2['reward'] = data2.rolling(10000, min_periods=1).mean()
    data3['reward'] = data3.rolling(10000, min_periods=1).mean()
    #data4['reward'] = data4.ewm(span=10000, min_periods=1).mean()
    #num = data.shape[0]

    data1['standard_deviation'] = data1['reward'].rolling(10000, min_periods=1).std()
    data2['standard_deviation'] = data2['reward'].rolling(10000, min_periods=1).std()
    data3['standard_deviation'] = data3['reward'].rolling(10000, min_periods=1).std()
    #data4['standard_deviation'] = data4['reward'].ewm(span=10000, min_periods=1).std()

    plt.plot(data1['reward'],  linewidth=0.7, linestyle='solid', color ='k', label="0.05")
    plt.plot(data2['reward'],  linewidth=0.7, linestyle='solid', color ='r', label="0.2")
    plt.plot(data3['reward'],  linewidth=0.7, linestyle='solid', color ='g', label="0.4")
    #plt.plot(data4['reward'],  linewidth=0.7, linestyle='solid', color ='b', label="0.4")

    plt.fill_between(data1['standard_deviation'].index,  (data1['reward'] - 2*data1['standard_deviation']),
                    (data1['reward'] + 2*data1['standard_deviation']), color='k', alpha=.05)
    plt.fill_between(data2['standard_deviation'].index,  (data2['reward'] - 2*data2['standard_deviation']),
                    (data2['reward'] + 2*data2['standard_deviation']), color='r', alpha=.05)
    plt.fill_between(data3['standard_deviation'].index,  (data3['reward'] - 2*data3['standard_deviation']),
                    (data3['reward'] + 2*data3['standard_deviation']), color='g', alpha=.05)
    #plt.fill_between(data4['standard_deviation'].index,  (data4['reward'] - 2*data4['standard_deviation']),
     #               (data4['reward'] + 2*data4['standard_deviation']), color='b', alpha=.05)

    plt.xticks(np.linspace(0, 1000000, 5), [0, 0.25, 0.5, 0.75, 1.0])
    plt.ylim(top=0)
    plt.xlabel('Time Steps')
    plt.ylabel('Reward')
    plt.legend()
    print(name)
    plt.savefig(name, dpi=300)


def data_concat(data, typ):
    path = os.getcwd()
    dat = pd.read_csv(os.path.join(path, data), header=None)
    if typ == 'result':
        files = [files for files in glob.glob(os.path.join(path, os.path.dirname(data), 'result*'))]
    else:
        files = [files for files in glob.glob(os.path.join(path, os.path.dirname(data), 'traveltime*'))]
    '''
    for f in files:
       f_data = pd.read_csv(f, header=None)
       new_data = pd.concat([dat, f_data], ignore_index=True)
       new_data.to_csv(f)
    '''
    f_data = pd.read_csv(files[0], header=None)
    new_data = pd.concat([dat, f_data], ignore_index=True)
    new_data.to_csv(files[0], header=None, )


def sort_files(data, typ):
    path = os.getcwd()
    if typ=='result':
        reward_files = [files for files in glob.glob(os.path.join(path, data, 'score*'))]
    else:
        reward_files = [files for files in glob.glob(os.path.join(path, data, 'testtraveltime*'))]
    if len(reward_files)==0:
        reward_files = [files for files in glob.glob(os.path.join(path, data, 'testscore_*'))]
    sorted_reward_file = []
    for i in range(len(reward_files)):
        try:
            if typ=='result':
                #csv = reward_files[i].split('result')[-1]
                csv = reward_files[i].split('score_')[-1]
            else:
                csv = reward_files[i].split('traveltime_')[-1]
        except:
            csv = reward_files[i].split('score_')[-1]
        sorted_reward_file.append(int(csv.split('.')[0]))
    sorted_reward_file.sort()
    sorted_reward_files = [i for i in sorted_reward_file if i<250000]
    dirname = os.path.dirname(reward_files[0])
    #result =  os.path.basename(reward_files[0]).split('.')[0]
    for i in range(len(sorted_reward_files)):
        if typ =='result':
            if data =='test_result/single_intersection/0.05':
                sorted_reward_files[i] = dirname + '/' + 'testscore_' + str(sorted_reward_files[i]) + '.csv'
            else:
                sorted_reward_files[i] = dirname + '/' + 'score_' + str(sorted_reward_files[i]) + '.csv'
            #sorted_reward_files[i] = dirname + '/' + 'result' + str(sorted_reward_files[i]) + '.csv'
        else:
             if data =='test_result/single_intersection/0.05':
                 sorted_reward_files[i] = dirname + '/' + 'testtraveltime_' + str(sorted_reward_files[i]) + '.csv'
             else:
                 sorted_reward_files[i] = dirname + '/' + 'traveltime_' + str(sorted_reward_files[i]) + '.csv'
             #sorted_reward_files[i] = dirname + '/' + 'traveltime' + str(sorted_reward_files[i]) + '.csv'
    return sorted_reward_files, sorted_reward_file


def average_reward_per_evaluation(sorted_reward_list, data_type):
    average_reward_list = []
    path = os.getcwd()
    reward_std = []
    if data_type == 'results/eight_intersection/0.4/maxplus/trial':
        initial = pd.read_csv('/home/azlaans/aienvs/test/results/eight_intersection/0.4/brute/traveltime0.csv', header=None)
        average_reward_list.append(initial.mean())
        reward_std.append(initial.std())
    if data_type == 'results/six_intersection/0.4/maxplus/trial':
        removed_data = pd.read_csv(sorted_reward_list[0], header=None)
        sorted_reward_list.pop(0)
        initial_data = pd.read_csv(os.path.join(path, 'results/six_intersection/0.4/maxplus/trial/10000/traveltime0.csv'), header=None)
        average_reward_list.append(initial_data.mean()[0])
        reward_std.append(initial_data.std())

    for i in range(len(sorted_reward_list)):
        data = pd.read_csv(sorted_reward_list[i], header=None)
        if i ==0:
            if data_type == 'results/six_intersection/0.05/maxplus/trial':
                start_row = removed_data.shape[0]
                average_reward_list.append(data.iloc[start_row:].mean()[0])
                reward_std.append(data.iloc[start_row:].std()[0])
                previous_data_shape = data.shape[0]
            else:
                start_row = data.shape[0]
                average_reward_list.append(data.mean()[0])
                reward_std.append(data.std()[0])
        else:
            if data_type=='results/six_intersection/0.05/maxplus/trial':
                if i ==1:
                #data = pd.concat([initial_data, data.iloc[start_row:]], ignore_index=True)
                    start_row = previous_data_shape 
            average_reward_list.append(data.iloc[start_row:].mean()[0])
            reward_std.append(data.iloc[start_row:].std()[0])
            start_row = data.shape[0]
    return average_reward_list, reward_std

def single_special(sorted_files):
    avg_reward_list = []
    reward_std = []
    
    for i in range(25):
        if i ==0:
            data = pd.read_csv(sorted_files[0], header=None)
            start_row = data.shape[0]
            avg_reward_list.append(data.mean()[0])
            reward_std.append(data.std()[0])
        else:
            if i == 7 or i==8:
                avg_reward_list.append(avg_reward_list[i-1])
                reward_std.append(data.iloc[i-1])
            if i > 9:
                data = pd.read_csv(sorted_files[i-3], header=None)
                avg_reward_list.append(data.iloc[start_row:].mean()[0])
                reward_std.append(data.iloc[start_row:].std()[0])
                start_row = data.shape[0] 
            else:
                data = pd.read_csv(sorted_files[i], header=None)
                avg_reward_list.append(data.iloc[start_row:].mean()[0])
                reward_std.append(data.iloc[start_row:].std()[0])
                start_row = data.shape[0]
    avg_reward_list = [i for i in avg_reward_list if str(i)!='nan']
    return avg_reward_list, reward_std

def special(sorted_reward_files, sorted_reward_file):
    average_reward_list = []
    average_std_list = []
    initial_data = pd.read_csv(sorted_reward_files[0], header=None)
    twentydata = pd.read_csv(sorted_reward_files[2], header=None)
    twentydatashape = twentydata.shape[0]
    sorted_reward_files.pop(0)
    average_reward_list.append(initial_data.mean())
    average_std_list.append(initial_data.std())
    for i in range(len(sorted_reward_files)):
        if i ==0:
            data0 = pd.read_csv(sorted_reward_files[i], header=None)
            start_row = data0.shape[0]
            average_reward_list.append(data0.mean()[0])
            average_std_list.append(data0.std()[0])
        if sorted_reward_file[i] > 0: 
            if sorted_reward_file[i]<120000:
                lastdata = pd.read_csv(sorted_reward_files[i], header=None)
                #start_row = lastdata.shape[0]
                average_reward_list.append(lastdata.iloc[start_row:].mean()[0])
                average_std_list.append(lastdata.iloc[start_row:].std()[0])
                start_row = lastdata.shape[0]
        if sorted_reward_file[i]< 150000:
            if sorted_reward_file[i] >= 120000:
                data = pd.read_csv(sorted_reward_files[i], header=None)
                average_reward_list.append(data.mean()[0])
                average_std_list.append(data.std()[0])
        if sorted_reward_file[i] >= 150000:
            data = pd.read_csv(sorted_reward_files[i], header=None)
            average_reward_list.append(data.iloc[twentydatashape:].mean()[0])
            average_std_list.append(data.iloc[twentydatashape:].std()[0])
            twentydatashape = data.shape[0]
    return average_reward_list, average_std_list

def minimum_list_length(reward_dict):
    minimum_reward_length = 0
    i = 0
    for keys in reward_dict.keys():
        #print(len(reward_dict[keys]))
        if i ==0:
            minimum_reward_length = len(reward_dict[keys])
            i +=1
        if len(reward_dict[keys]) < minimum_reward_length:
            minimum_reward_length =  len(reward_dict[keys])
    return minimum_reward_length
        

def plotter(data, labels, axes, name, typ):
    path = os.getcwd()
    average_reward_dict = {}
    reward_std_dict = {}
    colour = ['b', 'g', 'k', 'r']
    for i in range(len(data)):
        sorted_files, files = sort_files(data[i], typ)
        if data[i] =='test_result/single/0.4':
            average_reward, std = single_special(sorted_files)
        else:
            average_reward, std = average_reward_per_evaluation(sorted_files, data[i])
        average_reward_dict[data[i]] = average_reward
        reward_std_dict[data[i]] = std
        if data[i] =='results/six_intersection/0.4/maxplus/trial/10000':
           average_reward_dict[data[i]] = [i for i in average_reward_dict[data[i]] if str(i) != 'nan']
           reward_std_dict[data[i]] = [i for i in reward_std_dict[data[i]] if str(i) != 'nan']  
    try:
        length = minimum_list_length(average_reward_dict)
    except:
       length = len(average_reward_dict[data[0]])
    print(length)
    for i in range(len(data)):
        #pdb.set_trace()
        j = np.asarray([i for i in range(length)])
        #j1 = np.asarray([i+1 for i in range(length)])
        plt.plot(average_reward_dict[data[i]][:length],  linewidth=0.7, linestyle='solid', color =str(colour[i]), label=labels[i])
        plt.fill_between(j, np.subtract(np.asarray(average_reward_dict[data[i]][:length], dtype=float), 2*np.asarray(reward_std_dict[data[i]][:length], dtype=float), dtype=float),
                    np.add(np.asarray(average_reward_dict[data[i]][:length], dtype=float), 2*np.asarray(reward_std_dict[data[i]][:length], dtype=float), dtype=float), color=colour[i], alpha=.05)
    plt.xticks(np.linspace(0, length, 5), [0, 0.25, 0.5, 0.75, 1.0])
    if typ=='result':
        plt.ylim(top=0)
        plt.xlabel('Time Steps')
        plt.ylabel('Rewards')
    else:
        plt.ylim(bottom=0)
        plt.xlabel('Time Steps')
        plt.ylabel('Average Travel Time')
    #plt.xlabel(str(axes[0]))
    #plt.ylabel(str(axes[1]))
    plt.legend(loc='best')
    print(name)
    filename  = os.path.join(path, 'graphs', str(name))
    plt.savefig(filename, dpi=300)



if __name__=='__main__':

    #data_concat(data='results/four_intersection/0.05/individual/traveltime0.csv', typ='traveltime')
    
    plotter(data = ['test_result/single_intersection/0.05',
                    'test_result/single_intersection/0.2',
                    'test_result/single/0.4'],
                    labels = ['Low', 'Medium', 'High'],
                    #labels = ['maxplus', 'brute', 'individual'], # 'maxplus'],
                    #labels = ['trial22'],
                    axes = ['Time Steps', 'Rewards'],
                    name = "fourthtrialsingle",
                    typ = 'result')
