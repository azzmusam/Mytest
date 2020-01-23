import os
import math
import numpy as np
import pandas as pd
import pdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def smooth(scalars, weight):
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def bestmodel(data):
    avg_travel_time = []
    for i in range(0, len(data), 8):
        avg = np.mean(data[i:i+8])
        avg_travel_time.append(avg)
    print(avg_travel_time)
    print('Least Travel score', avg_travel_time[np.argmin(avg_travel_time)])
    print('Lenth of average travel time list', len(avg_travel_time))
    print('Index of the minimum travel time', np.argmin(avg_travel_time))
    return avg_travel_time

def avgttplotter(name: str, typ: str):
    path = os.getcwd()
    files = os.path.join(path, 'test_result', str(typ), str(name))
    #data = pd.read_csv(files, header=None)
    #i = data[0].values.tolist()
    print(len(i))
    avg_travel_time = bestmodel(i)
    print(avg_travel_time)
    print('Lenth of average travel time list', len(avg_travel_time))
    print('Index of the minimum travel time', np.argmin(avg_travel_time))
    j = [i for i in range(len(avg_travel_time))]
    fig, ax = plt.subplots()
    ax.grid(color='grey', linewidth=0.25, alpha=0.5)
    ax.set_title('Average Travel Time per Saved Model')
    ax.set_xlabel('Saved checkpoints')
    ax.set_ylabel('Average Travel Time(sec)')
    plt.plot(j, avg_travel_time, '-k', label='Average Travel Time')
    plt.legend(loc='best')
    plt.xticks([])
    plt.savefig(str(ylabel), dpi=300)

def traveltimeplotter(data1, data2, data3, name):#data3, data4, name):
    path = os.getcwd()
    try:
        file1 = os.path.join(path, 'test_result', str(data1[0]), str(data1[1]))
        file2 = os.path.join(path, 'test_result', str(data2[0]), str(data2[1]))
        file3 = os.path.join(path, 'test_result', str(data3[0]), str(data3[1]))
        #file4 = os.path.join(path, 'test_result', str(data4[0]), str(data4[1]))
        data1 = pd.read_csv(file1, header=None, nrows=157)
        data1['tt'] = data1.rolling(3, min_periods=1).mean()
        data1['std'] = data1['tt'].rolling(3, min_periods=1).std()

        data2 = pd.read_csv(file2, header=None,  skiprows=3, nrows=157)
        data2['tt'] = data2.rolling(3, min_periods=1).mean()
        data2['std'] = data2['tt'].rolling(3, min_periods=1).std()

        data3 = pd.read_csv(file3, header=None,  skiprows=11)#, nrows=160)
        data3['tt'] = data3.rolling(3, min_periods=1).mean()
        data3['std'] = data3['tt'].rolling(3, min_periods=1).std()

        #data4 = pd.read_csv(file4, header=None)
        i1 = data1[0].values.tolist()
        i2 = data2[0].values.tolist()
        i3 = data3[0].values.tolist()
        #i4 = data4[0].values.tolist()
        print(len(i1), len(i2))#, len(i3), len(i4))

        j = [i for i in range(len(i2))]
        fig, ax = plt.subplots()
        ax.grid(color='grey', linewidth=0.5, alpha=1)
        ax.set_title('Average Travel Time')
        ax.set_xlabel('Simulations')
        ax.set_ylabel('Average Travel Time(sec)')
        
        plt.plot(data1['tt'],  linewidth=0.7, linestyle='solid', color ='k', label="0.05")
        plt.fill_between(data1['std'].index,  (data1['tt'] - 2*data1['std']),
                    (data1['tt'] + 2*data1['std']), color='k', alpha=.2)

        plt.plot(data2['tt'],  linewidth=0.7, linestyle='solid', color ='r', label="0.2")
        plt.fill_between(data2['std'].index,  (data2['tt'] - 2*data2['std']),
                    (data2['tt'] + 2*data2['std']), color='r', alpha=.2)

        plt.plot(data3['tt'],  linewidth=0.7, linestyle='solid', color ='g', label="0.4")
        plt.fill_between(data3['std'].index,  (data3['tt'] - 2*data3['std']),
                    (data3['tt'] + 2*data3['std']), color='g', alpha=.2)

        '''plt.plot(j, i1, '-k', linewidth=0.7, label='0.05')
        plt.plot(j, i2, '-r', linewidth=0.7, label='0.2')
        plt.plot(j, i3, '-g', linewidth=0.7, label='0.4')'''
        #plt.plot(j, i4, '-b', linewidth=0.7, label='Individual')
        plt.legend(loc='upper right')
    #plt.grid()
        #plt.xticks(np.arange(len(i1)), [0, 0.25, 0.50, 0.75, 1.0])
        #plt.xlim(0,162)
    except:
        pass
    fignam = str(name)
    plt.savefig(fignam, dpi=300)

def plotloss(data):
    # j = [i for i in range(len(avg_travel_time))]
    fig, ax = plt.subplots()
    ax.grid(color='grey', linewidth=0.25, alpha=0.5)
    ax.set_title('Loss')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Loss')
    i  = data.values.tolist()
    print(len(i))
    data['loss'] = data.rolling(200, min_periods=1).mean()
    data['standard_deviation'] = data['loss'].rolling(200, min_periods=1).std()
    plt.plot(data.index, data['loss'], '-k', label='Loss', linewidth= 0.35)
    plt.fill_between(data['standard_deviation'].index,  (data['loss'] - 2*data['standard_deviation']),
                    (data['loss'] + 2*data['standard_deviation']), color='k', alpha=.1)
    plt.legend(loc='best')
    plt.xticks(np.linspace(start=0, stop=len(i), num=5, endpoint=True), [0, 0.25, 0.5, 0.75, 1.0])
    plt.savefig('Loss_single_high', dpi=300)

def normalplotter(typ: str, title: str, xlabel: str, ylabel: str, name: str):
    path = os.getcwd()
    files = os.path.join(path, 'test_result', str(typ), str(name))
    data = pd.read_csv(files, header=None, nrows= 1000000)
    #data = data.rolling(1000, min_periods=1).mean()
    i = data[0].values.tolist()
    j = [i for i in range(len(i))]
    fig, ax = plt.subplots()
    ax.grid(color='grey', linewidth=0.25, alpha=0.5)
    #ax.set_title(str(title))
    ax.set_xlabel(str(xlabel))
    ax.set_ylabel(str(ylabel))
    plt.plot(j, i, '-k', linewidth=0.5, label=str(ylabel))
    plt.legend(loc='best')
    fignam = 'single_low_'+str(title)
    plt.savefig(fignam, dpi=300)

def errplotter(data1, data2):
    path = os.getcwd()
    file1 = os.path.join(path, 'results', data1)
    file2 = os.path.join(path, 'test_result', data2)
    data1 = pd.read_csv(file1, header=None, nrows=1000000)
    data2 = pd.read_csv(file2, header=None, nrows=1000000)
    data1['reward'] = data1.rolling(5000, min_periods=1).mean()
    data2['reward'] = data2.rolling(5000, min_periods=1).mean()
    #num = data.shape[0]
    data1['standard_deviation'] = data1['reward'].rolling(5000, min_periods=1).std()
    data2['standard_deviation'] = data2['reward'].rolling(5000, min_periods=1).std()
    plt.plot(data1['reward'],  linewidth=0.7, linestyle='solid', color ='k', label="Reward per Time Step")
    plt.plot(data2['reward'],  linewidth=0.7, linestyle='solid', color ='g', label="Reward per Time Step")
    plt.fill_between(data1['standard_deviation'].index,  (data1['reward'] - 2*data1['standard_deviation']),
                    (data1['reward'] + 2*data1['standard_deviation']), color='k', alpha=.1)
    plt.fill_between(data2['standard_deviation'].index,  (data2['reward'] - 2*data2['standard_deviation']),
                    (data2['reward'] + 2*data2['standard_deviation']), color='g', alpha=.1)
    plt.savefig('old_vs_new_trial3single_high_plot', dpi=300)


def ttplot(data1, data2):
    path = os.getcwd()
    file1 = os.path.join(path, 'results', data1)
    file2 = os.path.join(path, 'results', data2)
    data1 = pd.read_csv(file1, header=None)
    data2 = pd.read_csv(file2, header=None, nrows=data1.shape[0])
    data1['rolling'] = data1.rolling(8, min_periods=1).mean()
    data1['standard_deviation'] = data1['rolling'].rolling(8, min_periods=1).std()
    plt.plot(data1['rolling'],  linewidth=0.7, linestyle='solid', color ='r')
    plt.fill_between(data1['standard_deviation'].index,  (data1['rolling'] - 2*data1['standard_deviation']),
                    (data1['rolling'] + 2*data1['standard_deviation']), color='r', alpha=.05)

    plt.plot(data2,  linewidth=0.7, linestyle='solid', color ='g')
    plt.fill_between(data2['standard_deviation'].index,  (data2['rolling'] - 2*data2['standard_deviation']),
                    (data2['rolling'] + 2*data2['standard_deviation']), color='g', alpha=.05)
    plt.savefig('four_new_high_tt', dpi=300)


def debugplot(data1, name):
    path = os.getcwd()
    file1 = os.path.join(path, 'results', 'qvalues',data1)
    #file2 = os.path.join(path, 'results', 'qvalues', data2)
    data1 = pd.read_csv(file1, header=None)
    #data1['BC'] = data1
    #data2 = pd.read_csv(file2, header=None, nrows=data1.shape[0])
    #data2['MP'] = data2
    #data1['diff'] = data1['BC']- data2['MP']
    plt.plot(data1, linewidth=0.7, linestyle='solid', color ='r') 
    #plt.plot(data2, linewidth=0.7, linestyle='solid', color ='g', label='MP')
    #plt.plot(data1['diff'], linewidth=0.7, linestyle='solid', color ='r', label='Difference')
    plt.legend()
    plt.savefig(str(name), dpi=300)

def oldtraveltimeplotter(name: str, typ: str):
    path = os.getcwd()
    files = os.path.join(path, 'results', str(typ), str(name))
    data = pd.read_csv(files, header=None)
    i = data.rolling(20, min_perios=1).mean
    print(len(i))
    j = [i for i in range(len(i))]
    fig, ax = plt.subplots()
    ax.grid(color='grey', linewidth=0.5, alpha=1)
    ax.set_title('Average Travel Time')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Average Travel Time(sec)')
    plt.plot(j, i, '-k', linewidth=0.7, label='Average Travel Time')
    plt.legend(loc='best')
    #plt.grid()
    #plt.xticks([  0,  38,  76, 114, 152], [0, 0.25, 0.50, 0.75, 1.0])
    #plt.xlim(0,155)
    fignam = '2ndtrialTT'
    plt.savefig(fignam, dpi=300)

def data_concat(data1, data2):
    path = os.getcwd()
    file1 = os.path.join(path, 'results', data1)
    file2 = os.path.join(path, 'results', data2) 
    dat1 = pd.read_csv(file1, header=None)
    print(dat1.head())
    print(dat1.tail())
    dat1_shape = dat1.shape	
    dat2 = pd.read_csv(file2, header=None)
    dat2 = dat2.reset_index(drop=True)
    print(dat2.head())
    print(dat2.tail())
    dat2_shape = dat2.shape
    final_data = pd.concat([dat1, dat2], axis=0, ignore_index=True)
    final_data.to_csv(os.path.join(path, 'results', 'four_intersection/0.4/brute', 'final_score.csv'), index=False)

def EWM_Reward_errplotter(data1, data2, data3, name):
    path = os.getcwd()
    file1 = os.path.join(path, 'test_result', str(data1))
    file2 = os.path.join(path, 'test_result', str(data2))
    file3 = os.path.join(path, 'test_result', str(data3))
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

def Reward_errplotter(data2, data3, data4, name):
    path = os.getcwd()
    #file1 = os.path.join(path, 'test_result', str(data1))
    file2 = os.path.join(path, 'test_result', str(data2))
    file3 = os.path.join(path, 'test_result', str(data3))
    file4 = os.path.join(path, 'results', str(data4))   

    #data1 = pd.read_csv(file1, header=None, nrows=32615)
    data2 = pd.read_csv(file2, header=None,  nrows=1000000)
    data3 = pd.read_csv(file3, header=None,  nrows=1000000)
    data4 = pd.read_csv(file4, header=None,  nrows=1000000)

    #data1['reward'] = data1.rolling(1500, min_periods=1).mean()
    data2['reward'] = data2.rolling(5000, min_periods=1).mean()
    data3['reward'] = data3.rolling(5000, min_periods=1).mean()
    data4['reward'] = data4.rolling(5000, min_periods=1).mean()
    #num = data.shape[0]

    #data1['standard_deviation'] = data1['reward'].rolling(1500, min_periods=1).std()
    data2['standard_deviation'] = data2['reward'].rolling(5000, min_periods=1).std()
    data3['standard_deviation'] = data3['reward'].rolling(5000, min_periods=1).std()
    data4['standard_deviation'] = data4['reward'].rolling(5000, min_periods=1).std()

    #plt.plot(data1['reward'],  linewidth=0.7, linestyle='solid', color ='k', label="Single")
    plt.plot(data2['reward'],  linewidth=0.7, linestyle='solid', color ='r', label="0.05")
    plt.plot(data3['reward'],  linewidth=0.7, linestyle='solid', color ='g', label="0.2")
    plt.plot(data4['reward'],  linewidth=0.7, linestyle='solid', color ='b', label="0.4")

    #plt.fill_between(data1['standard_deviation'].index,  (data1['reward'] - 2*data1['standard_deviation']),
                    #(data1['reward'] + 2*data1['standard_deviation']), color='k', alpha=.2)
    plt.fill_between(data2['standard_deviation'].index,  (data2['reward'] - 2*data2['standard_deviation']),
                    (data2['reward'] + 2*data2['standard_deviation']), color='r', alpha=.05)
    plt.fill_between(data3['standard_deviation'].index,  (data3['reward'] - 2*data3['standard_deviation']),
                    (data3['reward'] + 2*data3['standard_deviation']), color='g', alpha=.05)
    plt.fill_between(data4['standard_deviation'].index,  (data4['reward'] - 2*data4['standard_deviation']),
                    (data4['reward'] + 2*data4['standard_deviation']), color='b', alpha=.05)

    plt.xticks(np.linspace(0, 1000000, 5), [0, 0.25, 0.5, 0.75, 1.0])
    plt.ylim(top=0)
    plt.xlabel('Time Steps')
    plt.ylabel('Reward')
    plt.legend()
    print(name)
    plt.savefig(name, dpi=300)

def averagereward(data):
    path = os.getcwd()
    filename = os.path.join(path, 'test_result', str(data))
    data = pd.read_csv(filename, header=None)
    print(data.head())
    data_avg = data.mean()
    print(data_avg)

if __name__=="__main__":

    #ttplot('four_intersection/0.4/',
     #      'four_intersection/0.4/')

    '''EWM_Reward_errplotter(data1='two_intersection/0.05/test/score_450000.csv', 
                      data2='two_intersection/0.2/test/score_400000.csv', 
                      data3='two_intersection/0.4/test/score_200000.csv',
                      name='twoagent_reward')'''
    debugplot('diff_qval10.csv', 'difference_1010')

    #data_concat('four_intersection/0.4/brute/result00.csv',
     #           'four_intersection/0.4/brute/result110000.csv') 
 
    #path = os.getcwd()
    #files = os.path.join(path, 'test_result', 'three_intersection/test/0.05', 'traveltime_1000000.csv')
    #files = os.path.join(path, 'test_result', 'traveltime730000.csv')
    #data = pd.read_csv(files, header=None)
    #bestmodel(data)

    #plotloss(data)
    '''for i in range(len(tt)):
        print(tt[i])
    idx = np.argmin(tt)
    print(tt)
    print(tt[idx])
    print(idx)'''
 
    #data_concat('single/0.4/score_0.csv', 'single/0.4/score_210000.csv')
    #ttplot('single/0.4/traveltime_190000.csv')
    #oldtraveltimeplotter(typ='single/0.4', name='traveltime_250000.csv')
    #averagereward('six_intersection/maxplus/0.4/two_factor/result_110000.csv')
    #averagereward('six_intersection/brutecoord/0.4/two_factor/result_110000.csv')
    #averagereward('three_intersection/brutecoord/two_factor/0.4/result_110000.csv')
    #averagereward('three_intersection/maxplus/0.4/result_110000.csv')
    #averagereward('three_intersection/brutecoord/0.4/result_110000.csv')
    #averagereward('four_intersection/maxplus/0.2/result_20000.csv')
    #averagereward('four_intersection/brutecoord/0.2/result_20000.csv')
    #averagereward('eight_intersection/brutecoord/0.4/result_110000.csv')
    #averagereward('eight_intersection/maxplus/0.4/result_110000.csv')

   # Reward_errplotter('eight_intersection/brutecoord/0.2/result_20000.csv',
                      #'eight_intersection/maxplus/0.2/result_20000.csv',
                      #'eight_intersection/individual/0.2/result_110000.csv',
                     # 'eight_reward_medium_congestion')
 
    '''Reward_errplotter('eight_intersection/brutecoord/0.05/result_390000.csv',
                      'eight_intersection/maxplus/0.05/result_390000.csv',
                      'eight_intersection/individual/0.05/result_200000.csv',
                      'eight_reward_lowx_congestion')'''


    ''' Reward_errplotter('eight_intersection/brutecoord/0.4/result_110000.csv',
                      'eight_intersection/maxplus/0.4/result_110000.csv',
                      'eight_intersection/individual/0.4/result_10000.csv',
                      'eight_reward_high_ongestion')'''


    #oldtraveltimeplotter(typ ='single/0.4', name = 'traveltime_200000.csv' )
    #data_concat(['single/0.05', 'traveltime_80000.csv'], ['single/0.05', 'traveltime_210000.csv'])
    #normalplotter(typ='single/0.05', name='final_score_0.05.csv', title='Reward per Time Step', xlabel='Time Steps', ylabel='Rewards')

    '''traveltimeplotter(['single/0.05', 'final_travel_time_0.05.csv'],
                      ['single/0.2', 'traveltime_220000.csv'],
                      ['single/0.4', 'traveltime_220000.csv'],
                      'onefinal_tterrcongestion')'''

    '''traveltimeplotter(['six_intersection/brutecoord/0.05/two_factor', 'traveltime_390000.csv'],
                      ['six_intersection/brutecoord/0.05/three_factor', 'traveltime_320000.csv'],
                      'six_TP_brute_low')

    traveltimeplotter(['six_intersection/brutecoord/0.4/two_factor', 'traveltime_110000.csv'],
                      ['six_intersection/brutecoord/0.4/three_factor', 'traveltime_260000.csv'],
                      'six_TP_brute_high')'''

    '''traveltimeplotter(['four_intersection/brutecoord/0.05', 'traveltime_390000.csv'], 
                      ['four_intersection/maxplus/0.05', 'traveltime_390000.csv'], 
                      ['four_intersection/individual/0.05', 'traveltime_200000.csv'],
                      'four_intersection_lowcongestion_algo')

    traveltimeplotter(['four_intersection/brutecoord/0.2', 'traveltime_20000.csv'],
                      ['four_intersection/maxplus/0.2', 'traveltime_20000.csv'],
                      ['four_intersection/individual/0.2', 'traveltime_110000.csv'],
                      'four_intersection_medcongestion_algo')

    traveltimeplotter(['four_intersection/brutecoord/0.4', 'traveltime_110000.csv'],
                      ['four_intersection/maxplus/0.4', 'traveltime_110000.csv'],
                      ['four_intersection/individual/0.4', 'traveltime_10000.csv'],
                      'four_intersection_highcongestion_algo')

    traveltimeplotter(['six_intersection/brutecoord/0.05/two_factor', 'traveltime_390000.csv'],
                      ['six_intersection/maxplus/0.05/two_factor', 'traveltime_390000.csv'],
                      ['six_intersection/individual/0.05', 'traveltime_200000.csv'],
                       'six_intersection_twofactor_lowcongestion_algo')

    traveltimeplotter(['six_intersection/brutecoord/0.2/two_factor', 'traveltime_20000.csv'],
                      ['six_intersection/maxplus/0.2/two_factor', 'traveltime_20000.csv'],
                      ['six_intersection/individual/0.2', 'traveltime_110000.csv'],
                      'six_intersection_twofactor_mediumcongestion_algo')

    traveltimeplotter(['six_intersection/brutecoord/0.4/two_factor', 'traveltime_110000.csv'],
                      ['six_intersection/maxplus/0.4/two_factor', 'traveltime_110000.csv'],
                      ['six_intersection/individual/0.4', 'traveltime_10000.csv'],
                      'six_intersection_twofactor_highcongestion_algo')

    traveltimeplotter(['eight_intersection/brutecoord/0.05', 'traveltime_390000.csv'],
                      ['eight_intersection/maxplus/0.05', 'traveltime_390000.csv'],
                      ['eight_intersection/individual/0.05', 'traveltime_200000.csv'],
                       'eight_intersection_low_congestion_algo')

    traveltimeplotter(['eight_intersection/brutecoord/0.2', 'traveltime_20000.csv'],
                      ['eight_intersection/maxplus/0.2', 'traveltime_20000.csv'],
                      ['eight_intersection/individual/0.2', 'traveltime_110000.csv'],
                      'eight_intersection_medium_congestion_algo')

    traveltimeplotter(['eight_intersection/brutecoord/0.4', 'traveltime_110000.csv'],
                      ['eight_intersection/maxplus/0.4', 'traveltime_110000.csv'],
                      ['eight_intersection/individual/0.4', 'traveltime_10000.csv'],
                      'eight_intersection_high_congestion_algo')'''
