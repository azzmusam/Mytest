import os 
import numpy as np
import pandas as pd
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

def traveltimeplotter(name: str, typ: str):
    path = os.getcwd()
    files = os.path.join(path, 'test_result', str(typ), str(name))
    data = pd.read_csv(files, header=None)
    i = data[0].values.tolist()
    print(len(i))
    j = [i for i in range(len(i))]
    fig, ax = plt.subplots()
    ax.grid(color='grey', linewidth=0.001, alpha=0.5)
    ax.set_title('Average Travel Time')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Average Travel Time(sec)')
    plt.plot(j, i, '-k', label='Average Travel Time')
    plt.legend(loc='best')
    plt.grid()
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fignam = str(typ) + 'Travel Time'
    plt.savefig(fignam, dpi=200)

def normalplotter(typ: str, title: str, xlabel: str, ylabel: str, name: str):
    path = os.getcwd()
    files = os.path.join(path, 'test_result', str(typ), str(name))
    data = pd.read_csv(files, header=None, nrows= 1000000)
    #data = data.rolling(4000, min_periods=1).mean()
    i = data[0].values.tolist()
    j = [i for i in range(len(i))]
    fig, ax = plt.subplots()
    ax.grid(color='grey', linewidth=0.25, alpha=0.5)
    ax.set_title(str(title))
    ax.set_xlabel(str(xlabel))
    ax.set_ylabel(str(ylabel))
    plt.plot(j, i, '-k', linewidth=0.5, label=str(ylabel))
    plt.legend(loc='best')
    fignam = str(typ)+str(title)
    plt.savefig(fignam, dpi=300)


if __name__=='__main__':
    
    path = os.getcwd()
    files = os.path.join(path, 'test_result', 'single', 'traveltime_220000.csv')
    data = pd.read_csv(files, header=None)
    tt = bestmodel(data)
    for i in range(len(tt)):
        print(tt[i])
    idx = np.argmin(tt)
    print(tt)
    print(tt[idx])
    print(idx)
    
    #normalplotter(typ='single', name='score_220000.csv', title='Reward per Time Step', xlabel='Time Steps', ylabel='Rewards')
    traveltimeplotter(typ='single', name='traveltime_220000.csv')
    #normalplotter(typ='vertical', name='waitingtime280000.csv', title='Waiting Time per Time Step', xlabel='Time Steps', ylabel='Waiting Time')
    
    #normalplotter(typ='horizontal', name='delay200000.csv', title='Delay per Time Step', xlabel='Time Steps', ylabel='Delay')
    #traveltimeplotter(typ='horizontal', name='traveltime200000.csv')
    #normalplotter(typ='horizontal', name='waitingtime200000.csv', title='Waiting Time  per Time Step', xlabel='Time Steps', ylabel='Waiting Time')


