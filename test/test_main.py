import aienvs
from imp_DQRN import DeepQNetwork, Agent
import yaml
import logging
import pdb
from aienvs.Sumo.SumoGymAdapter import SumoGymAdapter
import numpy as np
import os
import pdb
import csv

def saver(data, name):
    name = str(name)
    filename = name+'.csv'
    outfile = open(filename, 'w')
    writer = csv.writer(outfile)
    writer.writerows(map(lambda x:[x], data))


def stack_frames(stacked_frames, frame, buffer_size):
    if stacked_frames is None:
        stacked_frames = np.zeros((buffer_size, *frame.shape))

        for idx, _ in enumerate(stacked_frames):
            stacked_frames[idx, :] = frame
    else:
        stacked_frames[0:buffer_size-1, :] = stacked_frames[1:, :]
        stacked_frames[buffer_size-1, :] = frame
    stacked_frame = stacked_frames
    stacked_state = stacked_frames.transpose(1,2,0)[None, ...]

    return stacked_frame, stacked_state


if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info("Starting test_traffic_new")

    with open("configs/new_config.yaml", 'r') as stream:
        try:
            parameters = yaml.safe_load(stream)['parameters']
        except yaml.YAMLError as exc:
            print(exc)

    env = SumoGymAdapter(parameters)
    mem_size = None
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.00025, input_dims=(84,84,1),
                  act_per_agent=2, num_agents=1, mem_size=mem_size, batch_size=32, test=True)

    maximum_time_steps = 4000
    total_number_simulation = 15
    test_time_steps_score = []
    test_time_steps_delay = []
    tesr_time_steps_waitingtime = []
    test_travel_time = []
    stack_size = 1

    path = '/home/azlaans/aienvs/test'
    
    for i in range(10000, 1000000, 10000):
        observation, average_train_times, average_train_time = env.reset(i)
        test_travel_time.append(average_train_time)
        print(test_travel_time)

        if i!=10000:
            saver(data=train_time_steps_score, name='single_test_time_steps_score_reward'+str(i))
            saver(data=train_time_steps_delay, name='single_test_time_steps_score_delay'+str(i))
            saver(data=train_time_steps_waitingtime, name='single_test_time_steps_score_waitingtime'+str(i))
            travel_filname = 'single_test_travel_time'+str(i)
            np.savetxt(travel_filname, train_travel_time)
        try:
            chkpt = os.path.join(*[path, 'tmp', 'q_eval'])
            filename = 'deepqnet.ckpt-' + str(i) + '.data-00000-of-00001'
            agent.load_models(filename)
        except:
            env.close()
        
        for j in range(total_number_simulation):
            done = False
            if j>0:    
                try:
                    observation, average_train_times, average_train_time = env.reset(j)
                    test_travel_time.append(average_train_time)
                    print(train_travel_time)
                except:
                    observation = env.reset()
                observation, stacked_state = stack_frames(stacked_frames= None, frame=observation, buffer_size=stack_size)
                agent.reset()

            while not done:
                action = agent.test(stacked_state)
                observation_, reward, done, info = env.step(action)
                agent.test_writer(reward)
                observation_, stacked_state_ = stack_frames(stacked_frames=observation, frame=observation_, buffer_size=stack_size)

                test_time_steps_score.append(reward['result'])
                test_time_steps_delay.append(reward['total_delay'])
                test_time_steps_waitingtime.append(reward['total_waiting'])

                observation = observation_
                stacked_state = stacked_state_
                if done:
                    pdb.set_trace()

    observation, average_train_times, average_train_time = env.reset(i)
    saver(data=train_time_steps_score, name='single_test_time_steps_score_reward'+str(i))
    saver(data=train_time_steps_delay, name='single_test_time_steps_score_delay'+str(i))
    saver(data=train_time_steps_waitingtime, name='single_test_time_steps_score_waitingtime'+str(i))
    travel_filname = 'single_test_travel_time'+str(i)
    np.savetxt(travel_filname, train_travel_time)
    env.close()
 
     
           

     

