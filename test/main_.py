import aienvs
from multi_DQRN import DeepQNetwork, Agent
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


    with open("configs/multi_hor_config.yaml", 'r') as stream:
        try:
            parameters = yaml.safe_load(stream)['parameters']
        except yaml.YAMLError as exc:
            print(exc)

    env = SumoGymAdapter(parameters)

    mem_size =30000
    i=0
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.00025, input_dims=(84,84,1),
                  act_per_agent=2, num_agents=2, mem_size=mem_size, batch_size=32)

    maximum_time_steps = 1000000
    stack_size = 1

    print("Loading up the agent's memory with random gameplay")
    while agent.mem_cntr < mem_size:
        done = False
        observation = env.reset()
        observation, stacked_state = stack_frames(stacked_frames = None, frame = observation, buffer_size = stack_size)

        while (not done) and (agent.mem_cntr < mem_size):
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            observation_, stacked_state_ = stack_frames(stacked_frames = observation, frame = observation_, buffer_size = stack_size)
            agent.store_transition(stacked_state, action, reward, stacked_state_, int(done))
            agent.upgrade()
            observation = observation_
            stacked_state = stacked_state_
            print('MEMORY_COUNTER: ', agent.mem_cntr)
    print("Done with random game play. Game on.")

    while i < maximum_time_steps:
        done = False
        if i>0:    
            try:
                observation, average_train_times, average_train_time = env.reset(i)
                print(average_train_times)
            except:
                observation = env.reset()
            observation, stacked_state = stack_frames(stacked_frames= None, frame=observation, buffer_size=stack_size)
            agent.reset()
        else:     
            observation = env.reset()
            observation, stacked_state = stack_frames(stacked_frames= None, frame=observation, buffer_size=stack_size)
            agent.reset()

        while (not done) and  i < maximum_time_steps:
            action = agent.choose_action(stacked_state)
            observation_, reward, done, info = env.step(action)
            observation_, stacked_state_ = stack_frames(stacked_frames=observation, frame=observation_, buffer_size=stack_size)
            agent.store_transition(stacked_state, action, reward, stacked_state_, int(done))
            observation = observation_
            stacked_state = stacked_state_
            agent.learn()

            i +=1
            agent.upgrade()

            if i % 10000==0:
                agent.save_models(i)
    env.close()
