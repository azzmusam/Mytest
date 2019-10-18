import aienvs
import yaml
import logging
import pdb
import sys
from aienvs.Sumo.SumoGymAdapter import SumoGymAdapter
import numpy as np
import os
import pdb

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
    
    	
    # stacked_frames.shape=(1, 90, 108, 2) after reshaping.
    # frame.shape=(2, 160, 1)

    return stacked_frame, stacked_state


if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info("Starting test_traffic_new")

    with open("configs/myconfig.yaml", 'r') as stream:
        try:
            parameters = yaml.safe_load(stream)['parameters']
        except yaml.YAMLError as exc:
            print(exc)

    env = SumoGymAdapter(parameters)

    mem = 0
    stack_size = 1
    taken_Actions = []
    
    while mem < 10000:
        if mem>1 and done:
            pdb.set_trace()
        done = False
        ob = env.reset()
        observation = ob
        observation, stacked_state = stack_frames(stacked_frames= None, frame=observation, buffer_size=stack_size)

        while (not done) and (mem<10000):

            action = env.action_space.sample()
            taken_Actions.append(action.get('0'))
            observation_, reward, done, info = env.step(action)

            observation_, stacked_state_ = stack_frames(stacked_frames = observation, frame=observation_, buffer_size=stack_size)

            observation = observation_

            mem +=1
    

