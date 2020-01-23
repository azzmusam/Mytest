import aienvs
import os
import logging
import yaml
import sys
from LoggedTestCase import LoggedTestCase
import numpy as np
from aienvs.Sumo.SumoGymAdapter import SumoGymAdapter
import pdb
from multi_DQRN import Agent
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import itertools as it


def three_padder(ob):
    for keys in ob[0].keys():
        if keys == '1':
            ob[0][keys] = np.pad(ob[0][keys], ((0,0),(1,0)), 'constant', constant_values= (0,0))
    return ob

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

if __name__ == "__main__":
	logging.info("Starting test_traffic_new")
	with open("configs/new_config.yaml", 'r') as stream:
		try:
			parameters=yaml.safe_load(stream)['parameters']
		except yaml.YAMLError as exc:
			print(exc)
	env = SumoGymAdapter(parameters)
	mem_size = None
	agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.00025, input_dims=(84,84,1),
	              act_per_agent=2, num_agents=1, mem_size=mem_size, batch_size=32)
	filename = 'deepqnet.ckpt-10000' 
	path = os.getcwd() 
	chkpt = os.path.join(*[path, 'tmp', 'q_eval', filename])
	agent.load_models(chkpt)
	print('LOADED CHECKPOINT:', filename)
	mem = 0
	#ob = env.reset()	
	print("Loading up the agent's memory with random gameplay")
	done = False
	taken_action = {}
	taken_action['0'] = []
	taken_action['1']= []
	for i in range(8):
		done = False
		if i >0:
			observation, avgtts, avgtt = env.reset(i)
			print(avgtts)
		else:
			observation = env.reset()
		observation, stacked_state = stack_frames(stacked_frames=None, frame= observation, buffer_size=1)
		agent.reset()
		while not done:
			#action = env.action_space.sample()
			action = agent.test(stacked_state)
			#print(action)
			observation_, reward, done, info = env.step(action)
			for keys in action.keys():
				if action[keys] == 1:
					taken_action[str(action[keys])].append(1)
				else:
					taken_action[str(action[keys])].append(1)
			#ob, global_reward, done, info = env.step(action)
			observation_, stacked_state_ = stack_frames(stacked_frames=observation, frame=observation_, buffer_size=1)
			observation = observation_
			stacked_state = stacked_state_
		#ob = three_padder(ob)
		#print(ob.shape)
		#mem += 1
		''''print(ob[1].shape)
		for keys in ob[0].keys():
			print(ob[0][keys].shape)'''
	for keys in taken_action.keys():
		avg = np.sum(taken_action[keys])
		print('number of times action ', keys, 'taken in 8 simulation is', avg)
	ob, avgtts, avgtt = env.reset(i)
	print(avgtts)
