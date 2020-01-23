import aienvs
import logging
import yaml
import sys
from LoggedTestCase import LoggedTestCase
import numpy as np
from aienvs.Sumo.sumogym import SumoGymAdapter
import pdb
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import itertools as it


def three_padder(ob):
    for keys in ob[0].keys():
        if keys == '1':
            ob[0][keys] = np.pad(ob[0][keys], ((0,0),(1,0)), 'constant', constant_values= (0,0))
    return ob

if __name__ == "__main__":
	logging.info("Starting test_traffic_new")
	with open("configs/eight_ind_config.yaml", 'r') as stream:
		try:
			parameters=yaml.safe_load(stream)['parameters']
		except yaml.YAMLError as exc:
			print(exc)

	env = SumoGymAdapter(parameters)
	mem = 0
	ob = env.reset()
	for keys in ob[0].keys():
		print(keys, ob[0][keys].shape)
	pdb.set_trace()
	print("Loading up the agent's memory with random gameplay")
	done = False
	taken_action = {}
	taken_action['0'] = []
	taken_action['1']= []
	for i in range(8):
		done = False
		if i>0:
			ob, avgtts, avgtt = env.reset(i)
			print(avgtts)
		else:
			ob = env.reset()
		while not done:
			action = env.action_space.sample()
			for keys in action.keys():
				if action[keys] == 1:
					taken_action[str(action[keys])].append(1)
				else:
					taken_action[str(action[keys])].append(1)		
			ob, global_reward, done, info = env.step(action)
		#ob = three_padder(ob)
		#print(ob.shape)
		#mem += 1
		''''print(ob[1].shape)
		for keys in ob[0].keys():
			print(ob[0][keys].shape)'''
	ob, avgtts, avgtt = env.reset(i)
	print(avgtts)
	for keys in taken_action.keys():
		avg = np.sum(taken_action[keys])
		print('number of times action ', keys, 'taken in 8 simulation is', avg)
	env.close()
