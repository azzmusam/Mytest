import aienvs
import logging
import yaml
import sys
from LoggedTestCase import LoggedTestCase
import numpy as np
from aienvs.Sumo.SumoGymAdapter import SumoGymAdapter
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
	with open("configs/three_config.yaml", 'r') as stream:
		try:
			parameters=yaml.safe_load(stream)['parameters']
		except yaml.YAMLError as exc:
			print(exc)

	env = SumoGymAdapter(parameters)
	mem = 0
	ob = env.reset()	
	print("Loading up the agent's memory with random gameplay")
	done = False
	while mem < 10000000:
		if done:
			ob = env.reset()
		action = env.action_space.sample()
		
		ob, global_reward, done, info = env.step(action)
		#ob = three_padder(ob)
		print(ob.shape)
		mem += 1
		''''print(ob[1].shape)
		for keys in ob[0].keys():
			print(ob[0][keys].shape)'''
		pdb.set_trace()

	    	







