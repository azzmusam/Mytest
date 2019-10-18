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



if __name__ == "__main__":
	logging.info("Starting test_traffic_new")
	with open("configs/new_config.yaml", 'r') as stream:
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
		mem += 1
		

	    	







