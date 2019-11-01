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
import collections


all_list = []
for j in it.product((0,1), repeat = 2):
	all_list.append(j) 

# change repeat depending on the number of agents(traffic light intersection).


def action_hot_encoder(actions, list_action):
	action = np.zeros((2**(len(actions.keys()))))
	value_list = tuple(actions.values())
	for key, val in enumerate(list_action):
		if val == value_list:
			action[key] = 1.
			break 
	return action


def action_decoder(encoded_action, all_list):
	index = (list(np.where(encoded_action==1.))[0])[0]
	decoded_action = collections.OrderedDict()
	for i in range(len(encoded_action)):
		try:
			decoded_action[str(i)] = all_list[index][i]
		except:
			break
	return decoded_action


if __name__ == "__main__":
	logging.info("Starting test_traffic_new")
	with open("configs/myconfig.yaml", 'r') as stream:
		try:
			parameters=yaml.safe_load(stream)['parameters']
		except yaml.YAMLErtiror as exc:
			print(exc)

	env = SumoGymAdapter(parameters)
	mem = 0
	ob = env.reset()
	print("Loading up the agent's memory with random gameplay")
	done = False
	encoded_act = []
	decoded_act = []
	act = []
	while mem < 100000:
		if done:
			ob = env.reset()

		actions = env.action_space.sample()
		#act.append(actions)

		action = action_hot_encoder(actions, all_list)
		#encoded_act.append(action)

		decoded_act.append(action_decoder(action, all_list))

		ob, global_reward, done, info = env.step(actions)

		mem += 1

		if mem==15:
			pdb.set_trace()

	    	







