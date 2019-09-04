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
actions = []

while mem < 100000:
    if done:
        ob = env.reset()
    action = env.action_space.sample()
    actions.append(env._intToPhaseString('0', action.get('0')))
    ob, global_reward, done, info = env.step(action)
    if global_reward['total_waiting']>0:
        pdb.set_trace()
    mem += 1

    if mem==450:
        pdb.set_trace()

	    	







