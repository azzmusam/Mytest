# import sys
# import os
# import unittest
import logging
from test.LoggedTestCase import LoggedTestCase
# from aienvs.FactoryFloor.FactoryFloor import FactoryFloor
# import random
# import yaml
from numpy import array, array_equal
# from numpy import ndarray
# from aienvs.Environment import Env
# from aienvs.runners.Episode import Episode
# from aienvs.runners.Experiment import Experiment
# from aienvs.utils import getParameters
from unittest.mock import Mock
# from aienvs.loggers.JsonLogger import JsonLogger
# import io
from aienvs.GroupingRobots.WorldState import WorldState
from aienvs.GroupingRobots.Robot import Robot

# from aienvs.loggers.PickleLogger import PickleLogger
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

ROBOT1 = "robot1"
ROBOT2 = "robot2"
ROBOT3 = "robot3"
# 3 different "positions". Nunpy has its own specialty functions
# so we need to stick with numpy things to get things working...
A = array([1, 1])
B = array([2, 2])
C = array([3, 3])


class testWorldState(LoggedTestCase):
    """
    This is an integration test that also tests aiagents.
    aiagents project must be installed 
    """

    def test_smoke(self):
        WorldState({}, Mock(), 1)
        
    def test_step(self):
        s = WorldState({}, Mock(), 1)
        s = s.withStep()
        self.assertEquals(2, s.getSteps())
    
    def test_withRobot(self):
        s = WorldState({}, Mock(), 1)
        robot = Mock()
        robot.getId = Mock(return_value="robot1")
        s = s.withRobot(robot)
        self.assertEquals([robot], s.getRobots())

    def test_freeWithoutRobot(self):
        env = Mock()
        env.getFreeMapPositions = Mock(return_value=[A, B, C])
        robot = self._mockRobot(ROBOT1, A)
        s = WorldState({ROBOT1:robot}, env, 1)

        free = s.getFreeWithoutRobot()
        self.assertTrue(array_equal([B, C], free))
        
    def test_getGroupedRobots(self):
        env = Mock()
        robot1 = self._mockRobot(ROBOT1, A)
        robot2 = self._mockRobot(ROBOT2, B)
        robot3 = self._mockRobot(ROBOT3, A)
        s = WorldState({ROBOT1:robot1, ROBOT2:robot2, ROBOT3:robot3}, env, 1)
        self.assertEquals(set([robot3, robot1]), set(s.getGroupedRobots()))

    ################# private #################
    def _mockRobot(self, id:str, pos) -> Robot:
        robot = Mock()
        robot.getId = Mock(return_value=id)
        robot.getPosition = Mock(return_value=pos)
        return robot
