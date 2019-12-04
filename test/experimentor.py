import numpy as np
import os
import sys
#from imp_DQRN import Agent
import yaml
import logging
from LoggedTestCase import LoggedTestCase
from aienvs.Sumo.sumogym import SumoGymAdapter
import pdb
logger = logging.getLogger()
logger.setLevel(logging.INFO)
from factor_graph import factor_graph
import csv
from maxplus import maxplus

class experimentor():
    
    def __init__(self, test, result_folder: str, total_simulation=8):
        logging.info("Starting test_traffic_new")
        with open("configs/testconfig.yaml", 'r') as stream:
            try:
                parameters=yaml.safe_load(stream)['parameters']
            except yaml.YAMLError as exc:
                print(exc)

        self.env = SumoGymAdapter(parameters)
        #self.test = test
        self.total_simulation = total_simulation
        self._parameters = parameters

        #*************CHANGE THE NUMBER OF AGENST FOR EVERY DIFFERENT TYPE OF CONFIGURATION*************
        #*************EVEN IN CASE OF 3 FACTOR TESTMODELNR DICT IN CONFIG SHOULD HAVE 'vertical' AS KEY********************

        self.factor_graph = factor_graph(factored_graph=self._parameters['factored_agents'], 
                                         num_agents=6, 
                                         factored_agent_type=self._parameters['factored_agent_type'],
                                         modelnr = self._parameters['testmodelnr'],
                                         self._parameters['coordination_algo'])
        if self._parameters['coordination_algo'] == 'maxplus':
            self.maxplus = maxplus(regular_factor=self._parameters['factored_agents'], agent_neighbour_combo=self._parameters['agent_neighbour_combo'], max_iter=self._parameters['max_iter'])
        self.result_initialiser()
        self.result_folder = result_folder
        self.fileinitialiser()
        self.factored_agent_type = self._parameters['factored_agent_type']

    def result_initialiser(self):
        self.test_result = {}
        self.test_result['result'] = []
        self.test_result['num_teleports'] = []
        self.test_result['emergency_stops'] = []
        self.test_result['total_delay'] = []
        self.test_result['total_waiting'] = []
        self.test_result['traveltime'] = []

    def store_result(self, reward):
        for keys in reward.keys():
            self.test_result[keys].append(reward[keys])

    def store_tt(self, tt):
        self.test_result['traveltime'].append(tt)

    def saver(self, data, name):
        path = os.getcwd()
        name = str(name)
        filename = 'test_result/'+ str(self.result_folder)+ '/' + name + '_' + str(self._parameters['testmodelnr']['vertical']) + '.csv'
        pathname = os.path.join(path, filename)
        outfile = open(pathname, 'w')
        writer = csv.writer(outfile)
        writer.writerows(map(lambda x:[x], data))
        outfile.close()

    def fileinitialiser(self):
        path = os.getcwd()
        for key in self.test_result.keys():
            filename = 'test_result/' + str(self.result_folder)+ '/' + key + '_' + str(self._parameters['testmodelnr']['vertical'])  + '.csv'
            pathname = os.path.join(path, filename)
            if os.path.exists(os.path.dirname(pathname)):
                print('Result directroy already exists: ', pathname)
            else:
                os.makedirs(os.path.dirname(pathname))
            with open(pathname, "w") as my_empty_csv:
                pass

    def file_rename(self, name, iternr):
        path = os.getcwd()
        res_dir = os.path.join(path, 'test_result', str(self.result_folder))
        #filename = 'test_result/'+ name + str(iternr-10000) +'.csv'
        #res_dir = os.path.join(path, filename)
        oldname = str(name) + str(iternr-10000)  + '.csv'
        newname = str(name) + str(iternr) + '.csv'
        os.rename(res_dir+ '/' + oldname, res_dir + '/' + newname)

    def reset(self):
        self.factor_graph.reset()

    def qarr_key_changer(self, q_arr):
        q_val = {}
        for keys in q_arr.keys():
            q_val[str(self._parameters['factored_agents'][keys])] = q_arr[keys]
        return q_val

    def take_action(self, state_graph):
        q_arr = self.factor_graph.get_factored_Q_val(state_graph)
        if self._parameters['coordination_algo'] == 'brute':
            sum_q_value, best_action, sumo_act = self.factor_graph.b_coord(q_arr)
        elif self._parameters['coordination_algo'] == 'maxplus':
            self.maxplus.initialise_again()
            q_arr = self.qarr_key_changer(q_arr)
            sumo_act = self.maxplus.max_plus_calculator(q_arr)
        else:
            sum_q_value, best_action, sumo_act = self.factor_graph.individual_coord(q_arr)
        return sumo_act

    def save(self, data):
        for key in data.keys():
            result = data[key]
            self.saver(data=result, name=key)

    def stack_frames(self, stacked_frames, frame, buffer_size, config):
        if stacked_frames is None:
            stacked_frames = np.zeros((buffer_size, *frame.shape))
            for idx, _ in enumerate(stacked_frames):
                if config=='horizontal':
                    stacked_frames[idx, :] = frame.transpose()
                else:
                    stacked_frames[idx, :] = frame
        else:
            stacked_frames[0:buffer_size-1, :] = stacked_frames[1:, :]
            if config== 'horizontal':
                stacked_frames[buffer_size-1, :] = frame.transpose()
            else:
                stacked_frames[buffer_size-1, :] = frame

        stacked_frame = stacked_frames
        stacked_state = stacked_frames.transpose(1,2,0)[None, ...]

        return stacked_frame, stacked_state

    def stack_state_initialiser(self):
        self.stacked_state_dict = {}
        self.ob_dict = {}

    def stacked_graph(self, ob, initial=True):
        for keys in self._parameters['factored_agents'].keys():
            if initial==True:
                self.ob_dict[keys], self.stacked_state_dict[keys] = self.stack_frames(stacked_frames=None, frame= ob[keys], buffer_size=1, config = self.factored_agent_type[keys]) 

            else:
                self.ob_dict[keys], self.stacked_state_dict[keys] = self.stack_frames(stacked_frames=self.ob_dict[keys], frame= ob[keys], buffer_size=1, config = self.factored_agent_type[keys])

        return self.ob_dict, self.stacked_state_dict

    def six_padder(self, ob):
        for keys in ob[0].keys():
            if keys == '0':
                ob[0][keys] = np.pad(ob[0][keys], ((15,15),(1,0)), 'constant', constant_values=(0,0))
            elif keys == '1':
                ob[0][keys] = np.pad(ob[0][keys], ((15,15),(1,1)), 'constant', constant_values=(0,0))
            elif keys == '2':
                ob[0][keys] = np.pad(ob[0][keys], ((15,14),(1,0)), 'constant', constant_values=(0,0))
            elif keys =='3':
                ob[0][keys] = np.pad(ob[0][keys], ((15,14),(1,1)), 'constant', constant_values=(0,0))
            elif keys == '4':
                ob[0][keys] = ob[0][keys][:84,:]
                ob[0][keys] = np.pad(ob[0][keys], ((0,0),(15,14)), 'constant', constant_values=(0,0))
            elif keys == '5':
                ob[0][keys] = ob[0][keys][:84,:]
                ob[0][keys] = np.pad(ob[0][keys], ((0,0),(15,16)), 'constant', constant_values=(0,0))
            elif keys == '6':
                ob[0][keys] = ob[0][keys][:84, :]
                ob[0][keys] = np.pad(ob[0][keys], ((0,0),(15,14)), 'constant', constant_values=(0,0))
        return ob

    def three_padder(self, ob):
        for keys in ob[0].keys():
            if keys == '1':
                ob[0][keys] = np.pad(ob[0][keys], ((0,0),(1,0)), 'constant', constant_values= (0,0))
        return ob

    def test(self):
        for i in range(self.total_simulation):
            done = False
            self.stack_state_initialiser()
            if i > 0:
                try:
                    ob, avg_travel_times, avg_travel_time = self.env.reset(i)
                    self.store_tt(avg_travel_time)
                    print(self.test_result['traveltime'])
                except:
                    ob = self.env.reset()
            else:
                ob = self.env.reset()
            ob = self.six_padder(ob)
            self.ob_dict, self.stacked_state_dict = self.stacked_graph(ob=ob[0], initial=True)
            self.reset()
            while not done:
                action = self.take_action(self.stacked_state_dict)
                ob_, reward, done, info = self.env.step(action)
                ob_ = self.six_padder(ob_)
                print(reward[1]['result'])
                self.store_result(reward[1])
                self.ob_dict, self.stacked_state_dict  = self.stacked_graph(ob_[0], initial=False)
        ob, avg_travel_times, avg_travel_time = self.env.reset(i)
        self.store_tt(avg_travel_time)
        self.save(self.test_result)
        self.env.close()


if __name__=="__main__":
    #******************DEFINE THE RESULT FOLDER WITH THE TYPE AND KIND OF ALGORITHM USING*************************
    #************************************************CHANGE PADDER*****************************************
    exp = experimentor(test=True, result_folder='six_intersection/maxplus', total_simulation=8)
    exp.test()
