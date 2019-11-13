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
        self.factor_graph = factor_graph(factored_graph=self._parameters['factored_agents'], 
                                         num_agents=6, 
                                         factored_agent_type=self._parameters['factored_agent_type'],
                                         modelnr = self._parameters['testmodelnr'])
        self.result_initialiser()
        self.result_folder = result_folder
        self.fileinitialiser()       

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
        test_result['traveltime'].append(tt)

    def saver(self, data, name):
        path = os.getcwd()
        name = str(name)
        filename = 'test_result/'+ str(self.result_folder)+ '/' + name +'.csv'
        pathname = os.path.join(path, filename)
        outfile = open(pathname, 'w')
        writer = csv.writer(outfile)
        writer.writerows(map(lambda x:[x], data))
        outfile.close()

    def fileinitialiser(self):
        path = os.getcwd()
        for key in self.test_result.keys():
            filename = 'test_result/' + str(self.result_folder)+ '/' + key +'.csv'
            pathname = os.path.join(path, filename)
            if os.path.exists(os.path.dirname(pathname)):
                continue
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

    def take_action(self, state_graph):
        q_arr = self.factor_graph.get_factored_Q_val(state_graph)
        sum_q_value, best_action, sumo_act = self.factor_graph.b_coord(q_arr)
        return sumo_act

    def save(self, data):
        for key in data.keys():
            result = data[key]
            saver(data=result, name=key)

    def stack_frames(self, stacked_frames, frame, buffer_size):
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

    def stack_state_initialiser(self):
        self.stacked_state_dict = {}
        self.ob_dict = {}

    def stacked_graph(self, ob, initial=True):
        for keys in self._parameters['factored_agents'].keys():
            if initial==True:
                self.ob_dict[keys], self.stacked_state_dict[keys] = self.stack_frames(stacked_frames=None, frame= ob[keys], buffer_size=1) 

            else:
                self.ob_dict[keys], self.stacked_state_dict[keys] = self.stack_frames(stacked_frames=self.ob_dict[keys], frame= ob[keys], buffer_size=1)

        return self.ob_dict, self.stacked_state_dict

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
            self.ob_dict, self.stacked_state_dict = self.stacked_graph(ob=ob[0], initial=True)
            self.reset()
      
            while not done:
                action = self.take_action(self.stacked_state_dict)
                ob_, reward, done, info = self.env.step(action)
                print(reward[1]['result'])
                self.store_result(reward[1])

                self.ob_dict, self.stacked_state_dict  = self.stacked_graph(ob_[0], initial=False)
        
        ob, avg_travel_times, avg_travel_time = self.env.reset(i)
        self.store_tt(avg_travel_time)
        self.save(self.test_result)
        self.env.close()


if __name__=="__main__":
    exp = experimentor(test=True, result_folder='six_intersection', total_simulation=8)
    exp.test()




            


