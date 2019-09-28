import aienvs
from multi_DQRN import DeepQNetwork, Agent
import yaml
import logging
import pdb
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

    return stacked_frame, stacked_state


if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info("Starting test_traffic_new")

    with open("configs/new_config.yaml", 'r') as stream:
        try:
            parameters = yaml.safe_load(stream)['parameters']
        except yaml.YAMLError as exc:
            print(exc)

    env = SumoGymAdapter(parameters)

    load_checkpoint = os.path.isfile('tmp/q_eval/deepqnet.ckpt')

    mem_size = 100

    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.00025, input_dims=(84,84,1),
                  act_per_agent=2, num_agents=1, mem_size=mem_size, batch_size=32)

    if load_checkpoint:
        agent.load_models()

    maximum_time_steps = 1000000
    train_time_steps_score = []
    train_episode_score = []
    train_travel_time = []
    stack_size = 1
    i = 0

    test_int = 100 #test interval
    maximum_test_time = 1000 #no of test simulation
    total_simulation = 3
    test_reward = []
    test_average_travel_time = []
    

    print("Loading up the agent's memory with random gameplay")

    while agent.mem_cntr < mem_size:
        done = False
        observation = env.reset()
        observation, stacked_state = stack_frames(stacked_frames = None, frame = observation, buffer_size = stack_size)

        while (not done) and (agent.mem_cntr < mem_size):
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            observation_, stacked_state_ = stack_frames(stacked_frames = observation, frame = observation_, buffer_size = stack_size)
            agent.store_transition(stacked_state, action, reward, stacked_state_, int(done))
            agent.upgrade()
            observation = observation_
            stacked_state = stacked_state_
            print('MEMORY_COUNTER: ', agent.mem_cntr)
    print("Done with random game play. Game on.")

    while i < maximum_time_steps:
        done = False
        try:
            observation, average_train_times, average_train_time = env.reset(i)
            train_travel_time.append(average_train_time)
            print(train_travel_time)
        except:
            observation = env.reset()
        observation, stacked_state = stack_frames(stacked_frames= None, frame=observation, buffer_size=stack_size)
        agent.reset()

        score = 0
        while (not done) and  i < maximum_time_steps:
            action = agent.choose_action(stacked_state)
            observation_, reward, done, info = env.step(action)
            observation_, stacked_state_ = stack_frames(stacked_frames=observation, frame=observation_, buffer_size=stack_size)
            agent.store_transition(stacked_state, action, reward, stacked_state_, int(done))
            train_time_steps_score.append(reward)
            score +=reward['result']
            observation = observation_
            stacked_state = stacked_state_
            print("reward: ", reward['result'])
            print("waiting Time: ", reward['total_waiting'])
            print("Delay: ", reward['total_delay'])
            agent.learn()

            test_it = agent.mem_cntr
            
            if test_it % test_int==0 and i>0:
                done = False 
                try:
                    observation, average_train_times, average_train_time = env.reset(i)
                    train_travel_time.append(average_train_time)
                    print(train_travel_time)
                except:
                    observation = env.reset()
                observation, stacked_state = stack_frames(stacked_frames= None, frame=observation, buffer_size=stack_size)
                agent.reset()
                test_it +=1
                #tt = 0
                if i == 60000:
                    pdb.set_trace()

                for tt in range(total_simulation):
                    try:
                        observation, average_travel_times, average_travel_time = env.reset(tt+1)
                        test_average_travel_time.append(average_travel_time)
                        print(test_average_travel_time)
                    except:
                        observation = env.reset()
                    observation, stacked_state = stack_frames(stacked_frames= None, frame=observation, buffer_size=stack_size)
                    agent.reset()
                    sim_step = 0
                    while (not done) or sim_step<maximum_test_time:
                        action = agent.test(stacked_state) 
                        observation_, reward, done, info = env.step(action)
                        observation_, stacked_state_ = stack_frames(stacked_frames=observation, frame=observation_, buffer_size=stack_size)
                        agent.store_transition(stacked_state, action, reward, stacked_state_, int(done))
                        test_reward.append(reward)
                        agent.upgrade()
                        sim_step+=1
                    done = False
                    print('ENDING TEST NO: ', tt)
            i +=1
            agent.upgrade()             
            if i == maximum_time_steps:
                break

    np.savetxt('scores/time_step.dat', time_steps_score, fmt='%.3f')
    np.savetxt('scores/episode.dat', episode_scores, fmt='%.3f')
