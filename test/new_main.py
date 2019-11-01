import aienvs
from new_dqn import DeepQNetwork, Agent
import yaml
import logging
import pdb
from aienvs.Sumo.SumoGymAdapter import SumoGymAdapter
import numpy as np
import os
import pdb
import csv
from statics_control import *

def preprocess(observation):
    return np.mean(observation[30:,:], axis=2).reshape(180, 160, 1)

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
    # Starting the Environment.

    load_checkpoint = os.path.isfile('tmp/q_eval/deepqnet.ckpt')

    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.00025, input_dims=(140,140,2),
                  n_actions=2, mem_size=50000, batch_size=32)

    if load_checkpoint:
        agent.load_models()
    episode_scores = []
    maximum_episode_time = 100
    maximum_time_steps = 1000000
    time_steps_score = []
    stack_size = 1
    i = 0
    episode_number = 0

    print("Loading up the agent's memory with random gameplay")

    while agent.mem_cntr < 50000:

        done = False
        observation = env.reset()
        observation, stacked_state = stack_frames(stacked_frames = None, frame = observation, buffer_size = stack_size)

        while (not done) and (agent.mem_cntr < 50000):
            action = env.action_space.sample()

            observation_, reward, done, info = env.step(action)
            env.stats_control.add_reward(reward)

            observation_, stacked_state_ = stack_frames(stacked_frames = observation, frame = observation_, buffer_size = stack_size)

            agent.store_transition(stacked_state, action,
                                   reward, stacked_state_, int(done))

            observation = observation_
            stacked_state = stacked_state_
            print('MEMORY_COUNTER: ', agent.mem_cntr)
    print("Done with random game play. Game on.")

    while i < maximum_time_steps:

        if episode_number % 10 == 0 and episode_number > 0:
            #avg_score = np.mean(episode_scores[max(0, episode_number-10):(episode_number+1)])
            print('episode: ', episode_number, 'score: ', score,
                  #'average score  %.3f' % avg_score,
                  'epsilon %.3f' % agent.epsilon)
            agent.save_models(episode_number= episode_number)

        try:
            print('episode: ', episode_number, 'score: ', score)
        except:
            print('episode: ', episode_number, 'score: ', 0)
            pdb.set_trace()

        done = False
        observation = env.reset(episode_number)
        observation, stacked_state = stack_frames(stacked_frames= None, frame=observation, buffer_size=stack_size)

        score = 0
        n = 0
        while not done and n < maximum_episode_time:

            action = agent.choose_action(stacked_state)
            observation_, reward, done, info = env.step(action)
            env.sats_control.add_reward(reward)

            observation_, stacked_state_ = stack_frames(stacked_frames=observation, frame=observation_, buffer_size=stack_size)

            score += reward['result']

            time_steps_score.append(reward['result'])

            agent.store_transition(stacked_state, action, reward, stacked_state_, int(done))
            observation = observation_
            stacked_state = stacked_state_
            print("reward: ", reward['result'])
            print("waiting Time: ", reward['total_waiting'])
            print("Delay: ", reward['total_delay'])
            agent.learn()

            n +=1

        episode_scores.append(score)
        i +=n

    with open('time_step.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for val in time_steps_score:
            writer.writerow([val])

    with open('episode.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for val in episode_scores:
            writer.writerow([val])
