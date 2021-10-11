import gym
import gym_carla
import carla
from dqn_agent import DQN_AGENT
from time import time, sleep
import numpy as np
import cv2
import random

import os

from queue import deque


params = {
    'number_of_vehicles': 0,
    'number_of_walkers': 0,
    'display_width': 512,  # screen size of bird-eye render
    'display_height' : 512,
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
    'discrete_acc': [1.0, 0.0, 1.0],  # discrete value of accelerations
    'discrete_steer': [-1, 0, 1],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.2, 0.2],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'address': 'localhost',
    'port': 2000,  # connection port
    'town': 'Town02',  # which town to simulate
    'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 2000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 1.0,  # threshold for out of lane
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
  }


KERNEL = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
def processFrame(frame):
    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #convert image to black and white
    frame = cv2.resize(frame,(84,84),interpolation=cv2.INTER_AREA)
    #_ , frame = cv2.threshold(frame,50,255,cv2.THRESH_BINARY)
    #frame = cv2.blur(frame,(5,5))
    frame = cv2.filter2D(frame,-1,KERNEL)
    #frame = cv2.Canny(frame,100,200)
    frame = frame.astype(np.float64)/255
    return frame



FPS = 60
# For stats
epsilon = 1
AGGREGATE_STATS_EVERY = 10
EPSILON_DECAY = 0.99 ## 0.9975 99975
MIN_EPSILON = 0.001
MIN_REWARD = -200
MODEL_NAME = "VGG_1"
REPLAY_MEMORY = 3000
# For more repetitive results
random.seed(1)
np.random.seed(1)
#tf.set_random_seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

# Create agent and environment
agent = DQN_AGENT()
replay_memory = deque( maxlen = REPLAY_MEMORY )


env = gym.make('carla-v0', params=params)
agent.get_qs(np.ones((84, 84, 3)))
episode = 1
while episode != 200:
        episode_reward = 0
        current_state = env.reset()
        for _ in range(20):
            current_state, _, _, _ = env.step([1,0])
        current_state = processFrame(current_state['birdeye'])
        
        done = False
        episode_start = time()
        while True:
            if np.random.random() > epsilon: 
                action = np.argmax(agent.get_qs(current_state))
            else: 
                action = np.random.randint(0, 3)

            new_state, reward, done, _ = env.step([1,action-1])
            new_state = processFrame(new_state['birdeye'])
            replay_memory.append((current_state, action, reward, new_state, done))
            current_state = new_state
            if done: break

        agent.train(replay_memory)
        if episode%25 == 0:
            agent.model.save_weights(f'models/{MODEL_NAME}_episode-{episode}_{int(time())}.hdf5')

        epsilon = max(MIN_EPSILON, epsilon*EPSILON_DECAY)
        with open("logs_dqn.txt", "a") as log:
            print(f'Training: {episode}/200 - Epsilon: {epsilon:.2f} - Replay Mem: {len(replay_memory)}', file=log)
        episode+=1

# Set termination flag for training thread and wait for it to finish
agent.model.save_weights(f'models/{MODEL_NAME}_episode-{episode}_{int(time())}.hdf5')