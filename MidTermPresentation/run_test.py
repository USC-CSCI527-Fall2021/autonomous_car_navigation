import numpy as np
import tensorflow as tf
from tensorflow import keras

import gym
import gym_carla
import carla
import argparse
import tensorflow as tf

params = {
	'number_of_vehicles': 0,
	'number_of_walkers': 0,
	'display_size': 250,  # screen size of bird-eye render
	'display_height' : 512,
	'display_main': True,
	'weather': "WetSunset",
	'max_past_step': 1,  # the number of past steps to draw
	'dt': 0.1,  # time interval between two frames
	'discrete': False,  # whether to use discrete control space
	'discrete_acc': [1.0, 0.0, 1.0],  # discrete value of accelerations
	'discrete_steer': [-1, 0, 1],  # discrete value of steering angles
	'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
	'continuous_steer_range': [-0.2, 0.2],  # continuous steering angle range
	'ego_vehicle_filter': 'vehicle.tesla.model3',  # filter for defining ego vehicle
	'address': "192.168.1.173", #'localhost',
	'port': 8080, #2000 # connection port
	'town': 'Town02', # which town to simulate
	'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
	'max_time_episode': 5000,  # maximum timesteps per episode
	'max_waypt': 12,  # maximum number of waypoints
	'obs_range': 32,  # observation range (meter)
	'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
	'd_behind': 12,  # distance behind the ego vehicle (meter)
	'out_lane_thres': 5.0,  # threshold for out of lane
	'desired_speed': 8,  # desired speed (m/s)
	'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
	'display_route': True,  # whether to render the desired route
	'pixor_size': 64,  # size of the pixor labels
	'pixor': False,  # whether to output PIXOR observation
}


model = keras.models.load_model("ppo_version5.h5")

def read_transform(img):
	return img[76:204,76:204,:]/255

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model", default="ppo_version5", help="Model weights directory")
	parser.add_argument("-i", "--ip", default="192.168.1.173", help="Server ip address")
	parser.add_argument("-p", "--port", default="8080", help="Server Port")
	args = parser.parse_args()
	MODEL = args.model
	SERVER = args.ip   
	PORT = args.port

	params["address"] = SERVER
	params["port"] = int(PORT)
	env = gym.make('carla-v0', params=params)

	observation =  env.reset()
	for _ in range(20): observation, _, _, _ = env.step([1,0])
	done = False
	cumulative_reward = 0
	while not done:
	    #action = np.random.choice( [-1,0,1], p=model.predict( read_transform(observation['birdeye']).reshape( (1, 128,128,3) ))[0])
	    action = np.argmax(model.predict( read_transform(observation['birdeye']).reshape( (1, 128,128,3) ))[0])-1
	    observation, reward, done, _ = env.step( [1,action] )
	    cumulative_reward += reward
	    act = {-1:"Right", 1: "Left", 0:"Straight"}[action]
	    print(f"Action : REWARD-{cumulative_reward}, ACTION-{act}")
