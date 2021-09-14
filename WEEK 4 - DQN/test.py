import gym
import gym_carla
import carla
import cv2
import numpy as np
from dqn_agent import DQN_AGENT

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
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'task_mode': 'roundabout',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 1000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
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

env = gym.make('carla-v0', params=params)
current_state = env.reset()

agent = DQN_AGENT()
agent.model.load_weights('models/VGG_1_episode-70_1631592886.hdf5')

input("Press Enter to Start")
for _ in range(20):
    current_state, _, _, _ = env.step([1,0])

current_state = processFrame(current_state['birdeye'])
while True:
    action = np.argmax(agent.get_qs(current_state))
    new_state, reward, done, _ = env.step([1,action-1])
    current_state = processFrame(new_state['birdeye'])
    if done: 
        input("Episode ends here. Press Enter to exit.")
        break