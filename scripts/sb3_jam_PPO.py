import functools
import gymnasium as gym
import highway_env
import pygame
import seaborn as sns
import torch as th
from highway_env.utils import lmap
from stable_baselines3 import DQN, PPO, DDPG,A2C
from gymnasium.wrappers import RecordVideo
import pprint
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from matplotlib import pyplot as plt
from scripts.sb3_highway_ppo_transformer import CustomExtractor, attention_network_kwargs


if __name__ == '__main__':
    env = gym.make('jam_normal_down', render_mode='rgb_array')
    env.reset()
    n_cpu = 6
    batch_size = 96
    policy_kwargs = dict(
        features_extractor_class=CustomExtractor,
        features_extractor_kwargs=attention_network_kwargs,
    )

    model = PPO("MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                n_steps=batch_size * 12 // n_cpu,
                batch_size=batch_size,
                n_epochs=10,
                learning_rate=5e-4,
                gamma=0.8,
                verbose=2,
                tensorboard_log="highway_ppo/")


    #Save the agent
    # model.learn(total_timesteps=int(1e5))
    # model.save("highway_ppo/model_jam")
    #
    model = PPO.load("highway_ppo/model_jam_save_1")
    env = RecordVideo(env, video_folder="highway_ppo/model_jam", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15})  # Higher FPS for rendering


    for videos in range(1):
        done = truncated = False
        obs, info = env.reset()
        guiji = []

        reward_record = []
        safe_record = []
        efficient_record = []

        while not (done or truncated):
            #num = calculate_acceleration(env)
            #env.env.env.env.road.vehicles[]
            #action = np.array(num, dtype=int)
            lane_id = env.env.env.env.road.vehicles[0].lane_index[2]
            t = env.env.env.env.time
            flag = 1
            for vehicle in env.env.env.road.vehicles:
                guiji.append(vehicle)

            for i in range(1, 3):
                if env.env.env.env.road.vehicles[i].lane_index[2] != lane_id :
                    flag = 0
                    break
            if flag == 1:
                for i in range(3,len(env.env.env.env.road.vehicles)):
                    if(env.env.env.env.road.vehicles[i].lane_index[2]==lane_id):
                        if(abs(env.env.env.env.road.vehicles[i].position[0]-env.env.env.env.road.vehicles[1].position[0])<22.0):
                            flag = 0
                            break
            if t < 15.0:
                action = model.predict(obs)
                action = action[0]
            else:
                if flag == 0:
                    action = model.predict(obs)
                    action = action[0]
                else:
                    action = 125
            obs, reward, done, truncated, info = env.step(action)
            env.render()

        reward_record.append(reward)
        safe_record.append(truncated)

        location = []
        for i in range(3):
            location.append(env.env.env.env.controlled_vehicles[i].position[0])
        efficient_record.append(np.mean(location))

    env.close()

