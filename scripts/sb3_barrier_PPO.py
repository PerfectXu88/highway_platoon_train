import functools
import gymnasium as gym
import highway_env
import pygame
import seaborn as sns
import torch as th
from highway_env.utils import lmap
from stable_baselines3 import DQN, PPO, DDPG, SAC, A2C
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
    env = gym.make('barrier_env',render_mode = 'rgb_array')
    env.reset()
    n_cpu = 6
    batch_size = 128
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
                seed=2024,
                tensorboard_log="highway_ppo/")

    model.learn(total_timesteps=int(1e5))
    model.save("highway_ppo/model_barrier_seed_2024")
    #
    model = PPO.load("highway_ppo/model_barrier_seed_0")
    env = RecordVideo(env, video_folder="highway_ppo/model_barrier_seed_0", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15})  # Higher FPS for rendering
    # model = DQN('MlpPolicy', env,
    #             policy_kwargs=dict(net_arch=[256, 256]),
    #             learning_rate=5e-4,
    #             buffer_size=15000,
    #             learning_starts=200,
    #             batch_size=32,
    #             gamma=0.8,
    #             train_freq=1,
    #             gradient_steps=1,
    #             target_update_interval=50,
    #             verbose=1,
    #             tensorboard_log="highway_dqn/")
    # model = A2C(
    #     'MlpPolicy',
    #     env,
    #     policy_kwargs=dict(net_arch=[256, 256]),
    #     learning_rate=5e-4,
    #     gamma=0.8,
    #     verbose=1,
    #     tensorboard_log="highway_a2c/"
    # )
    # #Save the agent
    # model.learn(total_timesteps=int(1e5))
    # model.save("highway_a2c/model_barrier_A2C")
    #
    # model = A2C.load("highway_a2c/model_barrier_A2C")
    # env = RecordVideo(env, video_folder="highway_A2C/video_barrier_A2C", episode_trigger=lambda e: True)
    # env.unwrapped.set_record_video_wrapper(env)
    # env.configure({"simulation_frequency": 15})  # Higher FPS for rendering

    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # num = calculate_acceleration(env)
            # env.env.env.env.road.vehicles[]
            # action = np.array(num, dtype=int)
            lane_id = env.env.env.env.road.vehicles[0].lane_index[2]
            t = env.env.env.env.time
            flag = 1
            for i in range(1, 3):
                if env.env.env.env.road.vehicles[i].lane_index[2] != lane_id:
                    flag = 0
                    break
            if flag == 1:
                for i in range(3, len(env.env.env.env.road.vehicles)):
                    if (env.env.env.env.road.vehicles[i].lane_index[2] == lane_id):
                        if (abs(env.env.env.env.road.vehicles[i].position[0] -
                                env.env.env.env.road.vehicles[1].position[0]) < 22.0):
                            flag = 0
                            break
            if t < 10.0:
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
    env.close()
    
