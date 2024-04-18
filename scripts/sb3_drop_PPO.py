import functools
import gymnasium as gym
import highway_env
import pygame
import seaborn as sns
import torch as th
from highway_env.utils import lmap
from stable_baselines3 import DQN, PPO, DDPG,A2C
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'

def get_decision():
    lane_id = env.env.env.env.road.vehicles[0].lane_index[2]     #获取观察车的车道
    flag = 1

    for i in range(1, 3):
        if env.env.env.env.road.vehicles[i].lane_index[2] != lane_id :
            flag = 0                                             #若受控车不在同一车道，则利用强化学习模型
            break
    if flag == 1:
            for i in range(3,len(env.env.env.env.road.vehicles)):
                if(env.env.env.env.road.vehicles[i].lane_index[2]==lane_id):
                    if(abs(env.env.env.env.road.vehicles[i].position[0]-env.env.env.env.road.vehicles[1].position[0])<22.0):
                        flag = 0                                 #若同一车道上据中心车质心22米以内有车，则利用强化决策模型
                        break
    return flag

if __name__ == '__main__':
    env = gym.make('drop_env', render_mode='rgb_array')
    env.reset()
    n_cpu = 6
    batch_size = 128
    # policy_kwargs = dict(
    #     features_extractor_class=CustomExtractor,
    #     features_extractor_kwargs=attention_network_kwargs,
    # )
    # model = PPO("MLP",
    #             env,
    #             policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
    #             n_steps=batch_size * 12 // n_cpu,
    #             batch_size=batch_size,
    #             n_epochs=10,
    #             learning_rate=5e-4,
    #             gamma=0.8,
    #             seed=0,
    #             verbose=2,
    #             tensorboard_log="highway_ppo/")
    # Train the agent
    # Save the agent
    # model.learn(total_timesteps=int(1e5))
    # model.save("highway_ppo/model_drop_cnn")

    model = PPO.load("highway_ppo/model_drop_PPO")
    env = RecordVideo(env, video_folder="highway_ppo/videos_new", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15})  # Higher FPS for rendering

    reward_record=[]
    safe_record=[]
    efficient_record=[]
    data_speed_1 = []
    data_speed_2 = []
    data_speed_3 = []
    data_x_1 = []
    data_x_2 = []
    data_x_3 = []
    data_y_1 = []
    data_y_2 = []
    data_y_3 = []

    for videos in range(1):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # data_speed_1.append(env.env.env.env.controlled_vehicles[0].speed)
            # data_speed_2.append(env.env.env.env.controlled_vehicles[1].speed)
            # data_speed_3.append(env.env.env.env.controlled_vehicles[2].speed)
            data_x_1.append(env.env.env.env.controlled_vehicles[0].position[0])
            data_x_2.append(env.env.env.env.controlled_vehicles[1].position[0])
            data_x_3.append(env.env.env.env.controlled_vehicles[2].position[0])
            data_y_1.append(env.env.env.env.controlled_vehicles[0].position[1])
            data_y_2.append(env.env.env.env.controlled_vehicles[1].position[1])
            data_y_3.append(env.env.env.env.controlled_vehicles[2].position[1])

            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
        x_label = list(range(len(data_x_1)))
        resault_1 = []
        resault_2 = []
        for i in range((len(data_x_1))):
            resault_1.append(data_x_2[i]-data_x_1[i])
            resault_2.append(data_x_3[i]-data_x_2[i])

        fig = plt.figure(1)
        plt.plot(x_label, resault_1, label='车头间距1')
        plt.plot(x_label, resault_2, label='车头间距2')
        plt.xlabel("time")
        plt.ylabel("车头间距")
        plt.title("车头间距变化曲线")
        plt.legend()
        plt.show()
    env.close()






