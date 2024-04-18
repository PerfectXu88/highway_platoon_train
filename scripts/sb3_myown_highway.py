import gymnasium as gym
import highway_env
from stable_baselines3 import DQN, PPO, DDPG
from gymnasium.wrappers import RecordVideo
import pprint
import itertools

if __name__ == '__main__':
    env = gym.make('my_own_highway',render_mode = 'rgb_array')
    env.reset()
    # pprint.pprint(env.config)
    model = PPO('MlpPolicy',env,
                policy_kwargs=dict(net_arch = [256,256]),
                learning_rate=1e-4,
                buffer_size=15000,
                learning_starts=0,
                batch_size=64,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1,
                tensorboard_log='highway_dqn/'
                 )
    # model.learn(int(1e6))
    model.save('highway_dqn_2/model')


    #Load and test saved model
    model = PPO.load('highway_dqn_1/model')
    env = RecordVideo(env, video_folder="highway_dqn/videos_6", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15})  # Higher FPS for rendering


    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs,deterministic=True)
            obs,reward,done,truncated,info = env.step(action)

            # next_obs, reward, done, truncated, info = env.step(action)
            # for obs_i, action_i, next_obs_i in zip(obs, action, next_obs):
            #     model.update(obs_i, action_i, next_obs_i, reward, info, done, truncated)
            # obs = next_obs
    env.close()
    
