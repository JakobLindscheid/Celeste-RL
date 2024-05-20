import os
from celeste_env import CelesteEnv
from config import Config
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.callbacks import BaseCallback
from sb_ppo.callback import CheckpointCallback


class ppo:
    def __init__(self, config_ppo, config_env: Config) -> None:
        self.checkpoint_callback = CheckpointCallback(
            save_freq=1024,
            save_path="./sb_ppo/logs/",
        )

        # load
        # model = PPO.load(path=r".\logs\rl_model.zip", env=env) 
        
    def train(self,env,config,metrics):
        self.model = PPO("MultiInputPolicy", env, verbose=1, n_steps=256, batch_size=64)
        env_checker.check_env(env)
        # train
        self.model.learn(total_timesteps=50_000, progress_bar=True, callback=self.checkpoint_callback)

        # test
        obs, _ = env.reset()
        while True:
            action, _state = self.model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
