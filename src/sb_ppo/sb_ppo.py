import os
from celeste_env import CelesteEnv
from config import Config
from sb_ppo.config_ppo import ConfigPPO
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.callbacks import BaseCallback
from sb_ppo.callback import CheckpointCallback


class ppo:
    def __init__(self, config_ppo, config_env: Config) -> None:
        self.config_ppo = config_ppo

        self.checkpoint_callback = CheckpointCallback(
            save_freq=self.config_ppo.save_freq,
            save_path=self.config_ppo.save_path,
        )

        # load
        # model = PPO.load(path=r".\logs\rl_model.zip", env=env) 
        
    def train(self,env,config,metrics):
        self.model = PPO("MultiInputPolicy", env, verbose=1, n_steps=self.config_ppo.n_steps, batch_size=self.config_ppo.batch_size)
        env_checker.check_env(env)
        # train
        self.model.learn(total_timesteps=self.config_ppo.timesteps, progress_bar=True, callback=self.checkpoint_callback)

        # test
        obs, _ = env.reset()
        while True:
            action, _state = self.model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
