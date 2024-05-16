import os
from celeste_env import CelesteEnv
from config import Config
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.callbacks import BaseCallback

class CheckpointCallback(BaseCallback):

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = os.path.join(save_path, "rl_model.zip")

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {self.save_path}")

        return True

config = Config()
env = CelesteEnv(config)
env_checker.check_env(env)

checkpoint_callback = CheckpointCallback(
  save_freq=1024,
  save_path="./logs/",
)

# load
# model = PPO.load(path=r".\logs\rl_model.zip", env=env) 

# init
model = PPO("MultiInputPolicy", env, verbose=1, n_steps=256, batch_size=64)

# train
model.learn(total_timesteps=50_000, progress_bar=True, callback=checkpoint_callback)

# test
obs, _ = env.reset()
while True:
    action, _state = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
