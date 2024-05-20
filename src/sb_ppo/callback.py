
import os
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