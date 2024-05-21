"""Config of SAC file
"""

class ConfigPPO:
    """Class for the config of SAC
    """

    def __init__(self) -> None:

        self.use_image_train = True
        self.save_path = "./src/sb_ppo/logs/"
        self.save_freq = 512
        self.timesteps = 50_000
        self.batch_size = 64
        self.n_steps = 256
