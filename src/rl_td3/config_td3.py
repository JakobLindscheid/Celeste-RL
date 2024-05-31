"""Config of SAC file
"""

class ConfigTD3:
    """Class for the config of SAC
    """

    def __init__(self) -> None:



        self.use_image_train = False
        self.only_image_actor = False

        self.discount_factor = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.epoch = 1
        self.frequency_training = 1
        self.policy_frequency = 2

        self.lr = 3e-4
        self.hidden_size = 1024
        self.size_buffer = 10_000

        self.policy_noise = 0.2
        self.explore_noise = [0.6,0.6,0.2,0.3,0.2, 0.2,0.2,0.2,0.2,0.2]
        self.clipped_noise = 0.4
        

        self.init_entropy = 2
        self.restore_networks = False
        self.file_save_network = "src/rl_td3/network"
        self.file_save_memory = "src/rl_td3/memory"

        self.noise_value = 1e-6
