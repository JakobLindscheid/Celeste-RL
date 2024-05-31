class ConfigPPO:
    """Class for the config of PPO
    """

    def __init__(self) -> None:

        self.use_image_train = True
        self.only_image_actor = False

        self.discount_factor = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.epoch = 1
        self.frequency_training = 1
        self.clip_epsilon = 1e-5

        self.lr = 3e-4
        self.size_buffer = 10_000

        self.init_entropy = 2
        self.restore_networks = True
        self.file_save_network = "src/rl_ppo/network"
        self.file_save_memory = "src/rl_ppo/memory"

        self.noise_value = 1e-6
        self.hidden_size = 512

        self.save_model = True
        self.restore = False
        self.file_save = "src/rl_ppo/network"

