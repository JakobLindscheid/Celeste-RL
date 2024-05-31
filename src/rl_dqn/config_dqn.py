"""Config of Multiple DQN file
"""

class ConfigDQN:
    """Class for the config of Multiple  DQN
    """

    def __init__(self) -> None:

        # Copy the targer network each *this value* of episodes
        self.nb_episode_copy_target = 200

        # Batch size when training
        self.batch_size = 128

        # Capacity of the memory
        self.size_buffer = 10_000

        # True if the networks are restored
        self.restore_networks = False

        # Gamma value
        self.discount_factor = 0.98

        # Hidden layers
        self.hidden_layers = 512

        # Epsilon configuration
        self.init_epsilon = 0.6
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.1

        # Learning rate configuration
        self.init_lr = 3e-4

        self.file_save = "src/rl_dqn/network"
