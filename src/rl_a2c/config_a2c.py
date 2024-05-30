class ConfigA2C:
    def __init__(self):
        self.use_image_train = False
        self.only_image_actor = False
        self.discount_factor = 0.99
        self.batch_size = 64
        self.lr = 3e-4
        self.hidden_size = 128
        self.restore_networks = False
        self.file_save_network = "src/rl_a2c/network"
