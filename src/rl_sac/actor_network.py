
import torch
import torch.nn as nn
import torch.optim as optim

from rl_sac.config_sac import ConfigSac

class ActorNetwork(nn.Module):

    def __init__(self, state_size, action_size, size_image, config: ConfigSac, name="actor"):
        super(ActorNetwork, self).__init__()

        self.save_file = config.file_save + "/" + name + ".pt"

        self.state_size = state_size
        self.action_size = action_size
        self.size_image = size_image
        self.hidden_size_1 = config.hidden_size
        self.hidden_size_2 = config.hidden_size

        if self.size_image is not None:
            self.base_image = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.Flatten()
            )
             # Divide to times by 4 because maxpooling, multiply by 16 with 16 output filter
            size_output_image = int(self.size_image[1] * self.size_image[2] / 4 / 4 * 16)
        else:
            size_output_image = 0

        self.base = nn.Sequential(
            nn.Linear(size_output_image + state_size, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.hidden_size_2),
            nn.ReLU()
        )

        self.mu = nn.Linear(self.hidden_size_2, action_size)
        self.sigma = nn.Linear(self.hidden_size_2, action_size)

        self.noise_value = config.noise_value

        self.optimizer = optim.Adam(self.parameters(), lr=config.lr)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x, image):
        if self.size_image is not None:
            x_image = self.base_image(image)
            x = self.base(torch.cat([x, x_image], dim=1))
        else:
            x = self.base(x)

        mu = self.mu(x)
        sigma = torch.clamp(self.sigma(x), min=self.noise_value, max=1)
        return mu, sigma
    
    def save_model(self):
        torch.save(self.state_dict(), self.save_file)

    def load_model(self):
        self.load_state_dict(torch.load(self.save_file))
