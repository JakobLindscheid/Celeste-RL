import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical
from rl_td3.config_td3 import ConfigTD3

import numpy as np

class ActorNetwork(nn.Module):

    def __init__(self, state_size, action_size, action_discrete, size_image, config: ConfigTD3, name="actor"):
        super(ActorNetwork, self).__init__()

        self.save_file =  config.file_save_network + "/" + name + ".pt"

        self.use_state = not config.only_image_actor
        self.state_size = state_size if self.use_state else 0
        self.action_size = action_size
        self.action_discrete = action_discrete
        self.size_image = size_image
        self.hidden_size_1 = config.hidden_size
        self.hidden_size_2 = config.hidden_size

        if self.size_image is not None:
            self.base_image = nn.Sequential(
                nn.Conv2d(self.size_image[2], 64, kernel_size=2, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 32, kernel_size=3, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten()
            )
             # Divide and minus three times by 2 because maxpooling, multiply by 16 with 16 output filter
            size_output_image = int(32 * np.prod(np.trunc(np.trunc(np.trunc((self.size_image[0:2] - 2)/2-2)/2-2)/2)))
        else:
            size_output_image = 0

        self.base = nn.Sequential(
            nn.Linear(size_output_image + self.state_size, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.hidden_size_2),
            nn.ReLU()
        )

        self.last = nn.Linear(self.hidden_size_2, action_size)

        self.noise_value = config.noise_value

        self.optimizer = optim.Adam(self.parameters(), lr=config.lr)
        self.softmax = nn.Softmax(dim=-1)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x, image):
        image = image.transpose(0, 2).transpose(1, 2)
        if self.size_image is not None:
            x_image = self.base_image(image)
            x = self.base(torch.cat([x, x_image.flatten()])) if self.use_state else self.base(x_image)
        else:
            x = self.base(x)

        x = torch.tanh(self.last(x))

        return x


    def sample_normal(self, state, image=None):
        actions = self.forward(state, image)
        return actions

    def save_model(self,recent=False):
        if recent:
            torch.save(self.state_dict(), self.save_file[:-3]+"_recent.pt")
        else:
            torch.save(self.state_dict(), self.save_file)

    def load_model(self,recent=False):
        if recent:
            self.load_state_dict(torch.load(self.save_file[:-3]+"_recent.pt"))
        else:
            self.load_state_dict(torch.load(self.save_file))



class CriticNetwork(nn.Module):

    def __init__(self, state_size, action_size, size_image, config: ConfigTD3, name="critic"):
        super(CriticNetwork, self).__init__()

        self.save_file = config.file_save_network + "/" + name + ".pt"

        self.state_size = state_size
        self.action_size = action_size
        self.size_image = size_image
        self.hidden_size_1 = config.hidden_size
        self.hidden_size_2 = config.hidden_size




        if self.size_image is not None:
            self.base_image = nn.Sequential(
                nn.Conv2d(self.size_image[2], 64, kernel_size=3, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 32, kernel_size=3, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten()
            )
             # Divide and minus three times by 2 because maxpooling, multiply by 16 with 16 output filter
            size_output_image = int(32 * np.prod(np.trunc(np.trunc(np.trunc((self.size_image[0:2] - 2)/2-2)/2-2)/2)))
        else:
            size_output_image = 0

        self.base = nn.Sequential(
            nn.Linear(size_output_image+state_size+action_size, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.hidden_size_2),
            nn.ReLU(),
            nn.Linear(self.hidden_size_2, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=config.lr)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, image, action):
        if self.size_image is None:
            return self.base(torch.cat([state, action], dim=1))
        else:
            flat_image = self.base_image(image)
            return self.base(torch.cat([state, flat_image,action], dim=1))

    def save_model(self,recent=False):
        if recent:
            torch.save(self.state_dict(), self.save_file[:-3]+"_recent.pt")
        else:
            torch.save(self.state_dict(), self.save_file)

    def load_model(self,recent=False):
        if recent:
            self.load_state_dict(torch.load(self.save_file[:-3]+"_recent.pt"))
        else:
            self.load_state_dict(torch.load(self.save_file))