
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from rl_sac.config_sac import ConfigSac

class ValueNetwork(nn.Module):

    def __init__(self, state_size, size_image, histo_size, config: ConfigSac, name="value"):
        super(ValueNetwork, self).__init__()

        self.save_file = config.file_save_network + "/" + name + ".pt"
        self.state_size = state_size
        self.size_image = size_image
        self.hidden_size_1 = config.hidden_size
        self.hidden_size_2 = config.hidden_size



        if self.size_image is not None:
            self.base_image = nn.Sequential(
                nn.Conv2d((histo_size+1)*self.size_image[0], 64, kernel_size=3, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 32, kernel_size=3, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten()
            )
             # Divide and minus three times by 2 because maxpooling, multiply by 16 with 16 output filter
            size_output_image = int(32 * np.prod(np.trunc(np.trunc(np.trunc((self.size_image[1:3] - 2)/2-2)/2-2)/2)))
        else:
            size_output_image = 0

        self.base = nn.Sequential(
            nn.Linear(size_output_image + state_size, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.hidden_size_2),
            nn.ReLU(),
            nn.Linear(self.hidden_size_2, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=config.lr)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, image):
        if self.size_image is None:
            return self.base(state)
        else:
            x_image = self.base_image(image)
            return self.base(torch.cat([state, x_image], dim=1))

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


