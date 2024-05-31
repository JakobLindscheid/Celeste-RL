
import numpy as np
import torch
import torch.nn as nn

from rl_dqn.config_dqn import ConfigDQN

class DQNet(nn.Module):
    def __init__(self, inputs, action_size, size_image, config: ConfigDQN, use_image=True) -> None:
        super(DQNet, self).__init__()

        self.use_image = use_image

        self.save_file = config.file_save + '/' + 'dqn.pt'

        self.hiddens = config.hidden_layers

        self.outfilter_size = 32

        if self.use_image:
            self.base_image = nn.Sequential(
                nn.Conv2d(size_image[0], 64, kernel_size=3, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, self.outfilter_size, kernel_size=3, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten()
            )
            # Divide and minus three times by 2 because maxpooling, multiply by 16 with 16 output filter
            size_output_image = int(self.outfilter_size * np.prod(np.trunc(np.trunc(np.trunc((size_image[1:3] - 2)/2-2)/2-2)/2)))
        else:
            size_output_image = 0

        self.base = nn.Sequential(
            nn.Linear(inputs + size_output_image, self.hiddens),
            nn.ReLU(),
            nn.Linear(self.hiddens, self.hiddens),
            nn.ReLU(),
            nn.Linear(self.hiddens, action_size)
        )

    def forward(self, x, image: torch.Tensor):
        if self.use_image:
            out = self.base(torch.cat((x, self.base_image(image)), 1))
        else:
            out = self.base(x)

        return out

    def save_model(self):
        torch.save(self.state_dict(), self.save_file)

    def load_model(self):
        self.load_state_dict(torch.load(self.save_file))