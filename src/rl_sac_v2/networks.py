import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical
from rl_sac_v2.config_sac import ConfigSac

import numpy as np

class ActorNetwork(nn.Module):

    def __init__(self, state_size, action_size, action_discrete, size_image, histo_size, config: ConfigSac, name="actor"):
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
                nn.Conv2d((histo_size+1)*self.size_image[2], 64, kernel_size=2, padding=0),
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

        self.mu = nn.Linear(self.hidden_size_2, action_size)
        self.sigma = nn.Linear(self.hidden_size_2, action_size)
        self.categorical = nn.Linear(self.hidden_size_2, 15)

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

        # mu = self.mu(x)
        # sigma = torch.clamp(self.sigma(x), min=self.noise_value, max=1)
        cate = self.categorical(x)
        # a = cate[:3]
        # b = cate[3:6]
        # c = cate[6:8]
        # d = cate[8:11]
        # e = cate[11:]
        # a = self.softmax(a)
        # b = self.softmax(b)
        # c = self.softmax(c)
        # d = self.softmax(d)
        # e = self.softmax(e)
        # return a,b,c,d,e
        cate = cate.reshape((5,3))
        cate = self.softmax(cate)
        return cate


    def sample_normal(self, state, image=None, reparameterize=True): # TODO: adapt this to new MultiDiscrete space
        cate = self.forward(state, image)
        # print(cate)
        # probabilities = Normal(mu, sigma)
        probabilities = Categorical(cate)
        # print(probabilities)
        if reparameterize:
            actions = probabilities.sample()
        else:
            actions = probabilities.sample()
        # print(actions)
        # print(actions)
        # action = torch.tanh(actions).to(self.device)
        # print(action)
        # print(action)
        log_probs = probabilities.log_prob(actions)
        # log_probs -= torch.log(1-actions.pow(2)+self.noise_value)
        log_probs = log_probs.sum()

        return actions, log_probs

    def save_model(self):
        torch.save(self.state_dict(), self.save_file)

    def load_model(self):
        self.load_state_dict(torch.load(self.save_file))



class CriticNetwork(nn.Module):

    def __init__(self, state_size, action_size, size_image, histo_size, config: ConfigSac, name="critic"):
        super(CriticNetwork, self).__init__()

        self.save_file = config.file_save_network + "/" + name + ".pt"

        self.state_size = state_size
        self.action_size = action_size
        self.size_image = size_image
        self.hidden_size_1 = config.hidden_size
        self.hidden_size_2 = config.hidden_size




        if self.size_image is not None:
            self.base_image = nn.Sequential(
                nn.Conv2d((histo_size+1)*self.size_image[2], 64, kernel_size=3, padding=0),
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

    def forward(self, state, action, image):
        if self.size_image is None:
            return self.base(torch.cat([state, action], dim=1))
        else:
            flat_image = self.base_image(image)
            return self.base(torch.cat([state, action, flat_image], dim=1))

    def save_model(self):
        torch.save(self.state_dict(), self.save_file)

    def load_model(self):
        self.load_state_dict(torch.load(self.save_file))