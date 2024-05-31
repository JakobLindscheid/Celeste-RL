import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from rl_a2c.config_a2c import ConfigA2C
import numpy as np
import random
class A2CActorNetwork(nn.Module):

    def __init__(self, state_size, action_size, action_discrete, size_image, config: ConfigA2C, name="actor"):
        super(A2CActorNetwork, self).__init__()

        self.save_file = config.file_save_network + "/" + name + ".pt"

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
            h, w = size_image[0], size_image[1]

            def conv2d_size_out(size, kernel_size=3, stride=1, padding=0):
                return (size - (kernel_size - 1) - 1 + 2 * padding) // stride + 1

            # Calculate the output dimensions after each layer
            h = conv2d_size_out(h, kernel_size=2)
            h = h // 2
            h = conv2d_size_out(h, kernel_size=3)
            h = h // 2
            h = conv2d_size_out(h, kernel_size=3)
            h = h // 2

            w = conv2d_size_out(w, kernel_size=2)
            w = w // 2
            w = conv2d_size_out(w, kernel_size=3)
            w = w // 2
            w = conv2d_size_out(w, kernel_size=3)
            w = w // 2

            size_output_image = 32 * h * w
            #actual_flattened_size = 760  
            #size_output_image = actual_flattened_size

        else:
            size_output_image = 0

        self.base = nn.Sequential(
            nn.Linear(size_output_image + self.state_size, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.hidden_size_2),
            nn.ReLU()
        )

        # Define separate linear layers for each action component
        self.horizontal_movement = nn.Linear(self.hidden_size_2, 3)
        self.vertical_movement = nn.Linear(self.hidden_size_2, 3)
        self.dash = nn.Linear(self.hidden_size_2, 2)
        self.jump = nn.Linear(self.hidden_size_2, 3)
        self.grab = nn.Linear(self.hidden_size_2, 2)

        self.optimizer = optim.Adam(self.parameters(), lr=config.lr)
        self.softmax = nn.Softmax(dim=-1)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x, image):
        if self.size_image is not None:
            image = image.transpose(-3, -1).transpose(-2, -1)
            if image.dim() == 3:
                image = image.unsqueeze(0)
            x_image = self.base_image(image)
            #print(f"x_image shape after conv layers: {x_image.shape}")
            if self.use_state:
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                x = torch.cat([x, x_image], dim=1)
                #print(f"Concatenated x shape: {x.shape}")
            else:
                x = x_image
        else:
            if x.dim() == 1:
                x = x.unsqueeze(0)
        #print(f"x shape before base layers: {x.shape}")
        x = self.base(x)
        #print(f"x shape after base layers: {x.shape}")
        
        horizontal_probs = self.softmax(self.horizontal_movement(x))
        vertical_probs = self.softmax(self.vertical_movement(x))
        dash_probs = self.softmax(self.dash(x))
        jump_probs = self.softmax(self.jump(x))
        grab_probs = self.softmax(self.grab(x))

        action_probs = (horizontal_probs, vertical_probs, dash_probs, jump_probs, grab_probs)
        return action_probs

    def sample_action(self, state, image=None):
        action_probs = self.forward(state, image)
        actions = []
        log_probs = []
        entropies = []

        N = [3,3,2,3,2]
        for i,probs in enumerate(action_probs):
            # distribution = Categorical(probs)
            # action = distribution.sample()
            # log_prob = distribution.log_prob(action)
            # entropy = distribution.entropy()

            action = random.choices(range(N[i]), probs[0])[0]
            log_prob = torch.log(probs[0][action])
            entropy = torch.sum(probs[0]*torch.log(probs[0]))
            # actions.append(action.item())
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
        
        return actions, log_probs, entropies

    def save_model(self, recent=False):
        if recent:
            torch.save(self.state_dict(), self.save_file[:-3] + "_recent.pt")
        else:
            torch.save(self.state_dict(), self.save_file)

    def load_model(self, recent=False):
        if recent:
            self.load_state_dict(torch.load(self.save_file[:-3] + "_recent.pt"))
        else:
            self.load_state_dict(torch.load(self.save_file))

class A2CCriticNetwork(nn.Module):

    def __init__(self, state_size, size_image, config: ConfigA2C, name="critic"):
        super(A2CCriticNetwork, self).__init__()

        self.save_file = config.file_save_network + "/" + name + ".pt"

        self.state_size = state_size
        self.size_image = size_image
        self.hidden_size_1 = config.hidden_size
        self.hidden_size_2 = config.hidden_size
        self.use_state = not config.only_image_actor
        self.state_size = state_size if self.use_state else 0

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

            def conv2d_size_out(size, kernel_size=3, stride=1, padding=0):
                return (size - (kernel_size - 1) - 1 + 2 * padding) // stride + 1

            h, w = size_image[0], size_image[1]
            h = conv2d_size_out(h, kernel_size=3)
            h = h // 2
            h = conv2d_size_out(h, kernel_size=3)
            h = h // 2
            h = conv2d_size_out(h, kernel_size=3)
            h = h // 2

            w = conv2d_size_out(w, kernel_size=3)
            w = w // 2
            w = conv2d_size_out(w, kernel_size=3)
            w = w // 2
            w = conv2d_size_out(w, kernel_size=3)
            w = w // 2

            size_output_image = 32 * h * w
            #actual_flattened_size = 760  
            #size_output_image = actual_flattened_size

        else:
            size_output_image = 0

        self.base = nn.Sequential(
            nn.Linear(size_output_image + self.state_size, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.hidden_size_2),
            nn.ReLU(),
            nn.Linear(self.hidden_size_2, 1)  # Output a single value representing V(s)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=config.lr)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x, image):
        if self.size_image is not None:
            image = image.transpose(-3, -1).transpose(-2, -1)
            if image.dim() == 3:
                image = image.unsqueeze(0)
            x_image = self.base_image(image)
            if self.use_state:
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                x = torch.cat([x, x_image], dim=1)
            else:
                x = x_image
        else:
            if x.dim() == 1:
                x = x.unsqueeze(0)
        x = self.base(x)
        value = x.squeeze()
        return value

    def save_model(self, recent=False):
        if recent:
            torch.save(self.state_dict(), self.save_file[:-3] + "_recent.pt")
        else:
            torch.save(self.state_dict(), self.save_file)

    def load_model(self, recent=False):
        if recent:
            self.load_state_dict(torch.load(self.save_file[:-3] + "_recent.pt"))
        else:
            self.load_state_dict(torch.load(self.save_file))
