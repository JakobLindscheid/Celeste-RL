"""Multiple DQN class file
"""

import random as r

import numpy as np
import torch
import torch.nn as nn

from config import Config
from rl_dqn.config_multi_qnetworks import ConfigMultiQnetworks
from rl_dqn.replay_buffer import ReplayBuffer


class Value(nn.Module):
    def __init__(self, inputs, action_shape) -> None:
        super(Value, self).__init__()

        self.outputs = []

        self.action_size = len(action_shape)

        for size in action_shape:
            self.outputs.append(nn.Linear(inputs, size))

    def forward(self, x):
        out = []
        for index in range(self.action_size):
            out.append(self.outputs[index](x))

        return out

class Print(nn.Module):
    def __init__(self) -> None:
        super(Print, self).__init__()


    def forward(self, x):
        print(x.size())
        return x

class DQN(nn.Module):
    def __init__(self, inputs, hiddens, action_shape, histo_size, size_image, config: ConfigMultiQnetworks, image=True) -> None:
        super(DQN, self).__init__()

        self.image = image

        self.save_file = config.file_save + '/' + 'dqn.pt'

        self.action_size = action_shape.shape[0]

        self.base_image = nn.Sequential(
            nn.Conv2d((histo_size+1)*size_image[0], 64, kernel_size=3, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 1, kernel_size=3, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(20*38, hiddens)
        )

        self.base_state = nn.Sequential(
            nn.Linear(inputs, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, hiddens),
            nn.ReLU(),
        )

        self.base = Value(hiddens+hiddens, action_shape)

    def forward(self, x, image: torch.Tensor):
        out = self.base_state(x)

        if image is not None:
            out = torch.cat((out, self.base_image(image).squeeze(1)), 1)

        out = self.base(out)

        return out

    def save_model(self):
        torch.save(self.state_dict(), self.save_file)

    def load_model(self):
        self.load_state_dict(torch.load(self.save_file))

class MultiQNetwork():
    """Class for Multiple DQN
    """
    def __init__(self, config_multi_qnetwork: ConfigMultiQnetworks, config: Config):

        # Config of Multiple Qnetwork
        self.config = config_multi_qnetwork

        self.device = config.device

        self.action_mode = "Discrete"

        # Save file
        self.save_file = self.config.file_save + "/network.pt"

        # Get the right model
        self.type_model = "SIMPLE_MLP" if not config.use_image else "CNN_MLP"

        # Get the size of the image (will not be used if simple MLP)
        self.permutation = (2,0,1)
        self.size_image = [config.size_image[x] for x in self.permutation]

        self.size_histo = 0

        # Observation and action size
        self.state_size = config.base_observation_size
        if config.give_goal_coords:
            self.state_size += 4
        if config.give_screen_value:
            self.state_size += 1

        self.action_size = config.action_size.shape[0]
        self.action_shape = config.action_size

        self.use_image = config.use_image

        # Memory of each step
        self.memory = ReplayBuffer(self.config.size_buffer, self.action_size, self.state_size, self.size_image, self.size_histo)

        # Current learning rate
        self.cur_lr = self.config.init_lr

        # Current epsilone
        self.epsilon = self.config.init_epsilon

        # count for updating target network
        self.count = 0

        # Create Qnetwork model
        self.q_network = DQN(self.state_size, self.config.hidden_layers, self.action_shape, self.size_histo, self.size_image, config_multi_qnetwork, self.use_image)

        # Create the target network by copying the Qnetwork model
        self.target_network = DQN(self.state_size, self.config.hidden_layers, self.action_shape, self.size_histo, self.size_image, config_multi_qnetwork, self.use_image)

        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), self.cur_lr)

        # Restore the network with saved one
        if self.config.restore_networks:
            self.restore()

    def choose_action(self, state, image):
        """Choose the different actions to take

        Args:
            state (np.array): Current state of the environnement
            image (np.array): Current screenshot of the environment

        Returns:
            np.array: array of action taken
        """
        # Create the action array
        actions = np.zeros((self.action_size,), dtype=np.int32)

        image = torch.tensor(image, dtype=torch.float).permute(self.permutation).to(self.device)
        state = torch.tensor(state, dtype=torch.float).to(self.device)

        # create batch of 1
        state = state.unsqueeze(0)
        image = image.unsqueeze(0)

        # Get the action tensor
        with torch.no_grad():
            list_action_tensor = self.q_network(state, image)

        for index, size in enumerate(self.action_shape):
            # If random < epsilon (epsilon have a minimal value)
            if r.random() < max(self.epsilon, self.config.min_epsilon):
                # Get random action probability
                actions[index] = np.random.randint(0,size)
            else:
                actions[index] = np.argmax(list_action_tensor[index].detach().numpy())

        return actions

    def insert_data(self, state, new_state, image, new_image, actions_probs, reward, terminated, truncated):
        """Insert the data in the memory

        Args:
            data: different data put in the memory
        """
        image = torch.permute(torch.from_numpy(image), self.permutation).detach().numpy()
        new_image = torch.permute(torch.from_numpy(new_image), self.permutation).detach().numpy()
        self.memory.insert_data(state, new_state, image, new_image, actions_probs, reward, terminated)

    def train(self):
        """Train the algorithm

        Args:
            episode (int): Current episode
        """

        # Do not train if not enough data, not the right episode or not enough episode to start training
        if self.memory.current_size < self.config.batch_size:
            return

        # Get the data
        states, next_states, image, new_image, actions, rewards, dones = self.memory.sample_data(self.config.batch_size)

        q_values = self.q_network(states, image)

        with torch.no_grad():
            next_q_values = self.target_network(next_states, new_image)

        loss = 0
        # For each q values
        for index, q_value in enumerate(q_values):
            # Get the q_values corresponding to the actions
            actual_q_value = q_value.gather(1, actions[:, index].view(-1,1).type(torch.int64))

            # Get maximal q_values
            max_next_q_values = torch.max(next_q_values[index], dim=1)[0].view(-1,1)

            # Apply the bellman equation of the next q value
            target_q_value = (max_next_q_values * (1-dones) * self.config.discount_factor) + rewards

            loss += nn.functional.huber_loss(actual_q_value, target_q_value)

        loss.backward()

        # Train the model
        self.optimizer.step()
        self.optimizer.zero_grad()
        # self.q_network.fit(states, q_values, epochs=3, verbose=0, batch_size=self.config.mini_batch_size)

        # Copy the target newtork if the episode is multiple of the coefficient
        if self.count % self.config.nb_episode_copy_target == 0:
            self.copy_target_network()
        self.count += 1

        # Decay epsilon
        self.epsilon_decay()

        # Decay learning rate
        # self.lr_decay(episode)


    def copy_target_network(self):
        """Set the weight on the target network based on the current q network
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def epsilon_decay(self):
        """Decay espilon
        """
        self.epsilon = self.config.epsilon_decay*self.epsilon

    # def lr_decay(self, cur_episode: int):
    #     """Decay the learning rate

    #     Args:
    #         cur_episode (int): current episode
    #     """
    #     if cur_episode % self.config.nb_episode_lr_decay == 0:
    #         self.cur_lr *= self.config.lr_decay
    #         tf.keras.backend.set_value(self.q_network.optimizer.learning_rate, max(self.cur_lr, self.config.min_lr))



    def save_model(self):
        self.q_network.save_model()

    def load_model(self):
        self.q_network.load_model()
        self.target_network.load_state_dict(self.q_network.state_dict())

