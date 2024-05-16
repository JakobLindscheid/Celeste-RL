"""Multiple DQN class file
"""

import random as r
import numpy as np
import torch.nn as nn
import torch

from config_multi_qnetworks import ConfigMultiQnetworks
from config import Config

class DQN(nn.Module):
    def __init__(self, inputs, hiddens, outputs, histo_size, size_image, image=False) -> None:
        super(DQN, self).__init__()
        self.image = image
        if self.image:
            self.base_image = nn.Sequential(
                nn.Conv2d((histo_size+1)*size_image[0], 64, kernel_size=2, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 32, kernel_size=3, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten()
            )
        else:
            self.input = nn.Linear(inputs, hiddens)
            self.hidden = nn.Linear(hiddens, hiddens)
            self.output = nn.Linear(hiddens, outputs)

            self.relu = nn.ReLU()

    def forward(self, x):
        if self.image:
            out = self.base_image(x)
        else:
            out = self.input(x)
            out = self.relu(out)
            out = self.hidden(out)
            out = self.relu(out)
            out = self.output(out)
        return out

class MultiQNetwork():
    """Class for Multiple DQN
    """
    def __init__(self, config_multi_qnetwork: ConfigMultiQnetworks, config: Config):

        # Config of Multiple Qnetwork
        self.config = config_multi_qnetwork

        self.action_mode = "Discrete"

        # Save file
        self.save_file = self.config.file_save + "/network.pt"

        # Get the right model
        self.type_model = "SIMPLE_MLP" if not config.use_image else "CNN_MLP"

        # Get the size of the image (will not be used if simple MLP)
        self.size_image = config.size_image

        # Memory of each step
        self.memory = []

        # Current learning rate
        self.cur_lr = self.config.init_lr

        # Current epsilone
        self.epsilon = self.config.init_epsilon

        # Observation and action size
        self.state_size = config.observation_size
        self.action_size = config.action_size

        # Create Qnetwork model
        self.q_network = DQN(self.state_size, 128, self.action_size, config.histo_image, config.size_image)

        # Create the target network by copying the Qnetwork model
        self.target_network = DQN(self.state_size, 128, self.action_size, config.histo_image, config.size_image)

        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), self.cur_lr)

        # Restore the network with saved one
        if self.config.restore_networks:
            self.restore()

    def choose_action(self, state: np.ndarray, available_actions:list=None):
        """Choose the different actions to take

        Args:
            state (np.array): Current state of the environnement
            available_actions (list): for each actions, 1 if the action is possible, else 0, if None, all actions are available

        Returns:
            np.array: array of action taken
        """
        # Create the action array
        actions = np.zeros(self.action_size.shape, dtype=np.int32)

        # Get the action tensor
        list_action_tensor = self.q_network(state)

        # For each action
        for index, action_tensor in enumerate(list_action_tensor):

            # If random < epsilon (epsilon have a minimal value)
            if r.random() < max(self.epsilon, self.config.min_epsilon):

                # Get random action probability
                actions_probs = np.random.rand(self.action_size[index])
            else:

                # Get QDN action probability
                actions_probs = action_tensor.numpy()[0]

            # Multiply action probability with available actions to avoid unavailable actions
            if available_actions is not None:
                actions_probs = actions_probs * available_actions[index]

            # Get the action
            actions[index] = np.argmax(actions_probs)

            # If action[0] != 0, Madeline is dashing in a direction, so the directional action is desactivate
            #if index == 1 and actions[0] != 0:
            #    actions[index] = 0


        return actions, None, None

    def insert_data(self, data: tuple):
        """Insert the data in the memory

        Args:
            data (tuple): different data put in the memory
        """
        # Insert the data
        self.memory.append(data)

        # If memory is full, delete the oldest one
        if len(self.memory) > self.config.memory_capacity:
            self.memory.pop(0)

    def train(self, episode: int):
        """Train the algorithm

        Args:
            episode (int): Current episode
        """

        # Do not train if not enough data, not the right episode or not enough episode to start training
        if len(self.memory) < self.config.batch_size or episode % self.config.nb_episode_learn != 0 or episode < self.config.start_learning:
            return

        # Get the data
        states, actions, rewards, next_states, dones = zip(*r.sample(self.memory, self.config.batch_size))

        # Those line switch the vector of shapes (because of the multiple outputs)

        states = [np.concatenate([states[i][j] for i in range(len(states))], axis=0) for j in range(len(states[0]))]
        next_states = [np.concatenate([next_states[i][j] for i in range(len(next_states))], axis=0) for j in range(len(next_states[0]))]
        states = np.array(states).reshape(self.config.batch_size, -1)
        next_states = np.array(next_states).reshape(self.config.batch_size, -1)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Get the q values and next q values
        q_values = [tensor.numpy() for tensor in self.q_network(states)]
        next_q_values = [tensor.numpy() for tensor in self.target_network(next_states)]

        vals = self.q_network(states)


        # For each q values
        for index, q_value in enumerate(q_values):

            # Apply the bellman equation of the next q value
            target_q_value = (next_q_values[index].max(axis=1) * self.config.discount_factor) + rewards

            # Only get the reward if the step has done = True
            target_q_value[dones] = rewards[dones]

            # Apply the target value on the q value
            q_value[np.arange(self.config.batch_size), actions[:, index]] = target_q_value

        # Train the model
        self.optimizer.zero_grad()
        self.optimizer.step()
        # self.q_network.fit(states, q_values, epochs=3, verbose=0, batch_size=self.config.mini_batch_size)

        # Copy the target newtork if the episode is multiple of the coefficient
        if episode % self.config.nb_episode_copy_target == 0:
            self.copy_target_network()

        # Decay epsilon
        self.epsilon_decay()

        # Decay learning rate
        self.lr_decay(episode)


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



    # def restore(self):
    #     """Restore the networks based on saved ones
    #     """
    #     self.q_network = tf.keras.models.load_model(self.save_file)
    #     self.q_network.compile(loss='mse', optimizer= tf.optimizers.Adam(learning_rate=self.cur_lr))
    #     self.copy_target_network()



    # def save(self):
    #     """Save the networks
    #     """
    #     self.q_network.save(self.save_file)
