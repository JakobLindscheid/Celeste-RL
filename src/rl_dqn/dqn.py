"""Multiple DQN class file
"""

import random as r

import numpy as np
import torch
import torch.nn as nn

from config import Config
from rl_dqn.replay_buffer import ReplayBuffer
from rl_dqn.config_dqn import ConfigDQN
from rl_dqn.networks import DQNet


class DQN():
    """Class for Multiple DQN
    """
    def __init__(self, config_multi_qnetwork: ConfigDQN, config: Config):

        # Config of Multiple Qnetwork
        self.config = config_multi_qnetwork

        self.device = config.device

        self.action_mode = "Discrete"

        # Save file
        self.save_file = self.config.file_save + "/network.pt"

        self.use_image = config.use_image

        # Get the right model
        self.type_model = "SIMPLE_MLP" if not self.use_image else "CNN_MLP"

        # Get the size of the image (will not be used if simple MLP)
        if self.use_image:
            self.permutation = (2,0,1)
            self.size_image = np.array([config.size_image[x] for x in self.permutation])
        else:
            self.size_image = None

        self.size_histo = 0

        # Observation and action size
        self.state_size = config.base_observation_size
        if config.give_goal_coords:
            self.state_size += 4
        if config.give_screen_value:
            self.state_size += 1

        self.action_size = config.action_size.shape[0]
        self.action_shape = config.action_size

        # Memory of each step
        self.memory = ReplayBuffer(self.config.size_buffer, self.action_size, self.state_size, self.size_image, self.size_histo)

        # Current learning rate
        self.cur_lr = self.config.init_lr

        # Current epsilone
        self.epsilon = self.config.init_epsilon

        # count for updating target network
        self.count = 0

        # Create Qnetwork model
        self.q_network = DQNet(self.state_size, self.action_shape, self.size_image, config_multi_qnetwork, self.use_image)

        # Create the target network by copying the Qnetwork model
        self.target_network = DQNet(self.state_size, self.action_shape, self.size_image, config_multi_qnetwork, self.use_image)

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

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        state = state.unsqueeze(0)
        if self.use_image:
            image = torch.tensor(image, dtype=torch.float).permute(self.permutation).to(self.device)
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
        if self.use_image:
            image = torch.permute(torch.from_numpy(image), self.permutation).detach().numpy()
            new_image = torch.permute(torch.from_numpy(new_image), self.permutation).detach().numpy()
        self.memory.insert_data(state, new_state, image, new_image, actions_probs, reward, terminated)

    def calculate_loss(self):
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

        # Copy the target newtork if the episode is multiple of the coefficient
        if self.count % self.config.nb_episode_copy_target == 0:
            self.copy_target_network()
        self.count += 1

        # Decay epsilon
        self.epsilon_decay()

        # Decay learning rate
        # self.lr_decay(episode)

    def train(self,env,config,metrics):
        learning_step = 1
        # For every episode
        while learning_step < config.nb_learning_step + 1:

            # Reset nb terminated
            metrics.nb_terminated_train = 0

            if learning_step > 1 or not config.start_with_test:
                episode_train = 1
                while episode_train < config.nb_train_episode + 1:

                    # Reset the environnement
                    obs, _ = env.reset()
                    state = obs["info"]
                    image = obs["frame"]
                    while obs["fail"]:
                        print("failed to sync try respawn")
                        obs, _ = env.reset()
                        state = obs["info"]
                        image = obs["frame"]

                    ep_reward = list()

                    terminated = False
                    truncated = False

                    # For each step
                    image_steps = 0
                    while not terminated and not truncated:
                        # if image_steps %10==0:
                        #     screen = env.get_image_game(normalize=False)[0]
                        #     cv2.imwrite('screen.png', np.rollaxis(screen, 0, 3))
                        # Get the actions
                        # print(state[0][:11])
                        actions = self.choose_action(state, image)
                        # Step the environnement
                        obs, reward, terminated, truncated, _ = env.step(actions)
                        next_state = obs["info"]
                        next_image = obs["frame"]

                        if not truncated:
                            # Insert the data in the algorithm memory
                            self.insert_data(state, next_state, image, next_image, actions, reward, terminated, truncated)


                            # Actualise state
                            state = next_state
                            image = next_image

                            # Train the algorithm
                            entropy=self.calculate_loss()

                            ep_reward.append(reward)

                    if ep_reward:
                        metrics.print_train_step(learning_step, episode_train, ep_reward, entropy)
                        episode_train += 1
                        learning_step += 1

            episode_test = 1
            while episode_test < config.nb_test_episode + 1:

                terminated = False
                truncated = False

                # Init the episode reward at 0
                reward_ep = list()

                # Reset the environnement
                obs, _ = env.reset(test=True)
                state = obs["info"]
                image = obs["frame"]
                while obs["fail"]:
                    print("failed to sync try respawn")
                    obs, _ = env.reset(test=True)
                    state = obs["info"]
                    image = obs["frame"]

                # For each step
                while not terminated and not truncated:

                    # Get the actions
                    actions = self.choose_action(state, image)

                    # Step the environnement
                    obs, reward, terminated, truncated, _ = env.step(actions)
                    next_state = obs["info"]
                    next_image = obs["frame"]

                    # Actualise state
                    state = next_state
                    image = next_image

                    # Add the reward to the episode reward
                    reward_ep.append(reward)

                if not truncated:
                    # Insert the metrics
                    save_model, save_video, restore, next_screen = metrics.insert_metrics(learning_step, reward_ep, episode_test, env.max_steps, env.game_step)

                    # Print the information about the episode
                    metrics.print_test_step(learning_step, episode_test)


                    if save_video:
                        print("save")
                        env.save_video()

                    if next_screen and config.max_screen_value_test < 7:
                        config.max_screen_value_test += 1
                        config.screen_used.append(config.max_screen_value_test)

                        config.prob_screen_used = np.ones(config.max_screen_value_test+1)
                        config.prob_screen_used[0] = config.max_screen_value_test
                        config.prob_screen_used[config.max_screen_value_test] = config.max_screen_value_test+1
                        config.prob_screen_used = config.prob_screen_used / np.sum(config.prob_screen_used)
                    episode_test += 1
                # else:
                #     episode_test -= 1

            # Save the model (will be True only if new max reward)
            if save_model:
                self.save_model()
            self.save_model()

            if restore:
                self.load_model()

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

