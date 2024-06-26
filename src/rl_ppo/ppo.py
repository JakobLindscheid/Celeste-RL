from rl_ppo.buffer import ReplayBuffer
from rl_ppo.networks import ActorNetwork, CriticNetwork
from rl_ppo.config_ppo import ConfigPPO
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

from config import Config

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

import os

class PPO(nn.Module):
    def __init__(self, config_ppo, config_env):
        super(PPO, self).__init__()
        self.config = config_ppo
        self.config_env = config_env
        self.size_histo = 0
        self.action_size = config_env.action_size.shape[0]
        self.action_size_discrete = config_env.action_size

        self.gamma = config_ppo.discount_factor
        self.clip_epsilon = config_ppo.clip_epsilon
       
        self.state_size = config_env.base_observation_size
        if config_env.give_goal_coords:
            self.state_size += 4
        if config_env.give_screen_value:
            self.state_size += 1  

        self.size_image = config_env.size_image if self.config.use_image_train else None
        self.use_image = config_env.use_image
        self.batch_size = config_ppo.batch_size

        self.actor = ActorNetwork(self.state_size, self.action_size, self.action_size_discrete, self.size_image, self.size_histo, self.config)
        self.critic = CriticNetwork(self.state_size, self.action_size, self.size_image, self.size_histo, self.config)

        self.policy_optimizer = optim.Adam(self.actor.parameters(), lr=config_ppo.lr)
        self.value_optimizer = optim.Adam(self.critic.parameters(), lr=config_ppo.lr)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.memory = ReplayBuffer(self.config.size_buffer, self.action_size, self.state_size, self.size_image, self.size_histo)

    def insert_data(self, state, new_state, image, new_image, actions_probs, reward, terminated, truncated):
        self.memory.insert_data(state, new_state, image, new_image, actions_probs, reward, terminated)

    def choose_action(self, state, image):
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device) 
        if self.use_image:
            image = torch.tensor(image, dtype=torch.float).to(self.actor.device)
        action, log_prob, entropy = self.actor.sample_action(state, image if self.use_image else None)
        action = torch.tensor(action).to(self.actor.device)
        log_prob = torch.tensor(log_prob).to(self.actor.device)
        return action.cpu().detach().numpy(), log_prob.cpu().detach().numpy()

    def compute_advantages(self, rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95):
        rewards = rewards.view(-1)
        values = values.view(-1)
        next_values = next_values.view(-1)
        dones = dones.view(-1)

        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages
    
    def update(self, state, actions, reward, next_state, terminated, image, next_image):
        #if self.memory.current_size < self.batch_size:
        #    return

        #if self.memory.index % self.config.frequency_training != 0:
        #    return
        
        #state, next_state, image, next_image, actions, reward, terminated = self.memory.sample_data(self.batch_size)

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        terminated = torch.tensor(terminated, dtype=torch.float).to(self.device)

        if self.use_image:
            image = torch.tensor(image, dtype=torch.float).to(self.device)
            next_image = torch.tensor(next_image, dtype=torch.float).to(self.device)
        else:
            image, next_image = None, None

        values = self.critic(state, image).squeeze()
        target_values = self.critic(next_state, next_image).squeeze()

        values = values.view(-1)
        target_values = target_values.view(-1)

        advantages = self.compute_advantages(reward, values, target_values, terminated)
        returns = (values + advantages).detach()
        
        _, log_probs, entropy = self.actor.sample_action(state, image)
        log_probs = torch.tensor(log_probs, dtype=torch.float, requires_grad=True).to(self.device)
        entropy = torch.tensor(entropy, dtype=torch.float, requires_grad=True).to(self.device)

        policy_loss = -(log_probs * advantages).mean() - self.entropy_weight * entropy.mean() 
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        value_loss = F.mse_loss(values, returns).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
       

    def train(self,env,config,metrics):
        # For every episode
        for learning_step in range(1, config.nb_learning_step + 1):

            # Reset nb terminated
            metrics.nb_terminated_train = 0
            iteration = 1
            if learning_step > 1 or not config.start_with_test:
                episode_train = 1
                while episode_train < config.nb_train_episode + 1:
                    # Reset the environnement
                    obs, _ = env.reset(test=False)
                    state = obs["info"]
                    image = obs["frame"]
                    while obs["fail"]:
                        print("failed to sync try respawn")
                        obs, _ = env.reset(test=False)
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
                        actions, _ = self.choose_action([state], [image])
                        # Step the environnement
                        obs, reward, terminated, truncated, _ = env.step(actions)
                        next_state = obs["info"]
                        next_image = obs["frame"]

                        if not truncated:
                            # Insert the data in the algorithm memory
                            #self.insert_data(state, next_state, image, next_image, actions, reward, terminated, truncated)

                            # Actualise state
                            state = next_state
                            image = next_image
                            ep_reward.append(reward)
                            iteration += 1
              
                    self.update(state, actions, reward, next_state, terminated, image, next_image)
                    self.memory.reset()

                    if ep_reward:
                        entropy = 0
                        metrics.print_train_step(learning_step, episode_train, ep_reward, entropy)
                        episode_train += 1


            for episode_test in range(1, config.nb_test_episode + 1):

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
                    actions, _ = self.choose_action([state], [image])

                    # Step the environnement
                    obs, reward, terminated, truncated, _ = env.step(actions.astype(int))
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
            if config.save_model:
                self.save_model()
            self.save_model(True)

            if config.restore:
                self.load_model()

    def save_model(self,recent=False):
        save_dir = os.path.dirname(self.config.file_save_network)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.actor.save_model(recent)
        self.critic.save_model(recent)
  
    def load_model(self,recent=False):
        self.actor.load_model(recent)
        self.critic.load_model(recent)

