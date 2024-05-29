from rl_ppo.buffer import ReplayBuffer
from rl_ppo.networks import ActorNetwork, CriticNetwork
from rl_ppo.config_ppo import ConfigPPO

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

    def compute_advantages(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95):
        advantages = torch.zeros_like(rewards)
        gae = 0
    
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
    
        return advantages
    
    def update(self, state, actions, reward, next_state, terminated, image, next_image):
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
        with torch.no_grad():
            target_values = reward + (1 - terminated) * self.gamma * self.critic(next_state, next_image).squeeze()

        #advantages = self.compute_advantages(reward, values, target_values, terminated)
        advantages = (target_values - values).detach()
        returns = (torch.FloatTensor(advantages) + values).detach
        
        for _ in range(10):
            new_log_probs = []
            entropies = []
            for s, a in zip(state, actions):
                print("state ", s)
                s = s.unsqueeze(0)
                print("state ", s)
                print("actions ", a)
                _, log_probs, entropy = self.actor.sample_action(s, image)
                new_log_probs.append(log_probs)
                entropies.append(entropy)
            new_log_probs = torch.cat(new_log_probs)
            entropies = torch.cat(entropies)
            
            ratios = torch.exp(new_log_probs - log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropies.mean()
            value_loss = (returns - values).pow(2).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
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
                            self.insert_data(state, next_state, image, next_image, actions, reward, terminated, truncated)

                             # Train the algorithm
                            self.update(state, actions, reward, next_state, terminated, image, next_image)

                            # Actualise state
                            state = next_state
                            image = next_image
                            ep_reward.append(reward)
                            iteration += 1

                    if ep_reward:
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
                    actions = self.choose_action([state], [image])

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
            if save_model:
                self.save_model()
            self.save_model(True)

            if restore:
                self.load_model()

    def save_model(self,recent=False):
        self.actor_critic.save_model(recent)
        self.buffer.save_model(recent)
        if recent:
            torch.save(self.log_entropy_coef, self.config.file_save_network + "/entropy_recent.pt")
        else:
            torch.save(self.log_entropy_coef, self.config.file_save_network + "/entropy.pt")

    def load_model(self,recent=False):
        self.actor_critic.load_model(recent)
        self.buffer.load_model(recent)
        if recent:
            self.log_entropy_coef = torch.load(self.config.file_save_network + "/entropy_recent.pt")
        else:
            self.log_entropy_coef = torch.load(self.config.file_save_network + "/entropy.pt")


    def get_expected_returns(self):

        t_rewards = torch.tensor(self.buffer.rewards, device=self.device).flip(0)
        t_dones = torch.tensor(self.buffer.dones, device=self.device).flip(0)
        returns = torch.zeros(t_rewards.size(), device=self.device)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        discounted_sum = 0
        for index in range(t_rewards.shape[0]):
            if t_dones[index]:
                discounted_sum = 0
            reward = t_rewards[index]
            discounted_sum = reward + self.discount_factor * discounted_sum
            returns[index] = discounted_sum
        returns = returns.flip(0)

        if self.standardize:
            returns = ((returns - torch.mean(returns)) / 
                    (torch.std(returns) + 10e-7))

        return returns

