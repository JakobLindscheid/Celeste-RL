
from rl_sac_v2.networks import ActorNetwork, CriticNetwork
from rl_sac_v2.replay_buffer import ReplayBuffer
from rl_sac_v2.config_sac import ConfigSac
from config import Config

import numpy as np
import torch
from torch.nn import functional as F
import torch.optim as optim

class ActorCritic():

    def __init__(self, config_sac: ConfigSac, config_env: Config) -> None:



        self.action_mode = "Continuous"

        self.config = config_sac
        self.config_env = config_env
        self.size_histo = 0
        self.action_size = config_env.action_size.shape[0]
        self.action_size_discrete = config_env.action_size
        
        self.state_size = config_env.base_observation_size
        if config_env.give_goal_coords:
            self.state_size += 4
        if config_env.give_screen_value:
            self.state_size += 1        

        self.size_image = config_env.size_image if self.config.use_image_train else None
        self.use_image = config_env.use_image

        self.gamma = self.config.discount_factor
        self.tau = self.config.tau
        self.batch_size = self.config.batch_size


        self.actor = ActorNetwork(self.state_size, self.action_size, self.action_size_discrete,self.size_image, self.size_histo, self.config)

        self.critic_1 = CriticNetwork(self.state_size, self.action_size, self.size_image, self.size_histo, self.config, name="critic_1")
        self.critic_2 = CriticNetwork(self.state_size, self.action_size, self.size_image, self.size_histo, self.config, name="critic_2")
        self.target_critic_1 = CriticNetwork(self.state_size, self.action_size, self.size_image, self.size_histo, self.config, name="target_critic_1")
        self.target_critic_2 = CriticNetwork(self.state_size, self.action_size, self.size_image, self.size_histo, self.config, name="target_critic_2")

        self.memory = ReplayBuffer(self.config.size_buffer, self.action_size, self.state_size, self.size_image, self.size_histo)


        self.target_entropy = -1 * self.action_size * torch.ones(1, device=self.actor.device)
        init_entropy_coef = self.config.init_entropy
        self.log_entropy_coef = torch.log(init_entropy_coef*torch.ones(1, device=self.actor.device)).requires_grad_(True)
        self.entropy_coef_optimizer = optim.Adam([self.log_entropy_coef], lr=self.config.lr)

        if self.config.restore_networks:
            self.load_model(True)


        self.update_network_parameters(init=True)

    def insert_data(self, state, new_state, image, new_image, actions_probs, reward, terminated, truncated):
        self.memory.insert_data(state, new_state, image, new_image, actions_probs, reward, terminated)

    def choose_action(self, state, image):
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)        
        if self.use_image:
            image = torch.tensor(image, dtype=torch.float).to(self.actor.device)
        action, _ = self.actor.sample_normal(state, image)
        # discretize the actions
        action = action.cpu().detach().numpy()
        # action = np.trunc((action.reshape(-1) + 1) * self.config_env.action_size / 2)
        return action

    def update_network_parameters(self, init=False):
        tau = 1 if init else self.tau

        target_critic_1_params = dict(self.target_critic_1.named_parameters())
        critic_1_params = dict(self.critic_1.named_parameters())

        for name in critic_1_params:
            critic_1_params[name] = tau*critic_1_params[name].clone() + \
                    (1-tau)*target_critic_1_params[name].clone()

        self.target_critic_1.load_state_dict(critic_1_params)


        target_critic_2_params = dict(self.target_critic_2.named_parameters())
        critic_2_params = dict(self.critic_2.named_parameters())

        for name in critic_2_params:
            critic_2_params[name] = tau*critic_2_params[name].clone() + \
                    (1-tau)*target_critic_2_params[name].clone()

        self.target_critic_2.load_state_dict(critic_2_params)

    def caclculate_loss(self):

        if self.memory.current_size < self.batch_size:
            return 0

        if self.memory.index % self.config.frequency_training != 0:
            return

        obs_arr, new_obs_arr, img_arr, new_img_arr, action_arr, reward_arr, dones_arr = self.memory.sample_data(self.batch_size)

        action, probs = self.actor.sample_normal(obs_arr, img_arr)


        entropy_coef = torch.exp(self.log_entropy_coef.detach())
        entropy_coef_loss = -(self.log_entropy_coef * (probs + self.target_entropy).detach()).mean()
        self.entropy_coef_optimizer.zero_grad()
        entropy_coef_loss.backward()
        self.entropy_coef_optimizer.step()

        with torch.no_grad():

            next_action, next_probs = self.actor.sample_normal(new_obs_arr, new_img_arr)

            next_q_values_1 = self.target_critic_1(new_obs_arr, next_action, new_img_arr)
            next_q_values_2 = self.target_critic_2(new_obs_arr, next_action, new_img_arr)
            next_q_values, _ = torch.min(torch.cat((next_q_values_1, next_q_values_2), dim=1), dim=1, keepdim=True)

            next_q_values = next_q_values - entropy_coef*next_probs

            target_q_values = reward_arr + (1-dones_arr) * self.gamma * next_q_values

        current_q_values_1 = self.critic_1(obs_arr, action_arr, img_arr)
        current_q_values_2 = self.critic_2(obs_arr, action_arr, img_arr)

        critic_loss_1 = 0.5 * F.mse_loss(current_q_values_1, target_q_values)
        critic_loss_2 = 0.5 * F.mse_loss(current_q_values_2, target_q_values)
        critic_loss = critic_loss_1 + critic_loss_2

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()


        q_values_1 = self.critic_1(obs_arr, action, img_arr)
        q_values_2 = self.critic_2(obs_arr, action, img_arr)
        q_values, _ = torch.min(torch.cat((q_values_1, q_values_2), dim=1), dim=1, keepdim=True)

        actor_loss = (entropy_coef*probs - q_values).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

        return np.round(entropy_coef.item(), 4)
    
    def train(self,env,config,metrics):
        # For every episode
        for learning_step in range(1, config.nb_learning_step + 1):

            # Reset nb terminated
            metrics.nb_terminated_train = 0

            if learning_step > 1 or not config.start_with_test:
                episode_train = 1
                while episode_train < config.nb_train_episode + 1:

                    # Reset the environnement
                    obs, _ = env.reset(test=True)
                    state = obs["info"]
                    image = obs["frame"]
                    while obs["fail"]:
                        print("failed to sync try respawn")
                        obs, _ = env.reset(test=True)
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
                            entropy=self.caclculate_loss()

                            ep_reward.append(reward)

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
            self.save_model(True)

            if restore:
                self.load_model()


    def save_model(self,recent=False):
        self.actor.save_model(recent)
        self.critic_1.save_model(recent)
        self.critic_2.save_model(recent)
        self.target_critic_1.save_model(recent)
        self.target_critic_2.save_model(recent)
        if recent:
            torch.save(self.log_entropy_coef, self.config.file_save_network + "/entropy_recent.pt")
        else:
            torch.save(self.log_entropy_coef, self.config.file_save_network + "/entropy.pt")

    def load_model(self,recent=False):
        self.actor.load_model(recent)
        self.critic_1.load_model(recent)
        self.critic_2.load_model(recent)
        self.target_critic_1.load_model(recent)
        self.target_critic_2.load_model(recent)
        if recent:
            self.log_entropy_coef = torch.load(self.config.file_save_network + "/entropy_recent.pt")
        else:
            self.log_entropy_coef = torch.load(self.config.file_save_network + "/entropy.pt")
