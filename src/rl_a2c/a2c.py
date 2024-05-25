import torch
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import os
from rl_a2c.networks import A2CActorNetwork, A2CCriticNetwork
from rl_a2c.config_a2c import ConfigA2C
from config import Config

class A2C():

    def __init__(self, config_a2c: ConfigA2C, config_env: Config) -> None:
        self.config = config_a2c
        self.config_env = config_env
        self.size_histo = 0
        self.entropy_weight = 0.01
        self.action_size = config_env.action_size.shape[0]
        self.action_size_discrete = config_env.action_size
        self.state_size = config_env.base_observation_size
        if config_env.give_goal_coords:
            self.state_size += 4
        if config_env.give_screen_value:
            self.state_size += 1

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.size_image = config_env.size_image if self.config.use_image_train else None
        self.use_image = config_env.use_image
        self.gamma = self.config.discount_factor
        self.batch_size = self.config.batch_size

        
        self.actor = A2CActorNetwork(self.state_size, self.action_size, self.action_size_discrete,self.size_image, self.config, name="actor")
        self.critic = A2CCriticNetwork(self.state_size, self.size_image, self.config, name="critic")

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.config.lr)

        if self.config.restore_networks:
            self.load_model(True)

    
    def choose_action(self, state, image):
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
    
        if self.use_image:
            image = torch.tensor(image, dtype=torch.float).to(self.actor.device)
    
        action, log_prob, entropy = self.actor.sample_action(state, image if self.use_image else None)
        action = torch.tensor(action).to(self.actor.device)
        log_prob = torch.tensor(log_prob).to(self.actor.device)
        #print("action",action)
        #print("log",log_prob)
        return action.cpu().detach().numpy(), log_prob.cpu().detach().numpy()
    
    
    def calculate_loss(self, state, action, reward, next_state, done, image=None, next_image=None):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.float).to(self.device)
        
        if self.use_image:
            image = torch.tensor(image, dtype=torch.float).to(self.device)
            next_image = torch.tensor(next_image, dtype=torch.float).to(self.device)
        else:
            image, next_image = None, None

        #Compute value targets
        with torch.no_grad():
            target_value = reward + (1 - done) * self.gamma * self.critic(next_state, next_image).squeeze()

        
        value = self.critic(state, image).squeeze()
        advantage = (target_value - value).detach()
        
        critic_loss = F.mse_loss(value, target_value)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Sample actions to get log probabilities
        _, log_probs, entropy = self.actor.sample_action(state, image)
        log_probs = torch.tensor(log_probs, dtype=torch.float, requires_grad=True).to(self.device)
        entropy = torch.tensor(entropy, dtype=torch.float, requires_grad=True).to(self.device)


        #actor_loss = -(log_probs * advantage).mean()    
        actor_loss = -(log_probs * advantage).mean() - self.entropy_weight * entropy.mean()    
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        return actor_loss, critic_loss.mean(), entropy

    
    def train(self, env, config, metrics):
        for learning_step in range(1, config.nb_learning_step + 1):
            metrics.nb_terminated_train = 0

            if learning_step > 1 or not config.start_with_test:
                episode_train = 1
                while episode_train < config.nb_train_episode + 1:
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

                    while not terminated and not truncated:
                        action, _ = self.choose_action(state, image)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        next_state = obs["info"]
                        next_image = obs["frame"]

                        if not truncated:
                
                            actor_loss, critic_loss, entropy = self.calculate_loss(state, action, reward, next_state, terminated, image, next_image)

                            state = next_state
                            image = next_image
                            ep_reward.append(reward)

                    if ep_reward:
                        metrics.print_train_step(learning_step, episode_train, ep_reward, entropy)
                        episode_train += 1

            episode_test = 1
            #for episode_test in range(1, config.nb_test_episode + 1):
            while episode_test < config.nb_test_episode + 1:
                terminated = False
                truncated = False
                reward_ep = list()
                obs, _ = env.reset(test=True)
                state = obs["info"]
                image = obs["frame"]
                while obs["fail"]:
                    print("failed to sync try respawn")
                    obs, _ = env.reset(test=True)
                    state = obs["info"]
                    image = obs["frame"]

                while not terminated and not truncated:
                    action, _ = self.choose_action(state, image)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    next_state = obs["info"]
                    next_image = obs["frame"]

                    state = next_state
                    image = next_image
                    reward_ep.append(reward)

                if not truncated:
                    save_model, save_video, restore, next_screen = metrics.insert_metrics(learning_step, reward_ep, episode_test, env.max_steps, env.game_step)
                    metrics.print_test_step(learning_step, episode_test)

                    if save_video:
                        print("save")
                        env.save_video()

                    if next_screen and config.max_screen_value_test < 7:
                        config.max_screen_value_test += 1
                        config.screen_used.append(config.max_screen_value_test)
                        config.prob_screen_used = np.ones(config.max_screen_value_test + 1)
                        config.prob_screen_used[0] = config.max_screen_value_test
                        config.prob_screen_used[config.max_screen_value_test] = config.max_screen_value_test + 1
                        config.prob_screen_used = config.prob_screen_used / np.sum(config.prob_screen_used)
                    episode_test += 1

            if save_model:
                self.save_model()
            self.save_model(True)

            if restore:
                self.load_model()

    def save_model(self, recent=False):
        save_dir = os.path.dirname(self.config.file_save_network)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.actor.save_model(recent)
        self.critic.save_model(recent)
        #if recent:
            #torch.save(self.actor.state_dict(), self.config.file_save_network + "/actor_recent.pt")
            #torch.save(self.critic.state_dict(), self.config.file_save_network + "/critic_recent.pt")
        #else:
            ##torch.save(self.actor.state_dict(), self.config.file_save_network + "/actor.pt")
            #torch.save(self.critic.state_dict(), self.config.file_save_network + "/critic.pt")

    def load_model(self, recent=False):
        self.actor.load_model(recent)
        self.critic.load_model(recent)
        #if recent:
            #self.actor.load_state_dict(torch.load(self.config.file_save_network + "/actor_recent.pt"))
            #self.critic.load_state_dict(torch.load(self.config.file_save_network + "/critic_recent.pt"))
        #else:
            #self.actor.load_state_dict(torch.load(self.config.file_save_network + "/actor.pt"))
            #self.critic.load_state_dict(torch.load(self.config.file_save_network + "/critic.pt"))