
from rl_td3.networks import ActorNetwork, CriticNetwork
from rl_td3.replay_buffer import ReplayBuffer
from rl_td3.config_td3 import ConfigTD3
from config import Config
from torch.distributions import Normal
import numpy as np
import torch
from torch.nn import functional as F
import torch.optim as optim
from utils.restart_celeste import *
import time
class TD3():

    def __init__(self, config_td3: ConfigTD3, config_env: Config) -> None:

        self.config = config_td3
        self.config_env = config_env
        self.size_histo = 0
        self.action_size = config_env.action_size.shape[0]
        self.action_size_discrete = config_env.action_size
        self.policy_noise=self.config.policy_noise
        self.state_size = config_env.base_observation_size
        if config_env.give_goal_coords:
            self.state_size += 4
        if config_env.give_screen_value:
            self.state_size += 1        

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.explore_noise = self.config.explore_noise
        self.clipped_noise = self.config.clipped_noise
        self.action_low = torch.tensor([1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6])
        self.action_high = torch.tensor([2.99,2.99,1.99,2.99,1.99,1.0,1.0,1.0,1.0,1.0])
        self.max_action = torch.tensor([3,3,2,3,2,1,1,1,1,1]).to(self.device)
        self.min_action = torch.tensor([1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6]).to(self.device)

        self.size_image = config_env.size_image if self.config.use_image_train else None
        self.use_image = config_env.use_image

        self.gamma = self.config.discount_factor
        self.tau = self.config.tau
        self.batch_size = self.config.batch_size
        

        self.actor = ActorNetwork(self.state_size, self.action_size, self.action_size_discrete,self.size_image, self.config,name="actor")
        self.target_actor = ActorNetwork(self.state_size, self.action_size, self.action_size_discrete,self.size_image, self.config,name="actor_target")

        self.critic_1 = CriticNetwork(self.state_size, self.action_size, self.size_image, self.config, name="critic_1")
        self.critic_2 = CriticNetwork(self.state_size, self.action_size, self.size_image, self.config, name="critic_2")
        self.target_critic_1 = CriticNetwork(self.state_size, self.action_size, self.size_image, self.config, name="target_critic_1")
        self.target_critic_2 = CriticNetwork(self.state_size, self.action_size, self.size_image, self.config, name="target_critic_2")

        self.memory = ReplayBuffer(self.config.size_buffer, self.action_size, self.state_size, self.size_image, self.size_histo)


        self.target_entropy = -1 * self.action_size * torch.ones(1, device=self.actor.device)
        init_entropy_coef = self.config.init_entropy
        self.log_entropy_coef = torch.log(init_entropy_coef*torch.ones(1, device=self.actor.device)).requires_grad_(True)
        self.entropy_coef_optimizer = optim.Adam([self.log_entropy_coef], lr=self.config.lr)

        if self.config.restore_networks:
            self.load_model(True)


        self.target_update(init=True)

    def insert_data(self, state, new_state, image, new_image, actions_probs, reward, terminated, truncated):
        self.memory.insert_data(state, new_state, image, new_image, actions_probs, reward, terminated)

    def choose_action(self, state, image):
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)        
        if self.use_image:
            image = torch.tensor(image, dtype=torch.float).to(self.actor.device)

        # call the actor network for mu and sigma 
        mu,sigma = self.actor.sample_normal(state, image)
        
        # add noise for exploration
        if self.explore_noise != 0:
            mu = (mu + torch.tensor(np.random.normal(0,self.explore_noise[:5],size=5),dtype=torch.float32).to(self.device))
            sigma = (sigma + torch.tensor(np.random.normal(0,self.explore_noise[5:],size=5),dtype=torch.float32).to(self.device))

        # print(mu,sigma)
        # clamp to make the actions in the correct range
        # mu = torch.clamp(mu, min=torch.tensor(self.action_low[:5]).to(self.device), max=torch.tensor(self.action_high[:5]).to(self.device))
        # sigma = torch.clamp(sigma, min=torch.tensor(self.action_low[5:]).to(self.device), max=torch.tensor(self.action_high[5:]).to(self.device))
        sigma = torch.log(1+torch.exp(sigma))
        # sample for mean and sigma
        probabilities = Normal(mu,sigma)
        actions = probabilities.sample()
        # print("actions",actions)
        actions = torch.clamp(actions, min=torch.tensor(self.action_low[:5]).to(self.device), max=torch.tensor(self.action_high[:5]).to(self.device))

        actions = actions.cpu().detach().numpy().flatten()

        # truncate to make them integers
        actions = np.trunc((actions.reshape(-1)))

        # return actions and mu sigma for backpropogation later on
        return actions,torch.cat([mu, sigma],dim=1)
    

    """"
    update the corresponding target network for actor and critic 1 and 2
    """
    def target_update(self, init=False):
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

        # target policy
        target_actor_params = dict(self.target_actor.named_parameters())
        actor_params = dict(self.actor.named_parameters())

        for name in actor_params:
            actor_params[name] = tau*actor_params[name].clone() + \
                    (1-tau)*target_actor_params[name].clone()

        self.target_actor.load_state_dict(actor_params)

    """
    calculate loss every step according to TD3 algorithm
    """
    def calculate_loss(self,iteration):

        if self.memory.current_size < self.batch_size:
            return 0

        if self.memory.index % self.config.frequency_training != 0:
            return

        obs_arr, new_obs_arr, img_arr, new_img_arr, action_arr, reward_arr, dones_arr = self.memory.sample_data(self.batch_size)
        obs_arr, new_obs_arr, action_arr, reward_arr, dones_arr = obs_arr.to(self.device), new_obs_arr.to(self.device), action_arr.to(self.device), reward_arr.to(self.device), dones_arr.to(self.device)

        if img_arr != None:
            img_arr.to(self.device), new_img_arr.to(self.device)

        noise = torch.empty(10).normal_(0,self.policy_noise)
        noise = noise.clamp(-self.clipped_noise,self.clipped_noise).to(self.device)

        target_mu,target_std = self.target_actor(new_obs_arr,new_img_arr)
        target_actions = (torch.cat([target_mu, target_std],dim=1)+noise)#.clamp(self.min_action,self.max_action)

        # Compute the target Q value
        with torch.no_grad():
            target_Q1 = self.target_critic_1(new_obs_arr, new_img_arr, target_actions)
            target_Q2 = self.target_critic_2(new_obs_arr, new_img_arr, target_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_arr + (dones_arr * self.gamma * target_Q).detach()

        # Get current Q estimates
        current_Q1 = self.critic_1(obs_arr, img_arr, action_arr)
        current_Q2 = self.critic_2(obs_arr, img_arr, action_arr)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)# Optimize the critic
        
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        if iteration % self.config.policy_frequency == 0:

            # Compute actor loss
            mu,std = self.actor(obs_arr,img_arr)
            actor_loss = -self.critic_1(obs_arr,img_arr,torch.cat([mu, std],dim=1)).mean()        
            # Optimize the actor 
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            self.target_update()
        return 
    
    def train(self,env,config,metrics):
        # For every episode
        doWrite = True
        if doWrite:
            if self.config.restore_networks == False:
                with open("./src/rl_td3/rewards_train.txt","w") as file:
                    file.write("start\n")
                file.close()
                with open("./src/rl_td3/rewards_test.txt","w") as file:
                    file.write("start\n")
                file.close()
            else:
                with open("./src/rl_td3/rewards_train.txt","a") as file:
                    file.write("\nstart\n")
                file.close()
                with open("./src/rl_td3/rewards_test.txt","a") as file:
                    file.write("\nstart\n")
                file.close()
        startTime = time.time()
        for learning_step in range(1, config.nb_learning_step + 1):
            if (time.time() - startTime) > 3600:
                restart_celeste(env)
                startTime = time.time()

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

                        actions,actions2 = self.choose_action([state], [image])
                        # Step the environnement
                        obs, reward, terminated, truncated, _ = env.step(actions)
                        if doWrite:
                            with open("./src/rl_td3/rewards_train.txt","a") as file:
                                file.write(f"{reward} ")
                            file.close()
                        next_state = obs["info"]
                        next_image = obs["frame"]

                        if not truncated:
                            # Insert the data in the algorithm memory
                            # save the concatenated mu and sigma for the actions to backpropogate
                            self.insert_data(state, next_state, image, next_image, actions2, reward, terminated, truncated)


                            # Actualise state
                            state = next_state
                            image = next_image

                            # Train the algorithm
                            entropy=self.calculate_loss(iteration)

                            ep_reward.append(reward)
                            iteration += 1
                    if doWrite:
                        with open("./src/rl_td3/rewards_train.txt","a") as file:
                            file.write(f"\n")
                        file.close()
                    if ep_reward:
                        metrics.print_train_step(learning_step, episode_train, ep_reward, entropy)
                        episode_train += 1

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
                    actions,actions2 = self.choose_action([state], [image])
                    # actions = torch.trunc(actions)
                    # Step the environnement
                    obs, reward, terminated, truncated, _ = env.step(actions)
                    if doWrite:
                        with open("./src/rl_td3/rewards_test.txt","a") as file:
                            file.write(f"{reward} ")
                        file.close()
                    next_state = obs["info"]
                    next_image = obs["frame"]

                    # Actualise state
                    state = next_state
                    image = next_image

                    # Add the reward to the episode reward
                    reward_ep.append(reward)
                save_model,restore = False,False
                if doWrite:
                    with open("./src/rl_td3/rewards_test.txt","a") as file:
                        file.write(f"\n")
                    file.close()
                if not truncated or truncated:
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
        self.target_actor.save_model(recent)
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
        self.target_actor.load_model(recent)
        self.critic_1.load_model(recent)
        self.critic_2.load_model(recent)
        self.target_critic_1.load_model(recent)
        self.target_critic_2.load_model(recent)
        if recent:
            self.log_entropy_coef = torch.load(self.config.file_save_network + "/entropy_recent.pt")
        else:
            self.log_entropy_coef = torch.load(self.config.file_save_network + "/entropy.pt")
