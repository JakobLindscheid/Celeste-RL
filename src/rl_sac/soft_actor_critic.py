
from rl_sac.replay_buffer import ReplayBuffer
from rl_sac.actor_network import ActorNetwork
from rl_sac.critic_network import CriticNetwork
from rl_sac.value_network import ValueNetwork
from rl_sac.config_sac import ConfigSac

from config import Config


import torch
import torch.nn.functional as F
from torch.distributions import Normal

class SoftActorCritic():

    def __init__(self, config_sac: ConfigSac, config_env: Config):

        self.action_mode = "Continuous"

        self.config = config_sac
        self.size_histo = config_env.histo_image
        self.action_size = config_env.action_size.shape[0]
        self.state_size = config_env.observation_size

        self.size_image = config_env.size_image if self.config.use_image_train else None
        self.use_image = config_env.use_image

        self.discount_factor = self.config.discount_factor
        self.tau = self.config.tau
        self.batch_size = self.config.batch_size
        self.epoch = self.config.epoch

        self.memory = ReplayBuffer(self.config.size_buffer, self.action_size, self.state_size, self.size_image, self.size_histo)
        self.actor = ActorNetwork(self.state_size, self.action_size, self.size_image, self.size_histo, self.config)
        self.critic_1 = CriticNetwork(self.state_size, self.action_size, self.size_image, self.size_histo, self.config, name="critic_1")
        self.critic_2 = CriticNetwork(self.state_size, self.action_size, self.size_image, self.size_histo, self.config, name="critic_2")
        self.value = ValueNetwork(self.state_size, self.size_image, self.size_histo, self.config, name="value")
        self.target_value = ValueNetwork(self.state_size, self.size_image, self.size_histo, self.config, name="value_target")

        self.alpha = self.config.alpha
        self.update_target_network(init=True)

        if self.config.restore_networks:
            self.load_model(True)

        self.has_save_memory = True
        if self.config.activate_supervised_learning_init:
            size_memory = self.config.size_supervised_buffer
            size_image = config_env.size_image if self.config.use_image_supervised_buffer else None
            supervised_memory = ReplayBuffer(size_memory, self.action_size, self.state_size, size_image, self.size_histo)
            for partition in range(20, self.config.partition_buffer):
                supervised_memory.file_save = self.config.file_save_memory + "/memory_" + str(partition)
                print("Loading buffer                    ", end="\r")
                supervised_memory.load()
                supervised_memory.current_size = self.config.size_supervised_buffer - 1
                supervised_memory.index = self.config.size_supervised_buffer - 1

                for index in range(self.config.nb_epochs_supervised_per_partitions):
                    print("Partition : {}/{} -- Supervised training : {}/{}              ".format(
                        partition+1, self.config.partition_buffer,
                        index+1, self.config.nb_epochs_supervised_per_partitions
                    ), end=("\r" if index+1 < self.config.nb_epochs_supervised_per_partitions else "\n"))
                    self.train(memory=supervised_memory, batch_size=self.config.supervised_batch_size)

                print("End partition            ", end="\r")

                self.save_model()

        if self.config.save_supervised_buffer:
            size_memory = self.config.size_supervised_buffer
            size_image = config_env.size_image if self.config.use_image_supervised_buffer else None
            file_save_memory = self.config.file_save_memory + "/memory_0"
            self.index_memory = 0
            self.supervised_memory = ReplayBuffer(size_memory, self.action_size, self.state_size, size_image, self.size_histo, file_save_memory)



    def update_target_network(self, init=False):
        tau = 1 if init else self.tau

        param_target = dict(self.target_value.named_parameters())
        param_value = dict(self.value.named_parameters())

        for key in param_value:
            param_value[key] = tau*param_value[key].clone() + (1-tau)*param_target[key].clone()

        self.target_value.load_state_dict(param_value)

    def insert_data(self, state, new_state, image, new_image, actions_probs, reward, terminated, truncated):
        self.memory.insert_data(state, new_state, image, new_image, actions_probs, reward, terminated)
        if self.config.save_supervised_buffer:
            self.supervised_memory.insert_data(state, new_state, image, new_image, actions_probs, reward, terminated)
            if self.supervised_memory.index + 1 == self.supervised_memory.size:
                self.supervised_memory.save()
                self.supervised_memory.reset()
                self.index_memory = (self.index_memory + 1) % self.config.partition_buffer
                self.supervised_memory.file_save = self.supervised_memory.file_save[:-len(str(self.index_memory))] + str(self.index_memory)


    def choose_action(self, state, image):
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        if self.use_image:
            image = torch.tensor(image, dtype=torch.float).to(self.actor.device)
        mu, sigma = self.actor(state, image)
        probs = Normal(mu, sigma)
        actions = probs.sample()
        action = torch.tanh(actions).to(self.actor.device)
        return action.cpu().detach().numpy()

    def get_action_and_log_probs(self, state, image, rsample=False):
        mu, sigma = self.actor(state, image)
        probs = Normal(mu, sigma)
        actions = probs.rsample() if rsample else probs.sample()
        action = torch.tanh(actions).to(self.actor.device)
        log_probs = probs.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + self.actor.noise_value)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs.view(-1)


    def train(self, memory:ReplayBuffer = None, batch_size:int = None):

        if memory is None:
            memory = self.memory

        if batch_size is None:
            batch_size = self.batch_size

        if memory.current_size < self.batch_size:
            return

        if self.memory.index % self.config.frequency_training != 0:
            return

        for _ in range(self.epoch):

            state, new_state, image, new_image, actions, reward, terminated = memory.sample_data(batch_size)

            value = self.value(state, image).view(-1)
            next_value = self.target_value(new_state, new_image).view(-1)
            next_value[terminated.view(-1)] = 0.0

            action, log_probs = self.get_action_and_log_probs(state, image, rsample=False)
            critic_value_1 = self.critic_1(state, action, image)
            critic_value_2 = self.critic_2(state, action, image)
            critic_value = torch.min(critic_value_1, critic_value_2).view(-1)

            value_loss = 0.5 * F.mse_loss(value, critic_value - log_probs)
            self.value.optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            self.value.optimizer.step()


            action, log_probs = self.get_action_and_log_probs(state, image, rsample=True)
            critic_value_1 = self.critic_1(state, action, image)
            critic_value_2 = self.critic_2(state, action, image)
            critic_value = torch.min(critic_value_1, critic_value_2).view(-1)

            actor_loss = torch.mean(log_probs - critic_value)
            self.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor.optimizer.step()

            critic_obj =  self.alpha * reward.view(-1) + self.discount_factor * next_value
            critic_value_1 = self.critic_1(state, actions, image).view(-1)
            critic_value_2 = self.critic_2(state, actions, image).view(-1)
            critic_loss_1 = 0.5 * F.mse_loss(critic_value_1, critic_obj)
            critic_loss_2 = 0.5 * F.mse_loss(critic_value_2, critic_obj)
            critic_loss = critic_loss_1 + critic_loss_2

            self.critic_1.optimizer.zero_grad()
            self.critic_2.optimizer.zero_grad()
            critic_loss.backward()
            self.critic_1.optimizer.step()
            self.critic_2.optimizer.step()

        self.update_target_network()

    def save_model(self,recent=False):
        self.actor.save_model(recent)
        self.critic_1.save_model(recent)
        self.critic_2.save_model(recent)
        self.value.save_model(recent)
        self.target_value.save_model(recent)

    def load_model(self,recent=False):
        self.actor.load_model(recent)
        self.critic_1.load_model(recent)
        self.critic_2.load_model(recent)
        self.value.load_model(recent)
        self.target_value.load_model(recent)












