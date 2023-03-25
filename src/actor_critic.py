"""Multiple DQN class file
"""

import random as r
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from config_actor_critic import ConfigActorCritic
from config import Config

class ActorCritic(nn.Module):
    def __init__(self, config_actor_critic: ConfigActorCritic, config: Config):
        super(ActorCritic, self).__init__()


        self.action_probs = None
        self.values = None
        self.rewards = None


        # Config of Multiple Qnetwork
        self.config = config_actor_critic

        # Get the right model
        self.type_model = "SIMPLE_MLP" if not config.use_image else "CNN_MLP"

        # Get the size of the image (will not be used if simple MLP)
        self.size_image = config.size_image


        # Current learning rate
        self.cur_lr = self.config.init_lr


        # Observation and action size
        self.state_size = config.observation_size
        self.action_size = config.action_size



        self.couche1 = nn.Linear(self.state_size, 128)
        self.couche2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, self.action_size[0])
        self.critic = nn.Linear(128,1)

        self.optimizer = torch.optim.Adam(params=list(self.parameters()), lr=self.cur_lr)


        # Restore the network with saved one
        if self.config.restore_networks:
            self.restore()

    def forward(self, x):
        x = self.couche1(x)
        x = torch.relu(x)
        x = self.couche2(x)
        x = torch.relu(x)
        x_actor = self.actor(x)
        x_critic = self.critic(x)
        return x_actor, x_critic


    def choose_action(self,state, available_actions):

        # Formate state
        state = torch.tensor(state.reshape(-1), requires_grad=True, dtype=torch.float32)

        action_logit, value = self(state)

        action_probs = torch.softmax(action_logit, 0)

        action = np.random.choice(len(action_probs), p=action_probs.detach().numpy())

        return action, action_probs[action], value


    def insert_data(self, action_probs, values, rewards):
        action_probs = torch.unsqueeze(action_probs, 0)
        rewards = torch.unsqueeze(torch.tensor(rewards), 0)

        if self.action_probs is None:
            self.action_probs = action_probs
            self.values = values
            self.rewards = rewards
        else:
            self.action_probs = torch.cat((self.action_probs, action_probs), dim=0)
            self.values = torch.cat((self.values, values), dim=0)
            self.rewards = torch.cat((self.rewards, rewards), dim=0)

    def train(self, episode: int):

        returns = self.get_expected_returns(self.rewards)

        returns = torch.unsqueeze(returns, 1)
        self.values = torch.unsqueeze(self.values, 1)
        self.action_probs = torch.unsqueeze(self.action_probs, 1)

        advantage = returns - self.values
        action_log_probs = torch.log(self.action_probs)
        actor_loss = -torch.sum(action_log_probs * advantage)

        critic_loss = F.huber_loss(self.values, returns, reduction="sum")

        loss = actor_loss + critic_loss


        self.zero_grad()
        loss.backward()

        self.optimizer.step()


        self.action_probs = None
        self.values = None
        self.rewards = None




    def get_expected_returns(self, rewards: torch.Tensor):

        returns = torch.zeros(rewards.size())

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = rewards.flip(0)
        discounted_sum = 0
        for index in range(rewards.shape[0]):
            reward = rewards[index]
            discounted_sum = reward + self.config.discount_factor * discounted_sum
            returns[index] = discounted_sum
        returns = returns.flip(0)

        if self.config.standardize:
            returns = ((returns - torch.mean(returns)) / 
                    (torch.std(returns) + 10e-8))

        return returns

    def save(self):
        torch.save(self.state_dict(), "network/network")

    def restore(self):
        self.load_state_dict(torch.load("network/network"))
