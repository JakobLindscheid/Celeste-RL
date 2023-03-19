"""Main program file
"""

import absl.logging

from celeste_env import CelesteEnv
from config_multi_qnetworks import ConfigMultiQnetworks
from config import Config
from qnetwork import MultiQNetwork
from metrics import Metrics

absl.logging.set_verbosity(absl.logging.ERROR)

def main():
    """Main program
    """

    # Create the instance of the general configuration and algorithm configuration
    config = Config()
    config_algo = ConfigMultiQnetworks()

    # Create the environnement
    env = CelesteEnv(config)

    # Create the RL algorithm
    algo = MultiQNetwork(config_algo, config)

    # Create the metrics instance
    metrics = Metrics(config)

    # For every episode
    episode = 1
    while episode <= config.num_episodes:

        # Init the episode reward at 0
        reward_ep = list()

        # Reset the environnement
        state, available_actions, done = env.reset(1)

        # For each step
        while env.current_step < config.max_steps and not done:

            # Get the actions
            action = algo.choose_action(state, available_actions)

            # Step the environnement
            next_state, reward, done, available_actions, _ = env.step(action)

            # Insert the data in the algorithm memory
            algo.insert_data((state, action, reward, next_state, done))

            # Actualise state
            state = next_state

            # Add the reward to the episode reward
            reward_ep.append(reward)

        # Train the algorithm
        algo.train(episode)

        # Insert the metrics
        save_model = metrics.insert_metrics(reward_ep, episode)

        # Save the model (will be True only if new max reward)
        if save_model:
            algo.save()

        # Print the information about the episode
        metrics.print_step(episode, algo.epsilon)

        # Incremente the episode
        episode += 1




if __name__ == "__main__":
    main()
