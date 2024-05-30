"""Main program file
"""

import argparse

import absl.logging
import torch

from celeste_env import CelesteEnv
from config import Config
from utils.metrics import Metrics

absl.logging.set_verbosity(absl.logging.ERROR)


def main(algName):
    """Main program
    """
    if algName == "ppo":
        import sb_ppo as lib
    elif algName == "td3":
        import rl_td3 as lib
    elif algName == "sac":
        import rl_sac_v2 as lib

    elif algName == "dqn":
        import rl_dqn as lib
    elif algName == "Astar":
        import a_star as lib
    elif algName == "A2C":
        import rl_a2c as lib

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.97)
        torch.cuda.empty_cache()

    # Create the instance of the general configuration and algorithm configuration
    config = Config()
    config_algo = lib.ConfigAlgo()

    # Create the environnement
    env = CelesteEnv(config)

    # Create the RL algorithm
    algo = lib.Algo(config_algo, config)

    # Create the metrics instance
    metrics = Metrics(config)

    algo.train(env, config, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-alg", "--algorithm",
        help="what alg to use",
        default="ppo", type=str,
    )
    args = parser.parse_args()
    algName = args.algorithm

    main(algName)
