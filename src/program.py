"""Main program file
"""

import absl.logging
import torch
import numpy as np

from celeste_env import CelesteEnv
from config import Config
import cv2

import rl_td3 as lib

from utils.metrics import Metrics

absl.logging.set_verbosity(absl.logging.ERROR)

def main():
    """Main program
    """
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

    algo.train(env,config,metrics)





if __name__ == "__main__":
    main()
