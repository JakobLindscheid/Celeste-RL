"""File for the general config
"""

import numpy as np

from pathlib import Path
import torch

class Config:
    """Class for the general config
    """

    def __init__(self) -> None:

        # GLOBAL CONFIG

        # -------------------------------------------

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # If you want to test right away (if networks are restored or pretrained)
        self.start_with_test = False

        # Total number of episode
        self.nb_learning_step = 10_000

        # Max step per episode
        self.max_steps = 150

        # Train episode per learning step
        self.nb_train_episode = 100

        # Test episode per learning step
        self.nb_test_episode = 10

        # True if the image is used for learning
        self.use_image = True

        # True if you want a video for the best test
        self.video_best_screen = True
        self.fps = 90

        self.region = (0,0,4*320,4*180)

        # -------------------------------------------



        # FUNCTIONNING ENVIRONNEMENT CONFIGURATION

        # -------------------------------------------

        # Number of frames per action
        self.nb_frame_action = 5

        self.screen_used = [0,1]

        self.prob_screen_used = np.array([0.5,0.5])
        self.prob_screen_used = self.prob_screen_used / np.sum(self.prob_screen_used)
        
        self.max_screen_value = 7
        self.max_screen_value_test = 0
        # Tas file to run for init the first screen
        self.init_tas_file = "console load 1 {}\n   38\n***\n# end\n   1"

        # Basic waiting time (equal to 1 frame)
        self.sleep = 0.016

        # Path of TAS file
        self.path_tas_file = f"{Path(__file__).parents[1].as_posix()}/file.tas"
        # self.path_tas_file = r"C:\Users\jakob\Desktop\file2.tas"

        # Reduction factor of image screen
        self.reduction_factor = 4

        # True to indicate that size image could be wrong
        self.allow_change_size_image = True
        # Size of the screenshoted image after pooling
        self.size_image = np.array([180, 320, 3])

        # -------------------------------------------



        # INTERACTION ENVIRONNEMENT CONFIGURATION

        # -------------------------------------------

        # Action size vector
        self.action_size = np.array([3,3,2,3,2]) # horizontal, vertical, dash, jump, grab

        # Base size of observation
        self.base_observation_size = 13
        # Pos x2 : 0,1
        # Speed x2 : 2,3
        # Stamina : 4
        # Wall-L/R nothing : 5
        # StNormal/StClimb/StDash : 6,7,8
        # CanDash : 9
        # Coyote : 10
        # Jump : 11
        # DashCD : 12

        # True if the action are given in the observation
        self.give_former_actions = True

        # True if the goal coordinate are given
        self.give_goal_coords = True

        # True if the index of the screen is give
        self.give_screen_value = True

        # Reward for death
        self.reward_death = -5

        # Reward when screen passed
        self.reward_screen_passed = 100

        # Reward when wrong screen passed
        self.reward_wrong_screen_passed = 0

        # Reward when nothing append
        self.natural_reward = -2

        # True will set done if screen is based, False only when last screen is passed
        self.one_screen = False

        # Train with start pos of screens only
        self.start_pos_only = False


        # -------------------------------------------



        # METRICS CONFIGURATION

        # -------------------------------------------

        self.val_test = 100

        self.color_graph = {
            "Death": "#1f77b4",
            "Level passed": "#2ca02c",
            "Unfinished": "#ff7f0e",
            "Step 1": "#d62728",
            "Step 2": "#9467bd",
            "Step 3": "#8c564b"
        }

        self.limit_restore = 3000

        # -------------------------------------------
