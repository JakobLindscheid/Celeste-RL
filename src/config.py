"""File for the general config
"""

import numpy as np

from pathlib import Path
from utils.screen_info import ScreenInfo
import torch

class Config:
    """Class for the general config
    """

    def __init__(self) -> None:

        # GLOBAL CONFIG

        # -------------------------------------------

        # Total number of episode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # If you want to test right away (if networks are restored or pretrained)
        self.start_with_test = False

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

        self.screen_used = [0]

        self.prob_screen_used = np.array([1])
        self.prob_screen_used = self.prob_screen_used / np.sum(self.prob_screen_used)
        
        self.max_screen_value = 7
        self.max_screen_value_test = 0
        # Tas file to run for init the first screen
        self.init_tas_file = "console load 1 {}\n   38\n\n# end\n   1"

        self.screen_info = [
            ScreenInfo(
                screen_id="1",
                screen_value=0,
                start_position = [[19, 144], [19, 144], [19, 144], [90, 128], [160, 80],  [260, 56]],
                first_frame=58,
                tas_file=self.init_tas_file,
                x_max=308, x_min=12,
                y_max=195, y_min=0,
                goal=[[ 250, 280], [0, 0]],
                next_screen_id = "2"
            ),
            ScreenInfo(
                screen_id="2",
                screen_value=1,
                start_position = [[264, -24], [360, -50], [403, -85], [445, -85], [530, -80]],
                first_frame=58,
                tas_file=self.init_tas_file,
                x_max=540, x_min=252,
                y_max=0, y_min=-190,
                goal=[[ 516, 540], [-190, -190]],
                next_screen_id = "3"
            ),
            ScreenInfo(
                screen_id="3",
                screen_value=2,
                start_position = [[528, -200], [645, -256], [580, -304], [600, -230], [508, -300], [700, -280], [760, -304]],
                first_frame=58,
                tas_file=self.init_tas_file,
                x_max=810, x_min=500,
                y_max=-170, y_min=-370,
                goal=[[ 764, 788], [-370, -370]],
                next_screen_id = "4"
            ),
            ScreenInfo(
                screen_id="4",
                screen_value=3,
                start_position = [[776, -392], [823, -480],[823, -480], [860, -475], [932, -480], [780, -450], [920, -520]],
                first_frame=58,
                tas_file=self.init_tas_file,
                x_max=1050, x_min=750,
                y_max=-360, y_min=-550,
                goal=[[ 908, 932], [-550, -550]],
                next_screen_id = "3b"
            ),
            ScreenInfo(
                screen_id="3b",
                screen_value=4,
                start_position = [[928, -568], [990, -584], [1110, -584], [1120, -672], [1035, -688],[1084, -680],[1028,-640]],
                first_frame=58,
                tas_file=self.init_tas_file,
                x_max=1180, x_min=880,
                y_max=-540, y_min=-735,
                goal=[[ 1075, 1052], [-735, -735]],
                next_screen_id = "5"
            ),
            ScreenInfo(
                screen_id="5",
                screen_value=5,
                start_position = [[1065, -752], [1140, -762], [1140, -880], [1230, -872], [1230, -872], [1320, -900]],
                first_frame=58,
                tas_file=self.init_tas_file,
                x_max=1340, x_min=1020,
                y_max=-700, y_min=-1020,
                goal=[[ 1310, 1325], [-1020, -1020]],
                next_screen_id = "6"
               ),
            ScreenInfo(
                screen_id="6",
                screen_value=6,
                start_position = [[1320, -1040], [1320, -1120], [1326, -1160], [1400, -1180], [1440, -1150], [1520, -1072], [1580, -1120]],
                first_frame=58,
                tas_file=self.init_tas_file,
                x_max=1620, x_min=1290,
                y_max=-1000, y_min=-1230,
                goal=[[ 1620, 1620], [-1128, -1220]],
                next_screen_id = "6a"
            ),
            ScreenInfo(
                screen_id="6a",
                screen_value=7,
                start_position = [[1624, -1128], [1643, -1080], [1700, -1040], [1732, -1125], [1833, -1136]],
                first_frame=58,
                tas_file=self.init_tas_file,
                x_max=2010, x_min=1620,
                y_max=-1000, y_min=-1230,
                goal=[[ 2010, 2010], [-1128, -1220]],
                next_screen_id = "6b"
            ),
            ScreenInfo(
                screen_id="6c",
                screen_value=9,
                start_position = [[2336, -1304], [1140, -762], [1140, -880], [1230, -872], [1230, -872], [1320, -900]],
                first_frame=58,
                tas_file=self.init_tas_file,
                x_max=2612, x_min=2330,
                y_max=-1260, y_min=-1450,
                goal=[[ 2596, 2612], [-1450, -1450]],
                next_screen_id = "7"
            )
        ]


        # Basic waiting time (equal to 1 frame)
        self.sleep = 0.017

        # Path of TAS file
        self.path_tas_file = f"{Path(__file__).parents[1].as_posix()}/file.tas"

        # Reduction factor of image screen
        self.reduction_factor = 4

        # True to indicate that size image could be wrong
        self.allow_change_size_image = True
        # Size of the screenshoted image after pooling
        self.size_image = np.array([3, 180, 320])

        # -------------------------------------------



        # INTERACTION ENVIRONNEMENT CONFIGURATION

        # -------------------------------------------

        # Action size vector
        self.action_size = np.array([3,3,2,3,2])
        # 9 for dashes in each direction
        # 9 for right, left, up, down + diagonals
        # 2 for jump
        # 2 for hold/climb

        # Base size of observation
        self.base_observation_size = 11
        # Pos x2 : 0,1
        # Speed x2 : 2,3
        # Stamina : 4
        # Wall-L/R nothing : 5
        # StNormal/StClimb/StDash : 6
        # CanDash : 7
        # Coyote : 8
        # Jump : 9
        # DashCD : 10

        # True if the action are given in the observation
        self.give_former_actions = True

        # If True, the base size of observation is bigger
        if self.give_former_actions:
            self.base_observation_size = self.base_observation_size + len(self.action_size)

        # Quantity of former iteration state and action (if action given) put if the observation vector
        self.histo_obs = 10
        self.histo_image = 2

        # Calculate the real size of observation
        self.observation_size = (self.histo_obs + 1) * self.base_observation_size

        # True if the goal coordinate are given
        self.give_goal_coords = True

        # If True, add 4 because the coordinate are 2 X value and 2 Y value
        if self.give_goal_coords:
            self.observation_size += 4

        # True if the index of the screen is give
        self.give_screen_value = True

        # Actualise the obseration size
        if self.give_screen_value:
            self.observation_size += 1

        # Reward for death
        self.reward_death = -5


        # Reward when screen passed
        self.reward_screen_passed = 100

        # Reward when wrong screen passed
        self.reward_wrong_screen_passed = 0

        # Reward when nothing append
        self.natural_reward = -1

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
