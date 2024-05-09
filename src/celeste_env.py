"""Celeste environnement
"""

import time
import requests
from bs4 import BeautifulSoup
import numpy as np
import torch
import dxcam
import cv2
from ahk import AHK
import gymnasium as gym
from gymnasium import spaces
from screens_config import hardcoded_screens

from config import Config

class CelesteEnv(gym.Env):
    """Class for celest environnement
    """

    def __init__(self, config: Config):

        # Create config
        self.config = config

        # True if the environnement is testing, else it is training
        self.is_testing = True

        # Init max step at config value
        self.max_steps = self.config.max_steps

        # All available screens
        self.screens = hardcoded_screens
        
        # Info about the current screen
        self.screen_info = self.screens[0]

        # Index to start the information of position, speed and stamina (decay of 4 if the goal coords are given)
        self.index_start_obs = 0
        if config.give_goal_coords:
            self.index_start_obs += 4
        if config.give_screen_value:
            self.index_start_obs += 1
        
        self.action_space = spaces.MultiDiscrete(self.config.action_size) 
        
        self.observation_space = spaces.Dict({
            "frame": spaces.Box(0, 255, tuple(self.config.size_image), dtype=np.uint8),
            "info": spaces.Box(-np.inf, np.inf, (self.config.base_observation_size + self.index_start_obs,), dtype=np.float64)
        })

        self.observation = np.zeros(self.observation_space["info"].shape)

        # Current step
        self.current_step = 0

        # inGame iteration
        self.game_step = 0

        # Position x and y of Madeline
        self.pos_x = 0
        self.pos_y = 0

        # True if Madeline is dead
        self.dead = False

        # True if the screen is passed
        self.screen_passed = False

        # True if the wrong screen is passed
        self.wrong_screen_passed = False

        # True if maddeline is dashing in the current step
        self.is_dashing = False

        # Object initiate for screen shot
        if config.use_image or config.video_best_screen:

            ahk = AHK()

            win = ahk.win_get(title='Celeste')

            if win:
                win.activate()
                win.to_top()
                
                try: # this will fail when run a second time
                    win.set_style("-0xC40000")
                except:
                    pass
                
                win.move(x=config.region[0], y=config.region[1], width=config.region[2], height=config.region[3])
                win.redraw()
            else:
                raise Exception("Celeste window not found")

            self.camera = dxcam.create(output_idx=0, output_color="BGR")

        # Object initiate for create video during test
        if config.video_best_screen:
            self.video = dict()

        self.path_tas_file = config.path_tas_file

        # Tas file to init screen
        self.init_tas_file = config.init_tas_file
        
        self.controls_before_start()

    def wait_for_frame(self, frame_searched):
        n_tries = 0
        while self.game_step != frame_searched: # and n_tries < 11:

            n_tries += 1
            
            # wait for the next frame
            time.sleep(self.config.sleep*3)

            # Init the l_text
            l_text = [""]

            # If Timer is in the text, that mean Celeste have rightfully start
            # Sometimes it has start but the delay between FastForward and getting info is too short
            while "Timer" not in "".join(l_text):

                # Get the information of the game
                response = requests.get("http://localhost:32270/tas/info", timeout=5)

                # Get the corresponding text
                l_text = BeautifulSoup(response.content, "html.parser").text.split("\n")

            # Normally Timer appear on -8 index, but sometimes there is a "Cursor" info that can crash the code
            # If "Timer" is not on -8 index, we just check all the index
            if len(l_text) > 8 and "Timer" in l_text[-8]:
                # Get game step
                self.game_step = int(l_text[-8].replace(")","").split("(")[1])
            else:
                for line in l_text:
                    if "Timer" in line:
                        # Get game step
                        self.game_step = int(line.replace(")","").split("(")[1])

            print(f"Frame: {self.game_step} / {frame_searched}")
        print()
        return n_tries < 11

    def step(self, actions):
        """Step method

        Args:
            actions (np.array): Array of actions

        Returns:
            tuple: New state, reward, done and info
        """
        # Incremente the step
        self.current_step += 1

        # Init the frame with the quantity of frames/step
        frame_to_add_l1 = "   1"
        frame_to_add_l2 = f"   {self.config.nb_frame_action-1}"

        # Add the corresponding actions (See CelesteTAS documentation for explanation)
        # horizontal
        if actions[0] == 2:
            frame_to_add_l1 += ",R"
            frame_to_add_l2 += ",R"
        if actions[0] == 0:
            frame_to_add_l1 += ",L"
            frame_to_add_l2 += ",L"
        
        # vertical
        if actions[1] == 2:
            frame_to_add_l1 += ",U"
            frame_to_add_l2 += ",U"
        if actions[1] == 0:
            frame_to_add_l1 += ",D"
            frame_to_add_l2 += ",D"

        # dash
        if actions[2] == 1:
            frame_to_add_l1 += ",X"
            self.is_dashing = True
        else:
            self.is_dashing = False

        # short jump
        if actions[3] == 1:
            frame_to_add_l1 += ",J"

        # long jump
        if actions[3] == 2:
            frame_to_add_l1 += ",J"
            frame_to_add_l2 += ",J"

        # grab
        if actions[4] == 1:
            frame_to_add_l1 += ",G"
            frame_to_add_l2 += ",G"

        # Sometimes there is a Exception PermissionError access to the file
        # The while / Try / Except is to avoid the code crashing because of this
        changes_made = False
        while not changes_made:
            try:
                # Rewrite the tas file with the frame
                with open(self.path_tas_file, "w+", encoding="utf-8") as file:
                    file.write(frame_to_add_l1 + "\n" + frame_to_add_l2 + "\n***\n# end\n   1")
                changes_made = True
            except PermissionError:
                # If error, wait 1 second
                time.sleep(1)

        # Run the tas file
        requests.get("http://localhost:32270/tas/playtas?filePath={}".format(self.config.path_tas_file), timeout=5)

        success = self.wait_for_frame(self.game_step + self.config.nb_frame_action)
        fail_see_death = not success

        # Get observation and done info
        observation, terminated = self.get_madeline_info()
        self.observation[self.index_start_obs:] = observation

        # If the image of the game is used
        screen_obs = None
        if self.config.use_image:
            screen_obs = self.get_image_game()

        if self.is_testing and self.config.video_best_screen:
            self.screen_image()

        truncated = False
        if self.current_step == self.max_steps and not terminated:
            truncated = True

        # Get the reward
        reward = self.get_reward()

        # No info passed but initiate for convention
        info = {"fail_death": fail_see_death}

        return {"frame":screen_obs, "info":self.observation}, reward, terminated, truncated, info

    def reset(self, test=False, seed=None):
        """Reset the environnement

        Args:
            test (bool): True if this is a test

        Returns:
            tuple: New state, reward
        """

        self.is_testing = test

        if self.config.video_best_screen and self.is_testing:
            self.video.clear()

        # Init the screen by choosing randomly in the screen used if not testing
        if not self.is_testing:
            self.screen_info = self.screens[np.random.choice(self.config.screen_used, p=self.config.prob_screen_used)]
        # If testing and checking all the screen, start with the first screen
        elif not self.config.one_screen:
            self.screen_info = self.screens[self.config.screen_used[0]]

        # Init the current tas file
        if self.is_testing:
            start_pos = self.screen_info.get_true_start()
        elif self.config.start_pos_only:
            start_pos = self.screen_info.get_true_start()
        else:
            start_pos = self.screen_info.get_random_start()

        tas_file = self.init_tas_file.format(str(start_pos[0]) + " " + str(start_pos[1]))

        # Init the current step
        self.current_step = 0

        # Init max step at config value
        self.max_steps = self.config.max_steps

        # True if Madeline is dead
        self.dead = False

        # True if the screen is passed
        self.screen_passed = False

        # True if the wrong screen is passed
        self.wrong_screen_passed = False

        # Write the tas file
        with open(file=self.path_tas_file, mode="w", encoding="utf-8") as file:
            file.write(tas_file)

        # Run it
        requests.get("http://localhost:32270/tas/playtas?filePath={}".format(self.config.path_tas_file), timeout=0.5)

        requests.get("http://localhost:32270/tas/sendhotkey?id=FastForwardComment", timeout=0.5)

        success = self.wait_for_frame(1)
        fail_see_death = not success

        self.observation = np.zeros(self.observation_space["info"].shape)
        
        # Get the observation of Madeline, no use for Done because it can not be True at reset
        madeline_info, _ = self.get_madeline_info(reset=True)
        self.observation[self.index_start_obs:] = madeline_info

        # If the goal coords are given, put it in the observation vector
        if self.config.give_goal_coords:
            # Get the two coords for X and Y (coords create a square goal)
            reward_goal_x = np.array(self.screen_info.goal[0])
            reward_goal_y = np.array(self.screen_info.goal[1])

            # Make sure to normalize the values
            self.observation[0:2] = self.screen_info.normalize_x(reward_goal_x)
            self.observation[2:4] = self.screen_info.normalize_y(reward_goal_y)

        if self.config.give_screen_value:
            self.observation[self.index_start_obs - 1] = self.screen_info.screen_value / self.config.max_screen_value

        # If the image of the game is used
        screen_obs = None
        if self.config.use_image:
            # Get the array of the screen
            screen_obs = self.get_image_game()

        return {"frame":screen_obs, "info":self.observation}, {"fail_death": fail_see_death}

    def change_next_screen(self):
        """Change the screen Maddeline is in.
        To use only if it is node the last screen.
        """
        # Init the screen by choosing randomly in the screen used
        self.screen_info = self.screens[self.screen_info.screen_value + 1]

        # Add the necessary step to the next screen
        self.max_steps += self.current_step

        # If the goal coords are given, put it in the observation vector
        if self.config.give_goal_coords:
            # Get the two coords for X and Y (coords create a square goal)
            reward_goal_x = np.array(self.screen_info.goal[0])
            reward_goal_y = np.array(self.screen_info.goal[1])

            # Make sure to normalize the values
            self.observation[0:2] = self.screen_info.normalize_x(reward_goal_x)
            self.observation[2:4] = self.screen_info.normalize_y(reward_goal_y)

        if self.config.give_screen_value:
            screen_value = self.screen_info.screen_value
            index = self.index_start_obs - 1
            self.observation[index] = screen_value / self.config.max_screen_value

    def screen_image(self):
        """Store the current image of the game for the video
        """
        # Capture the current image
        screen = self.camera.grab(region=self.config.region)

        while screen is None:
            time.sleep(0.05)
            screen = self.camera.grab(region=self.config.region)

        # Add the screen
        self.video[self.game_step] = screen

    def get_image_game(self):
        """Get a np array of the current screen

        Args:
            normalize (bool): True to normalize array, default=1

        Returns:
            np.array: array of the current screen
        """
        # Coordinates correspond to the celeste window when place on the automatic render left size on windows
        # So region is for me, you have to put the pixels square of your Celeste Game
        frame = self.camera.grab(region=self.config.region)

        # Sometimes the frame is None because it need more time to refresh, so wait..
        while frame is None:
            time.sleep(0.05)
            frame = self.camera.grab(region=self.config.region)

        # Definition of pooling size to reduce the size of the image
        pooling_size = self.config.reduction_factor

        frame = frame[0::pooling_size, 0::pooling_size, :]
        
        return frame

    def get_madeline_info(self, reset=False):
        """Get the observation of madeline

        Args:
            reset (bool): True if the method is used during reset. Defaults to False.

        Returns:
            np.array, bool: observation and done
        """
        # Init the observation vector
        observation = np.zeros(self.config.base_observation_size)

        # Get the observation information, not gonna detail those part because it is just the string interpretation
        # Run "http://localhost:32270/tas/info" on a navigator to understand the information gotten
        response = requests.get("http://localhost:32270/tas/info", timeout=5)
        l_text = BeautifulSoup(response.content, "html.parser").text.split("\n")

        # Init done at False
        done = False
        self.screen_passed = False

        for line in l_text:

            if "Pos" in line:
                # get position
                pos = line.split(' ')[-2:]
                self.pos_x = float(pos[0].replace(",",""))
                self.pos_y = float(pos[1].replace("\r", ""))

                # Normalise the information
                observation[0] = self.screen_info.normalize_x(self.pos_x)
                observation[1] = self.screen_info.normalize_y(self.pos_y)

            if "Speed" in line:
                # Only speed is get for now, maybe velocity will be usefull later
                speed = line.split()
                observation[2] = float(speed[1].replace(",", ""))/300
                observation[3] = float(speed[2])/300

            if "Stamina" in line:
                # Stamina
                observation[4] = float(line.split()[1])/110

            # By default, 0
            if "Wall-L" in line:
                observation[5] = -1
            elif "Wall-R" in line:
                observation[5] = 1

            # NOTE: this does not really make sense. Could be different inputs.
            if "StNormal" in line:
                observation[6] = 0
            elif "StDash" in line:
                observation[6] = 0.5
            elif "StClimb" in line:
                observation[6] = 1

            # By default 0
            if "CanDash" in line:
                observation[7] = 1

            if "Coyote" in line:
                observation[8] = int(line.split("Coyote")[1][1]) / 5 # 5 is max value of coyotte

            if "Jump" in line and not "AutoJump" in line and not "IntroJump" in line: # If more than 10
                if line.split("Jump")[1][2].isnumeric():
                    value = int(line.split("Jump")[1][1:3])
                else:
                    value = int(line.split("Jump")[1][1])
                observation[9] = value / 14 # 14 is max value of jump

            if "DashCD" in line:
                if line.split("DashCD")[1][2].isnumeric():
                    value = int(line.split("DashCD")[1][1:3])
                else:
                    value = int(line.split("DashCD")[1][1])
                observation[10] = value / 11 # 14 is max value of jump

            # If dead
            if "Dead" in line:
                done = True
                self.dead = True

            if "Timer" in line:
                # If screen pasted. Only on screen 1 for now, will be change later
                if f"[{self.screen_info.next_screen_id}]" in line:
                    self.screen_passed = True
                    if self.config.one_screen:
                        done = True

                    else:
                        if self.screen_info.screen_value == self.config.max_screen_value_test:
                            done = True

                        else:
                            self.change_next_screen()

                # Else if the current screen id is not in text, then the wrong screen as been pasted
                elif f"[{self.screen_info.screen_id}]" not in line:
                    self.wrong_screen_passed = True
                    done = True

        return observation, done

    def get_reward(self):
        """Get the reward

        Returns:
            int: Reward
        """
        # If dead
        if self.dead:
            return self.config.reward_death

        # If screen passed
        if self.screen_passed:
            return self.config.reward_screen_passed

        if self.wrong_screen_passed:
            return self.config.reward_wrong_screen_passed

        # Else reward is natural reward
        return self.config.natural_reward * np.square(self.screen_info.distance_goal(self.pos_x, self.pos_y) / 0.7)

    def save_video(self):
        """Save the saved video of this episode
        """

        # Write the images in the file
        all_frames = list(self.video)
        last_frame = all_frames[-1]
        current_frame_index = 0

        # Configuration
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = self.config.fps
        size = (self.video[last_frame].shape[1], self.video[last_frame].shape[0])

        # Object creation VideoWriter
        out = cv2.VideoWriter("result.mp4", fourcc, fps, size)

        for frame in range(last_frame+1):
            current_frame = all_frames[current_frame_index]
            out.write(self.video[current_frame])

            if frame > current_frame:
                current_frame_index += 1

        # Close the file
        out.release()

    def controls_before_start(self):
        """Controls before the start of the test: 
        - Make sure that the init tas file work
        - Check the screen
        """

        # Reset the environnement
        self.reset()

        # Save the image
        if self.config.use_image:
            screen = self.get_image_game()
            cv2.imwrite('screen.png', screen)

            # If the image shape is not the same as the one in config
            assert screen.shape[0] == self.config.size_image[0] and screen.shape[1] == self.config.size_image[1], f"Image shape is not the same as the one in the config file. Expected: {self.config.size_image}, got: {screen.shape}"

    def render(self):
        pass