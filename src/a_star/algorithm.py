import itertools
import math
import random
import time

import numpy as np
from a_star.node import Node


class AStar:
    def __init__(self, config_algo, config_env):
        self.config_env = config_env
        self.size_histo = 0
        self.action_size = config_env.action_size.shape[0]
        self.action_size_discrete = config_env.action_size
        self.state_size = config_env.base_observation_size
        self.index_start_obs = 0
        if config_env.give_goal_coords:
            self.state_size += 4
            self.index_start_obs += 4
        if config_env.give_screen_value:
            self.state_size += 1
            self.index_start_obs += 1

        self.action_low = [0, 0, 0, 0, 0]
        self.action_high = [3, 3, 2, 3, 2]

        self.open_list = list()
        self.close_list = list()

    @staticmethod
    def path(node: Node):
        action_sequence = []
        while node.parent:
            action_sequence.append(node.action_taken)
            node = node.parent

        return action_sequence[::-1]

    def heuristic(self, state_node: Node, goal: list):
        return 20 * (abs(state_node.state[self.index_start_obs] - goal[0]) + abs(
            state_node.state[self.index_start_obs + 1] - goal[1]))

    def distance(self, u: Node, v: Node):
        return 20 * (math.sqrt((u.state[self.index_start_obs] - v.state[self.index_start_obs]) ** 2 + (
                    u.state[self.index_start_obs + 1] - v.state[self.index_start_obs + 1]) ** 2))

    def generate_all_actions(self):
        ranges = [range(low, high) for low, high in zip(self.action_low, self.action_high)]
        all_actions = list(itertools.product(*ranges))
        random.shuffle(all_actions)

        return all_actions

    def expand(self, state_node: Node, env, goal: list):
        t0 = time.time()
        all_actions = self.generate_all_actions()
        actions_taken = self.path(state_node)
        next_nodes = []
        for i, action in enumerate(all_actions):
            _, _ = env.reset(test=True)
            for take_action in actions_taken:
                env.step(take_action)

            next_state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                continue

            next_state_obs = next_state["info"]
            next_state_node = Node(state=next_state_obs, action_taken=action, parent=state_node)
            next_state_node.g_cost = state_node.g_cost + 1
            next_state_node.h_cost = self.heuristic(state_node=next_state_node, goal=goal)
            next_state_node.f_cost = next_state_node.g_cost + next_state_node.h_cost

            next_nodes.append(next_state_node)
        print(f"Time taken for each expand operation: {time.time() - t0}s")
        return list(set(next_nodes))

    def improve(self, u: Node, v: Node):
        distance_uv = self.distance(u, v)
        if v in self.open_list:
            if u.g_cost + distance_uv < v.g_cost:
                v.parent = u
                v.f_cost = u.g_cost + self.distance(u, v) + v.h_cost

        elif v in self.close_list:
            v.parent = u
            v.f_cost = u.g_cost + self.distance(u, v) + v.h_cost
            self.close_list.remove(v)
            self.open_list.append(v)

        else:
            v.f_cost = u.g_cost + self.distance(u, v) + v.h_cost
            self.open_list.append(v)

    @staticmethod
    def check_goal_reach(reward_goal_x: np.ndarray, reward_goal_y: np.ndarray, state_node: Node):
        x, y = state_node.state[0], state_node.state[1]

        if (reward_goal_x[0] <= x <= reward_goal_x[1]) and (reward_goal_y[0] <= y <= reward_goal_y[1]):
            return True

        return False

    def train(self, env, config, metrics):
        t0 = time.time()
        obs, _ = env.reset(test=True)
        state = obs["info"]
        screen_info = env.screen_info

        reward_goal_x = np.array(screen_info.goal[0])
        reward_goal_y = np.array(screen_info.goal[1])
        reward_goal_x = screen_info.normalize_x(reward_goal_x)
        reward_goal_y = screen_info.normalize_y(reward_goal_y)

        goal = [np.mean(reward_goal_x), np.mean(reward_goal_y)]

        start_state = Node(state=state)
        start_state.f = self.heuristic(start_state, goal)
        self.open_list.append(start_state)
        while self.open_list:
            self.open_list.sort()
            u = self.open_list.pop(0)
            self.close_list.append(u)
            if self.check_goal_reach(reward_goal_x=reward_goal_x, reward_goal_y=reward_goal_y, state_node=u):
                print(time.time() - t0)
                print(self.path(u))
                return self.path(u)
            else:
                succ = self.expand(state_node=u, env=env, goal=goal)
                for v in succ:
                    self.improve(u, v)
