#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type
import torch
import habitat
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry
import numpy as np
import copy
from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_rotate_vector,
)
from habitat.utils.geometry_utils import *
from habitat_baselines.common.utils import (
    Relative_Goal,
    get_relative_goal
)
import math
from einops import asnumpy
from planner.planner import AStarPlannerVector, AStarPlannerSequential
from planner.test import _compute_plans
from collections import defaultdict, deque

pi = math.pi


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="NavRLEnv")
class NavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE
        self._subsuccess_measure_name = self._rl_config.SUBSUCCESS_MEASURE
        self._exp_reward_measure_name = None
        if config.expose_type == "grown":
            self._exp_reward_measure_name = "grown_fow_map"
        elif config.expose_type == "depth":
            self._exp_reward_measure_name = "depth_fow_map"

        self._previous_measure_nav = None
        self._previous_measure_expose = None
        self._previous_action = None

        config.defrost()
        config.NUM_PROCESSES = 1
        config.RL.PLANNER.nplanners = config.NUM_PROCESSES
        config.RL.PLANNER.use_weighted_graph = False
        config.freeze()
        self.camera_planner = AStarPlannerSequential(config.RL.PLANNER)
        self.camera_collision_map = torch.zeros(config.RL.PLANNER.nplanners, 2*(config.anchor_length+10), 2*(config.anchor_length+10))

        self.before_he_relative_angle = np.zeros(1)
        self.after_he_relative_angle = np.zeros(1)


        config.defrost()
        config.RL.PLANNER.use_weighted_graph = True
        config.freeze()
        self.config = config
        self.Relative_Goal = Relative_Goal(config)
        self.body_action = deque(maxlen=5)
        
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self.body_action = deque(maxlen=5)
        self._previous_action = None
        observations = super().reset()
        self._previous_measure_nav = self._env.get_metrics()[
            self._reward_measure_name
        ]
        if self._exp_reward_measure_name is not None:
            self._previous_measure_expose = self._env.get_metrics()[self._exp_reward_measure_name].numpy().sum() * 0.08 * 0.08
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_circle_points(self, body_change_angle = 0):
        agent_state = self._env._sim.get_agent_state()
        agent_position = self._env.conv_grid(agent_state.position[0], agent_state.position[2], grid_resolution=(3000, 3000))
        agent_rotation = compute_heading_from_quaternion(agent_state.sensor_states['rgb'].rotation)
        agent_body_rotation = compute_heading_from_quaternion(agent_state.rotation) + body_change_angle

        curr_map = self._env.map_masked
        ego_map = curr_map[agent_position[0] - (self.config.anchor_length+10): agent_position[0] + (self.config.anchor_length+10), \
                           agent_position[1] - (self.config.anchor_length+10): agent_position[1] + (self.config.anchor_length+10), :]

        def points_in_circum(center, rotation, r, n=8, extra_center = None):
            beta = 3 * pi / 2 - rotation
            return_extra = True
            if extra_center == None:
                extra_center = center
                return_extra = False
                
            circum = []
            extra_circum = []
            for x in range(n):
                circum.append([int(math.cos(2 * pi / n * x + beta) * r + center[0]), int(math.sin(2 * pi / n * x + beta) * r + center[1])])
                extra_circum.append([int(math.cos(2 * pi / n * x + beta) * r + extra_center[0]), int(math.sin(2 * pi / n * x + beta) * r + extra_center[1])])
            # circum = [
            #     [int(math.cos(2 * pi / n * x + beta) * r + center[0]),
            #      int(math.sin(2 * pi / n * x + beta) * r + center[1])] for x in range(n)
            # ]
            if return_extra:
                return circum, extra_circum

            return circum

        center_point = (self.config.anchor_length+10, self.config.anchor_length+10)
        circle_points, intergral_circle_points= points_in_circum(
            center_point, agent_rotation, self.config.anchor_length,
            self.config.circle_point_num, (agent_position[0],agent_position[1])
        )
        body_circle_point = points_in_circum(center_point, agent_body_rotation, self.config.anchor_length, 1)
        # intergral_circle_points = points_in_circum((agent_position[0],agent_position[1]), agent_rotation, 30)

        heuristic_inputs = {
            'center_point': center_point,
            'circle_points': circle_points,
            'body_circle_point': body_circle_point,
            'ego_map': ego_map,
            'agent_state': [agent_position, agent_rotation],
            'intergral_circle_points': intergral_circle_points
        }

        return heuristic_inputs

    def get_heuristic_goal(self, curr_body_action = 0):
        if curr_body_action == 2:
            body_change_angle = -1/6 * math.pi
        elif curr_body_action == 3:
            body_change_angle = 1/6 * math.pi
        else:
            body_change_angle = 0
        inputs = self.get_circle_points(body_change_angle)
        agent_position = inputs['agent_state'][0]
        agent_rotation = compute_quaternion_from_heading(inputs['agent_state'][1])
        circle_one_hot = [0] * len(inputs['circle_points'])

        ego_map = inputs['ego_map'].transpose(2, 1, 0)
        ego_map[1, :, :] = 1
        ego_map = np.expand_dims(ego_map, 0)
        center_point = torch.tensor(np.expand_dims(inputs['center_point'], 0))
        circle_points = torch.tensor(np.expand_dims(inputs['circle_points'], 1))
        dis_list = [0] * len(inputs['circle_points'])

        for idx, circle_point in enumerate(circle_points):    
             
            if inputs['ego_map'][inputs['circle_points'][idx][0], inputs['circle_points'][idx][1], 0] != 0:
                continue
            planning_path = _compute_plans(
                self.camera_planner, self.config,
                self.camera_collision_map,
                ego_map, center_point, circle_point, [1],
            )
            dis = 0
                     
            if planning_path[0][0] is not None:
                for i in range(len(planning_path[0][0]) - 1):
                    dis += np.sqrt(
                        (planning_path[0][0][i] - planning_path[0][0][i + 1]) ** 2 +
                        (planning_path[0][1][i] - planning_path[0][1][i + 1]) ** 2
                    )
                if dis * 0.08 < 0.08 * self.config.anchor_length * self.config.exp_threshold:
                    circle_one_hot[idx] = 1
            dis_list[idx] =  round(dis * 0.08, 2)   
                 
        if sum(circle_one_hot) != 0:
            pos_idx = (circle_one_hot[:1] + circle_one_hot[1:]).index(1)
            neg_idx = (circle_one_hot[:1] + circle_one_hot[:0:-1]).index(1)
            idx = pos_idx if pos_idx <= neg_idx else - neg_idx + self.config.circle_point_num
            heuristic_goal = [
                inputs['circle_points'][idx][0] - (self.config.anchor_length+10) + agent_position[0],
                inputs['circle_points'][idx][1] - (self.config.anchor_length+10) + agent_position[1]
            ]
        else:
            idx = 0
            heuristic_goal = [
                inputs['body_circle_point'][idx][0] - (self.config.anchor_length+10) + agent_position[0],
                inputs['body_circle_point'][idx][1] - (self.config.anchor_length+10) + agent_position[1]
            ]
        heuristic_goal = self._env.grid_conv(heuristic_goal[0], heuristic_goal[1], grid_resolution=(3000, 3000))

        center_point = self._env._sim.get_agent_state().position
        self.camera_he_goal_action_before = [heuristic_goal[0], center_point[1], heuristic_goal[1]]
        relative_goal = get_relative_goal(
            center_point,
            agent_rotation,
            self.camera_he_goal_action_before,
        )
        self.before_he_relative_angle = abs(self.compute_relative_angle(np.array([relative_goal])))

        return np.array([relative_goal]), inputs['intergral_circle_points'], circle_one_hot, dis_list

    def compute_all_points_info(self, circle_point, agent_position, agent_rotation):
        point_goal = [
            circle_point[0] - (self.config.anchor_length+10) + agent_position[0],
            circle_point[1] - (self.config.anchor_length+10) + agent_position[1]
        ]
        point_goal = self._env.grid_conv(point_goal[0], point_goal[1], grid_resolution=(3000, 3000))
        center_point = self._env._sim.get_agent_state().position
        camera_he_goal_action_before = [point_goal[0], center_point[1], point_goal[1]]
        relative_goal = get_relative_goal(
            center_point,
            agent_rotation,
            camera_he_goal_action_before,
        )
        return relative_goal

    def record_action(self, curr_camera_action = 0):
        if curr_camera_action in [1,2]:
            action = True
        else:
            action = False
        self.body_action.append(action)
        if self.body_action.count(True) == 5:
            return False
        else:
            return True

    def compute_relative_angle(self, goal_xy):
        return np.arctan2(goal_xy[:, 1], goal_xy[:, 0])

    def compute_after_relative_angle(self):
        center_point = self._env._sim.get_agent_state().position
        agent_rotation = self._env._sim.get_agent_state().sensor_states['rgb'].rotation
        relative_goal = get_relative_goal(
            center_point,
            agent_rotation,
            self.camera_he_goal_action_before,
        )
        self.after_he_relative_angle = abs(self.compute_relative_angle(np.array([relative_goal])))

        return self.after_he_relative_angle

    def get_planning_inputs(self):

        agent_state = self._env._sim.get_agent_state()
        agent_position = agent_state.position

        agent_position =  self._env.conv_grid(agent_position[0], agent_position[2], grid_resolution=(3000, 3000))
        planning_inputs = {
            'map': self._env.map_masked,
            'agent_position': np.array([agent_position[0],agent_position[1]]),
            'goal':  np.array(self._env.planning_goal[self._env.task.currGoalIndex]),
            'agent_state': agent_state,
            'bias': self._env.bias,
            '_elapsed_steps': np.array([self._env._elapsed_steps]),
            "collided": np.array([self._env._sim.previous_step_collided * 1]),
            "curr_goal_info": np.array([self._env.curr_goal_info])
        }

        return planning_inputs
    
    def get_relative_goal(self):
        planning_inputs = self.get_planning_inputs()
        relative_goal, body_masks, planning_path, collision_map, local_goals = \
            self.Relative_Goal.compute_relative_goal([planning_inputs], self._env.observations, self._env.action, curr_env_len=1)

        return relative_goal, body_masks, planning_path, collision_map, local_goals
    
    def get_time_step(self):
        return self._env._elapsed_steps

    def get_NonOracle_pt(self):

        return self._env.NonOracle_pt
        
    def get_searching_info(self):
        # if not self._env.curr_goal_info and self.prev_curr_goal_info:
        #     self.found_time_info.append(self._env._elapsed_steps)
        # self.prev_curr_goal_info = self._env.curr_goal_info

        return self._env.copy_found_goal_info
    
    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations, **kwargs):
        # 计算reward_nav
        reward_nav = self._rl_config.SLACK_REWARD

        current_measure_nav = self._env.get_metrics()[self._reward_measure_name]

        if self._episode_subsuccess():
            current_measure_nav = self._env.task.foundDistance

        reward_nav += self._previous_measure_nav - current_measure_nav
        self._previous_measure_nav = current_measure_nav

        if self._episode_subsuccess():
            self._previous_measure_nav = self._env.get_metrics()[self._reward_measure_name]

        if self._episode_success():
            reward_nav += self._rl_config.SUCCESS_REWARD
        elif self._episode_subsuccess():
            reward_nav += self._rl_config.SUBSUCCESS_REWARD
        elif self._env.task.is_found_called and self._rl_config.FALSE_FOUND_PENALTY:
            reward_nav -= self._rl_config.FALSE_FOUND_PENALTY_VALUE

        # 计算reward_exp
        reward_exp = 0
        reward_exp_area = 0
        reward_exp_he = 0
        if self._exp_reward_measure_name is not None and self._rl_config.USE_AREA_REWARD:
            expose = self._env.get_metrics()[self._exp_reward_measure_name].numpy()
            current_measure_expose = expose.sum() * 0.08 * 0.08 # 单位为平方米
            reward_exp += current_measure_expose - self._previous_measure_expose
            reward_exp_area += current_measure_expose - self._previous_measure_expose
            self._previous_measure_expose = current_measure_expose
        if self._rl_config.USE_FOUND_REWARD:
            reward_exp += self._rl_config.FOUND_REWARD * self._env.increase_founding_object # 10.0
        if self._rl_config.HE_RELATIVE_ANGLE_REWARD_SCALE != 0:
            # reward_exp += (self.prev_he_relative_angle - self.curr_he_relative_angle) \
            #     * self._rl_config.HE_RELATIVE_ANGLE_REWARD_SCALE
            self.compute_after_relative_angle()
            if self._rl_config.HE_RELATIVE_ANGLE_REWARD_SCALE_DECAY:
                reward_scale = self._rl_config.HE_RELATIVE_ANGLE_REWARD_SCALE - 9.5 / 2000000 * self._env._elapsed_steps
                reward_exp += (self.before_he_relative_angle - self.after_he_relative_angle) \
                    * max(reward_scale, 0.5)
                reward_exp_he += (self.before_he_relative_angle - self.after_he_relative_angle) \
                    * max(reward_scale, 0.5)
            else:
                reward_exp += (self.before_he_relative_angle - self.after_he_relative_angle) \
                    * self._rl_config.HE_RELATIVE_ANGLE_REWARD_SCALE
                reward_exp_he += (self.before_he_relative_angle - self.after_he_relative_angle) \
                    * self._rl_config.HE_RELATIVE_ANGLE_REWARD_SCALE
        if self._rl_config.RELATIVE_ANGLE_PENALTY != 0:
            if abs(observations['heading']-observations['body_heading']) > (1/6 * math.pi):
                reward_exp -= self._rl_config.RELATIVE_ANGLE_PENALTY
        if self._rl_config.TURN_HEAD_PENALTY != 0:
            if self._env.action.item() not in [0, 1, 5, 9]:
                reward_exp -= self._rl_config.TURN_HEAD_PENALTY
            

        return reward_nav, reward_exp, reward_exp_area, reward_exp_he

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def _episode_subsuccess(self):
        return self._env.get_metrics()[self._subsuccess_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
