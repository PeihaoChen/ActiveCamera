#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
from collections import defaultdict
from typing import Dict, List, Optional
import cv2

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from habitat.utils.visualizations.utils import images_to_video
from habitat.tasks.utils import quaternion_from_coeff, quaternion_rotate_vector, cartesian_to_polar
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat.utils.visualizations.utils import draw_triangle
import quaternion
import zipfile
from glob import glob
from planner.planner import AStarPlannerVector, AStarPlannerSequential
from planner.test import _compute_plans
import math
from einops import asnumpy
from skimage.measure import label
from multiprocessing.dummy import Pool as ThreadPool
import time

class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class CustomFixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)


def linear_decay(epoch: int, total_num_updates: int) -> float:
    r"""Returns a multiplicative factor for linear value decay

    Args:
        epoch: current epoch number
        total_num_updates: total number of epochs

    Returns:
        multiplicative factor that decreases param value linearly
    """
    return 1 - (epoch / float(total_num_updates))


def _to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


def batch_obs(
    observations: List[Dict], device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of lists of observations.
    """
    batch = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(_to_tensor(obs[sensor]))

    for sensor in batch:
        batch[sensor] = (
            torch.stack(batch[sensor], dim=0)
            .to(device=device)
            .to(dtype=torch.float)
        )

    return batch


def poll_checkpoint_folder(
    checkpoint_folder: str, previous_ckpt_ind: int
) -> Optional[str]:
    r""" Return (previous_ckpt_ind + 1)th checkpoint in checkpoint folder
    (sorted by time of last modification).

    Args:
        checkpoint_folder: directory to look for checkpoints.
        previous_ckpt_ind: index of checkpoint last returned.

    Returns:
        return checkpoint path if (previous_ckpt_ind + 1)th checkpoint is found
        else return None.
    """
    assert os.path.isdir(checkpoint_folder), (
        f"invalid checkpoint folder " f"path {checkpoint_folder}"
    )
    models_paths = list(
        filter(os.path.isfile, glob.glob(checkpoint_folder + "/*"))
    )
    # models_paths.sort(key=os.path.getmtime)
    models_paths.sort(key = lambda x: int(x.split(".")[1]))
    ind = previous_ckpt_ind + 1
    if ind < len(models_paths):
        return models_paths[ind]
    return None


def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    episode_id: int,
    checkpoint_idx: int,
    metrics: Dict[str, float],
    tb_writer: TensorboardWriter,
    fps: int = 5,
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    Returns:
        None
    """
    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}={v:.2f}")

    video_name = f"episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(
        metric_strs
    )
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name)
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(
            f"episode{episode_id}", checkpoint_idx, images, fps=fps
        )


def quat_from_angle_axis(theta: float, axis: np.ndarray) -> np.quaternion:
    r"""Creates a quaternion from angle axis format

    :param theta: The angle to rotate about the axis by
    :param axis: The axis to rotate about
    :return: The quaternion
    """
    axis = axis.astype(np.float)
    axis /= np.linalg.norm(axis)
    return quaternion.from_rotation_vector(theta * axis)


def save_code(name, file_list=None,ignore_dir=['']):
    with zipfile.ZipFile(name, mode='w',
                         compression=zipfile.ZIP_DEFLATED) as zf:
        if file_list is None:
            file_list = []

        first_list = []
        for first_contents in ['*']:
            first_list.extend(glob(first_contents, recursive=True))

        for dir in ignore_dir:
            if dir in first_list:
                first_list.remove(dir)
        patterns = [x + '/**' for x in first_list]
        for pattern in patterns:
            file_list.extend(glob(pattern, recursive=True))

        file_list = [x[:-1] if x[-1] == "/" else x for x in file_list]
        for filename in file_list:
            zf.write(filename)

def save_config(config, type):
    if config is not None:
        from habitat import logger
        F = open(config.configs+'/config_of_{}.txt'.format(type), 'a')
        F.write(str(config))
        F.close()

def refine_planning_inputs(planning_inputs):
    planning_inputs['map'][:, :, 1] = 1
    planning_inputs['map'] = planning_inputs['map'].transpose(2, 1, 0)  
    planning_inputs['map'] = np.expand_dims(planning_inputs['map'], 0)  # (1, 3, 3000, 3000)

    # planning_inputs['crop_map'][:, :, 1] = 1
    # planning_inputs['crop_map'] = planning_inputs['crop_map'].transpose(2, 1, 0)
    # planning_inputs['crop_map'] = np.expand_dims(planning_inputs['crop_map'], 0)

    planning_inputs['agent_position'] = np.expand_dims(planning_inputs['agent_position'], 0)
    planning_inputs['goal'] = np.expand_dims(planning_inputs['goal'], 0)      

    return planning_inputs

def _compute_pointgoal(
    source_position, source_rotation, goal_position
):
    direction_vector = goal_position - source_position
    direction_vector_agent = quaternion_rotate_vector(
        source_rotation.inverse(), direction_vector
    )

    return np.array(
        [-direction_vector_agent[2], direction_vector_agent[0]],
        dtype=np.float32,
    )


def get_relative_goal(agent_position, agent_rotation, goal_position):
    source_position = np.array(agent_position, dtype=np.float32)
    # rotation_world_start = quaternion_from_coeff(agent_rotation)
    rotation_world_start = agent_rotation
    goal_position = np.array(goal_position, dtype=np.float32)

    return _compute_pointgoal(
        source_position, rotation_world_start, goal_position
    )

def convert_map2world(map_coors, map_shape, map_scale):
    """
    World coordinate system:
        Agent starts at (0, 0) facing upward along X. Y is rightward.
    Map coordinate system:
        Agent starts at (W/2, H/2) with X rightward and Y downward.

    Inputs:
        map_coors: (bs, 2) --- (x, y) in map coordinates
        map_shape: tuple with (H, W)
        map_scale: scalar indicating the cell size in the map
    """
    H, W = map_shape
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2

    x_map = map_coors[:, 0]
    y_map = map_coors[:, 1]

    x_world = (Hby2 - y_map) * map_scale
    y_world = (x_map - Wby2) * map_scale

    world_coors = torch.stack([x_world, y_world], dim=1)  # (bs, 2)

    return world_coors

def subtract_pose(pose_a, pose_b):
    """
    Compute pose of pose_b in the egocentric coordinate frame of pose_a.
    Inputs:
        pose_a - (bs, 3) --- (x, y, theta)
        pose_b - (bs, 3) --- (x, y, theta)

    Conventions:
        The origin is at the center of the map.
        X is upward with agent's forward direction
        Y is rightward with agent's rightward direction
    """

    x_a, y_a, theta_a = torch.unbind(pose_a, dim=1)
    x_b, y_b, theta_b = torch.unbind(pose_b, dim=1)

    r_ab = torch.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2)  # (bs, )
    phi_ab = torch.atan2(y_b - y_a, x_b - x_a) - theta_a  # (bs, )
    theta_ab = theta_b - theta_a  # (bs, )
    theta_ab = torch.atan2(torch.sin(theta_ab), torch.cos(theta_ab))

    x_ab = torch.stack(
        [r_ab * torch.cos(phi_ab), r_ab * torch.sin(phi_ab), theta_ab,], dim=1
    )  # (bs, 3)

    return x_ab

def grid2real(
    grid_x,
    grid_y,
    coordinate_min=-120.3241-1e-6,
    coordinate_max=120.0399+1e-6,
    grid_resolution=(3000, 3000)
):
    r"""Return gridworld index of realworld coordinates assuming top-left corner
    is the origin. The real world coordinates of lower left corner are
    (coordinate_min, coordinate_min) and of top right corner are
    (coordinate_max, coordinate_max)
    """
    grid_size = (
        (coordinate_max - coordinate_min) / grid_resolution[0],
        (coordinate_max - coordinate_min) / grid_resolution[1],
    )

    realworld_x = coordinate_max - grid_x * grid_size[0]
    realworld_y = grid_y * grid_size[1] + coordinate_min

    return realworld_x, realworld_y

def concat_planning_outputs(outputs):

    relative_goal, body_masks, planning_path, collision_map, local_goals = zip(*outputs)

    relative_goal = torch.cat(relative_goal)
    body_masks = torch.cat(body_masks)
    # collision_map = torch.cat(collision_map)
    # local_goals = list(g[0] for g in local_goals)
    # planning_path = list(g[0] for g in planning_path)

    return relative_goal, body_masks, planning_path, collision_map, local_goals

class Relative_Goal():
    def __init__(self, config):
        self.config = config
        self.prev_agent_position = None
        self.collided_times = torch.zeros(self.config.NUM_PROCESSES).int()
        self.collision_map = torch.zeros(self.config.RL.PLANNER.nplanners, 3000, 3000)
        self.visited_map = torch.zeros(self.config.RL.PLANNER.nplanners, 3000, 3000)
        self.col_width = torch.ones(self.config.RL.PLANNER.nplanners)
        self.sample_random_goal_mask = [False]*self.config.RL.PLANNER.nplanners
        self.pre_use_random_goal_mask = [False]*self.config.RL.PLANNER.nplanners
        self.sample_goal = torch.zeros(self.config.RL.PLANNER.nplanners, 2)
        self.prev_env_len = self.config.RL.PLANNER.nplanners
        self.sample_goal_type = self.config.sample_goal_type
        if self.config.RL.PLANNER.nplanners > 1:
            self.planner = AStarPlannerVector(self.config.RL.PLANNER)
        else:
            self.planner = AStarPlannerSequential(self.config.RL.PLANNER)

        self.frontier_record = [[] for i in range(self.config.NUM_PROCESSES)]

    def compute_relative_goal(self,planning_inputs, observations, prev_actions, curr_env_len, envs_to_pause = []):
        # if curr_env_len != self.prev_env_len:
        if len(envs_to_pause) > 0:
            self.config.defrost()
            self.config.RL.PLANNER.nplanners = curr_env_len
            self.config.freeze()
            if self.config.RL.PLANNER.nplanners > 1:
                self.planner = AStarPlannerVector(self.config.RL.PLANNER)
            else:
                self.planner = AStarPlannerSequential(self.config.RL.PLANNER)
            
            state_index = list(range(self.prev_env_len))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                self.frontier_record.pop(idx)

            # indexing along the batch dimensions
            self.collided_times = self.collided_times[state_index]
            self.collision_map = self.collision_map[state_index]
            self.visited_map = self.visited_map[state_index]
            self.col_width = self.col_width[state_index]
            self.sample_goal = self.col_width[state_index]

            self.sample_random_goal_mask = [self.sample_random_goal_mask[j] for j in range(len(self.sample_random_goal_mask)) if j in state_index ]
            self.pre_use_random_goal_mask = [self.pre_use_random_goal_mask[k] for k in range(len(self.pre_use_random_goal_mask)) if k in state_index ]
        
        self.prev_env_len = curr_env_len

        for i in range(self.config.RL.PLANNER.nplanners):
            if planning_inputs[i]['_elapsed_steps'] == 0:
                self.frontier_record[i] = []

        for i in range(self.config.RL.PLANNER.nplanners):
            planning_inputs[i] = refine_planning_inputs(planning_inputs[i])

        inputs = {}

        for key in planning_inputs[0].keys():
            if key != 'agent_state':
                inputs[key] = np.concatenate([i[key] for i in planning_inputs])
        
        use_random_goal_mask = inputs['curr_goal_info']

        self.collided_times = (self.collided_times + inputs["collided"]) * inputs["collided"]

        masks = torch.ones(self.config.RL.PLANNER.nplanners).int()

        self.add_wall2map(observations,prev_actions,inputs) # 补墙操作

        self.prev_agent_position = inputs["agent_position"].copy()

        self.sample_random_goal_mask = (inputs['_elapsed_steps'] != 0) & self.sample_random_goal_mask
        self.pre_use_random_goal_mask = (inputs['_elapsed_steps'] != 0) & self.pre_use_random_goal_mask

        self.sample_random_goal_mask = ((~ self.pre_use_random_goal_mask) &  use_random_goal_mask) * use_random_goal_mask \
                                     |  ( self.pre_use_random_goal_mask | (~ use_random_goal_mask)) * self.sample_random_goal_mask
        self.pre_use_random_goal_mask = use_random_goal_mask


        # ==================== sample goal=====================
        map_goal = torch.zeros(self.config.RL.PLANNER.nplanners, 2)
        for i in range(inputs['map'].shape[0]):
            if not self.sample_random_goal_mask[i] and not use_random_goal_mask[i]:
                if self.config.measure_exploration:
                    map_goal[i] = self.sample_goal[i]
                else:
                    map_goal[i] =  torch.tensor(inputs['goal'][i])
            elif not self.sample_random_goal_mask[i] and use_random_goal_mask[i]:
                map_goal[i] = self.sample_goal[i]
            else:
                free_area = np.where(inputs['map'][i][0] == 3)
                if len(free_area[0]) == 0:
                    self.sample_goal[i] = torch.tensor(inputs['agent_position'][i])
                else:
                    self.sample_goal[i] = torch.tensor(self.get_frontier(inputs['map'][i][0],inputs['agent_position'][i], i))
                    if (self.sample_goal[i] == torch.tensor(inputs['agent_position'][i])).all():
                        if len(free_area[0]) == 0:
                            self.sample_goal[i] = torch.tensor(inputs['agent_position'][i])
                        else:
                            random_goal_index = np.random.randint(0, len(free_area[0]), 1)
                            self.sample_goal[i] = torch.tensor([free_area[1][random_goal_index[0]], free_area[0][random_goal_index[0]]])
                map_goal[i] = self.sample_goal[i]
            
        # ==================
        max_len = max(20, inputs['bias'][:,2:4].max())

        inputs['crop_map'] = np.zeros([self.config.RL.PLANNER.nplanners, 3, max_len, max_len])
        inputs['crop_agent_position'] = np.zeros([self.config.RL.PLANNER.nplanners, 2])
        inputs['crop_map_goal'] = torch.zeros_like(map_goal)
        crop_collsion_map = []

        for i in range(self.config.RL.PLANNER.nplanners):
            
            y,x,_,_ = inputs['bias'][i]
            inputs['crop_map'][i] = inputs['map'][i][:,x:x+max_len,y:y+max_len].copy()

            inputs['crop_agent_position'][i][0] =  inputs['agent_position'][i][0].copy() - y
            inputs['crop_agent_position'][i][1] =  inputs['agent_position'][i][1].copy() - x

            inputs['crop_map_goal'][i][0] =  map_goal[i][0].clone() - y
            inputs['crop_map_goal'][i][1] =  map_goal[i][1].clone() - x

            crop_collsion_map.append(self.collision_map[i][x:x+max_len,y:y+max_len].unsqueeze(0))
  

        crop_collsion_map = torch.cat(crop_collsion_map)

        planning_path = _compute_plans(
            self.planner,
            self.config,
            crop_collsion_map,
            inputs['crop_map'],
            torch.tensor(inputs['crop_agent_position']),
            inputs['crop_map_goal'],
            [1]*self.config.RL.PLANNER.nplanners,
        )

        for i in range(self.config.RL.PLANNER.nplanners):
            ###如果规划路径为None，或者快到目标点了，或者连续撞墙五次以上了，就执行found动作
            if (
                (planning_path[i][0] is not None and len(planning_path[i][0]) < 5) or
                planning_path[i][0] is None or
                self.collided_times[i] >= 5
            ):
                if not use_random_goal_mask[i] and not self.config.measure_exploration:
                    masks[i] = 0
                else:
                    self.sample_random_goal_mask[i] = True
            else:
                self.sample_random_goal_mask[i] = False
        relative_goal = []
        local_goals = []
        for i in range(self.config.RL.PLANNER.nplanners):
            if planning_path[i][0] is not None:
                planning_path[i]  = ([planning_path[i][0][j] + planning_inputs[i]['bias'][0][0] for j in range(len(planning_path[i][0]))],
                                    [planning_path[i][1][j] + planning_inputs[i]['bias'][0][1] for j in range(len(planning_path[i][1]))]
                                    )
            if planning_path[i][0] is None or len(planning_path[i][0]) < 5:
                local_goals.append(map_goal[i])
            else:
                
                local_goals.append(torch.tensor([planning_path[i][0][-4], planning_path[i][1][-4]]))
            real_goal_x, real_goal_y = grid2real(local_goals[i][0], local_goals[i][1])
            relative_goal.append(
                get_relative_goal(
                    planning_inputs[i]['agent_state'].position,
                    planning_inputs[i]['agent_state'].rotation,
                    [real_goal_x, planning_inputs[i]['agent_state'].position[1], real_goal_y]
                )
            )  # (x,y,z)

        relative_goal = torch.tensor(relative_goal)
        
        return relative_goal,masks.unsqueeze(1), planning_path, self.collision_map, local_goals
        
    def add_wall2map(self,observations,prev_actions,inputs, s= 0.08):
        # Update collision maps
        forward_step = self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE

        for i in range(self.config.RL.PLANNER.nplanners):

            if inputs["_elapsed_steps"][i] == 0:
                self.collision_map[i] *= 0.0
                self.col_width[i] = self.col_width[i] *0 + 1
                continue

            prev_action_i = prev_actions[i, 0].item()
            # If not forward action, skip
            if prev_action_i != 1 and prev_action_i != 2 and prev_action_i != 3:
                continue
            x1, y1 = inputs['agent_position'][i].tolist()
            x2, y2 = self.prev_agent_position[i].tolist()
            t2 = observations['body_heading'][i].item() - math.pi / 2
            if abs(x1 - x2) < 1 and abs(y1 - y2) < 1:
                self.col_width[i] += 3
                self.col_width[i] = min(self.col_width[i], 12)
                if forward_step == 1.0:
                    self.col_width[i] = 36
            else:
                self.col_width[i] = 1
            dist_trav_i = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * s
            # Add an obstacle infront of the agent if a collision happens
            if dist_trav_i < 0.7 * forward_step:  # Collision
                length = 2
                width = int(self.col_width[i].item())
                buf = 3 * (forward_step/0.25)
                cmH, cmW = self.collision_map[i].shape
                for j in range(length):
                    for k in range(width):
                        wx = (
                            x2
                            + ((j + buf) * math.cos(t2))
                            + ((k - width / 2) * math.sin(t2))
                        )
                        wy = (
                            y2
                            + ((j + buf) * math.sin(t2))
                            - ((k - width / 2) * math.cos(t2))
                        )
                        wx, wy = int(wx), int(wy)
                        if wx < 0 or wx >= cmW or wy < 0 or wy >= cmH:
                            continue
                        self.collision_map[i, wy, wx] = 1

    def get_frontier(self, full_input_map, input_position, num_env):
        # 获取有效的地图边界
        y, x, h, w = cv2.boundingRect(full_input_map)
        bias = np.array([x - 10, y - 10, w + 20, h + 20])
        input_map = full_input_map[x - 10:x + w + 10, y - 10:y + h + 10].copy()
        # 保存bias，用于后面加上相对位置，从而得到在原来地图上的位置
        input_position[0] -= bias[1]
        input_position[1] -= bias[0]

        sample_point = input_position.copy()

        if self.sample_goal_type == 'FBE': # 以FBE的方式采样得到目标点
            kernel = np.ones((3, 3), np.float32) # 给障碍物地图的障碍物膨胀一下，避免agent卡主
            best_dis = np.inf
            min_dis = 30 # 采样点的距离要大于一个步长的距离
            

            input_map = input_map[:, :]

            free = (input_map == 3).astype(np.uint8)
            free_edge = cv2.Canny(free, 1, 1)
            free_edge_dilate = cv2.dilate(free_edge, kernel, iterations=1)

            explore = (input_map == 0).astype(np.uint8)
            explore_edge = cv2.Canny(explore, 1, 1)
            explore_edge_dilate = cv2.dilate(explore_edge, kernel, iterations=1)

            unexplore_edge = free_edge_dilate & explore_edge_dilate
            labeled_img, num = label(unexplore_edge, connectivity=1, background=0, return_num=True)
            for label_num in range(1, num + 1):
                position = np.where(labeled_img == label_num)
                avg_x = int(position[0].mean())
                avg_y = int(position[1].mean())
                # dis = (map_position[0] - avg_x)**2 + (map_position[1] - avg_y)**2
                dis = (input_position[1] - avg_x) ** 2 + (input_position[0] - avg_y) ** 2
                temp_point = [avg_y + bias[1], avg_x + bias[0]]
                if self.config.No_Repeat_FBE:
                    if min_dis < dis < best_dis and temp_point not in self.frontier_record[num_env]:
                        sample_point = [avg_y, avg_x]
                        best_dis = dis
                else:
                    if min_dis < dis < best_dis:
                        sample_point = [avg_y, avg_x]
                        best_dis = dis

            record_y = np.linspace(sample_point[0] + bias[1] - 1, sample_point[0] + bias[1] + 1, 3)
            record_x = np.linspace(sample_point[1] + bias[0] - 1, sample_point[1] + bias[0] + 1, 3)
            yv, xv = np.meshgrid(record_y, record_x)
            record_point = np.stack([yv.reshape(-1), xv.reshape(-1)], axis=1).tolist()
            self.frontier_record[num_env].extend(record_point)
        elif self.sample_goal_type == 'random':
            unexplored_area = np.where(input_map == 0)
            random_goal_index = np.random.randint(0, len(unexplored_area[0]), 1)
            sample_point = np.array([unexplored_area[1][random_goal_index[0]], unexplored_area[0][random_goal_index[0]]])
        
        input_position[0] += bias[1]
        input_position[1] += bias[0]

        sample_point[0] += bias[1]
        sample_point[1] += bias[0]

        return sample_point


class to_grid():
    def __init__(self, global_map_size, coordinate_min, coordinate_max):
        self.global_map_size = global_map_size
        self.coordinate_min = coordinate_min
        self.coordinate_max = coordinate_max
        self.grid_size = (coordinate_max - coordinate_min) / global_map_size

    def get_grid_coords(self, positions):
        grid_x = ((self.coordinate_max - positions[:, 0]) / self.grid_size).round()
        grid_y = ((positions[:, 1] - self.coordinate_min) / self.grid_size).round()
        return grid_x, grid_y


def color_map(origin_map, color_mask, color):
    origin_map[color_mask, 0] = color[0]
    origin_map[color_mask, 1] = color[1]
    origin_map[color_mask, 2] = color[2]
    return origin_map


def flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
    r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

    Args:
        t: first dimension of tensor.
        n: second dimension of tensor.
        tensor: target tensor to be flattened.

    Returns:
        flattened tensor of size (t*n, ...)
    """
    return tensor.view(t * n, *tensor.size()[2:])

class SupVis:
    def __init__(self, config, crop_size=30):
        self.crop_size = crop_size
        self.save_image_flag = "frame" in config.VIDEO_OPTION
        self.UNSEEN_COLOR = (255, 255, 255)
        self.OUTSIDE_COLOR = (125, 0, 0)
        self.OCC_COLOR = (50, 90, 0)
        self.FREE_COLOR = (190, 150, 190)
        self.PATH_COLOR = (0, 0, 0)
        self.GOAL_COLOR = (255, 0, 0)
        self.LOCAL_GOAL_COLOR = (128, 0, 128)
        self.COLLISION_COLOR = (0, 0, 205)
        self.BODY_COLOR = (20, 20, 20)
        self.CAMERA_COLOR = (255, 125, 0)
        self.CIRCLE_POINTS_COLOR = (0, 139, 139)

    @staticmethod
    def color(origin_map, color_mask, color):
        origin_map[color_mask, 0] = color[0]
        origin_map[color_mask, 1] = color[1]
        origin_map[color_mask, 2] = color[2]
        return origin_map

    @staticmethod
    def crop(origin_map, map_range):
        crop_map = origin_map[map_range[0]: map_range[1], map_range[2]: map_range[3]]
        return crop_map

    def draw_basic_area(self, frame_map, occupancy_map):
        frame_map[occupancy_map == 0] = np.array(self.UNSEEN_COLOR)
        frame_map[occupancy_map == 1] = np.array(self.OUTSIDE_COLOR)
        frame_map[occupancy_map == 2] = np.array(self.OCC_COLOR)
        frame_map[occupancy_map == 3] = np.array(self.FREE_COLOR)
        return frame_map

    def draw_planning_path(self, frame_map, planning_path, map_range):
        if planning_path[0][0] is None: # 当前无法规划出路径
            return
        mask = np.zeros_like(frame_map[:, :, 0]).astype(np.bool)
        for index in range(len(planning_path)):
            mask[planning_path[index][0] - map_range[0], planning_path[index][1] - map_range[2]] = True
        frame_map[mask] = np.array(self.PATH_COLOR)

    def draw_goal(self, frame_map, obj_map):
        goal_category = np.unique(obj_map)
        for cls in goal_category:
            if cls > 1:
                frame_map[obj_map == cls] = np.array(self.GOAL_COLOR)
        return frame_map

    def draw_local_goal(self, frame_map, goals, map_range):
        mask = np.zeros_like(frame_map[:, :, 0]).astype(np.bool)
        for i in range(len(goals)):
            mask[goals[i][0]- map_range[0], goals[i][1] - map_range[2]] = True
        frame_map[mask] = np.array(self.CIRCLE_POINTS_COLOR)

    def draw_collision(self, frame_map, collision_map):
        frame_map[collision_map[..., 0] == 1] = np.array(self.COLLISION_COLOR)
        return frame_map

    def draw_agent(self, frame_map, body_heading, agent_heading, curr_pix_x, curr_pix_y):
        frame_map = self.crop(frame_map, (curr_pix_x - self.crop_size, curr_pix_x + self.crop_size, curr_pix_y - self.crop_size, curr_pix_y + self.crop_size))
        frame_map = draw_triangle(frame_map, [self.crop_size, self.crop_size], body_heading, radius=10, color=(20, 20, 20))
        frame_map = draw_triangle(frame_map, [self.crop_size, self.crop_size], agent_heading, radius=10, color=(255, 125, 0))
        return frame_map

    def save_image(self, frame_map, time_step, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        plt.imshow(frame_map, interpolation='none')
        plt.savefig('{}/{}.png'.format(save_dir, str(time_step)))

    def reize(self, map_list, size):
        assert len(map_list) <= 3
        height, width = size
        for i, map in enumerate(map_list):
            map = cv2.resize(map, (height, height))
            map_list[i] = map
        frame_map = np.concatenate(map_list, axis=1)

        # 补零使第二行与第一行相同长度
        zero_pad = np.zeros([height, width - frame_map.shape[1], 3])
        frame_map = np.concatenate([frame_map, zero_pad], axis=1)
        return frame_map

    def draw_visMap(self, vis_inputs, save_dir):
        visMap = np.zeros_like(vis_inputs['vis_map']).astype(np.uint8)
        vis_inputs['vis_map'] = vis_inputs['vis_map'].numpy()

        occ_map = vis_inputs['vis_map'][:, :, 0]
        obj_map = vis_inputs['vis_map'][:, :, 1]
        visMap = self.draw_basic_area(visMap, occ_map)
        visMap = self.draw_goal(visMap, obj_map)

        if 'collision_map' and 'planning_path' in vis_inputs.keys():
            planning_path = vis_inputs['planning_path']
            map_range = vis_inputs['map_range']
            collision_map = self.crop(vis_inputs['collision_map'], map_range)
            if planning_path[0] is not None:
                self.draw_planning_path(visMap, planning_path, map_range)
            self.draw_collision(visMap, collision_map)  

        if 'circle_points' in vis_inputs.keys():
            self.draw_local_goal(visMap, vis_inputs['circle_points'], map_range)

        body_heading, agent_heading = vis_inputs['body_heading'], vis_inputs['agent_heading']
        curr_pix_x = vis_inputs['map_position'][0] - vis_inputs['map_range'][0]
        curr_pix_y = vis_inputs['map_position'][1] - vis_inputs['map_range'][2]
        self.draw_agent(visMap, body_heading, agent_heading, curr_pix_x, curr_pix_y)
        if 'one_hot_vector' in vis_inputs.keys():
            one_hot_vector_text = 'V: {}'.format(vis_inputs['one_hot_vector']) # one hot向量
            cv2.putText(visMap, one_hot_vector_text, (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)
        if 'dis_list' in vis_inputs.keys():
            dis_list = 'D: {}'.format(vis_inputs['dis_list']) # 对应到每个目标的规划路径长度
            cv2.putText(visMap, dis_list, (10, 60), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 0), 1)


        if self.save_image_flag:
            time_step = vis_inputs['time_step']
            self.save_image(visMap, time_step, save_dir)

        return visMap

    def draw_semMap(self, vis_inputs):
        semMap = np.zeros_like(vis_inputs['semMap']).astype(np.uint8)

        occ_map = vis_inputs['semMap'][:, :, 0]
        obj_map = vis_inputs['semMap'][:, :, 1]
        semMap = self.draw_basic_area(semMap, occ_map)
        semMap = self.draw_goal(semMap, obj_map)
        # semMap = draw_triangle(semMap, [25, 25], 0, radius=2, color=(255, 125, 0))
        semMap[25, 25] = np.array((255, 125, 0))

        semMap = np.rot90(semMap, 2)

        return semMap

    def draw_new(self, vis_inputs, save_dir):
        visMap = self.draw_visMap(vis_inputs, save_dir)
        semMap = self.draw_semMap(vis_inputs)
        draw_map_list = [visMap, semMap]
        if 'NonOracle_pt' in vis_inputs.keys():
            draw_map_list.append(vis_inputs['NonOracle_pt'])

        frame_map = self.reize(draw_map_list, vis_inputs['frame_shape'][:2])

        return frame_map
