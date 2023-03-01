#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
import time
import warnings
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter
import tqdm
from einops import rearrange
from torch.optim.lr_scheduler import LambdaLR

from habitat.tasks.nav import camera_action
from habitat.tasks.nav.camera_action import joint_action
from habitat_baselines.rl.ppo.ppo import PPONonOracle
import torch

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

import cv2

from habitat import Config, logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import (
    BaseRLTrainerNonOracle,
    BaseRLTrainerOracle,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage import (
    RolloutStorageNonOracle,
    RolloutStorageOracle,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    Relative_Goal,
    SupVis,
    batch_obs,
    generate_video,
    linear_decay,
    concat_planning_outputs,
)
from habitat_baselines.rl.ppo import (
    BaselinePolicyNonOracle,
    BaselinePolicyOracle,
    PPONonOracle,
    PPOOracle,
    RandomCameraPolicy,
    OnlyLeftCameraPolicy,
    NoneCameraPolicy,
    SLAMBodyPolicy,
    HeuristicCameraPolicy
)
from sklearn.metrics import confusion_matrix

def to_grid(coordinate_min, coordinate_max, global_map_size, position):
    grid_size = (coordinate_max - coordinate_min) / global_map_size
    grid_x = ((coordinate_max - position[0]) / grid_size).round()
    grid_y = ((position[1] - coordinate_min) / grid_size).round()
    return int(grid_x), int(grid_y)


def draw_projection(image, depth, s, global_map_size, coordinate_min, coordinate_max):
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
    depth = torch.tensor(depth).permute(2, 0, 1).unsqueeze(0)
    spatial_locs, valid_inputs = _compute_spatial_locs(depth, s, global_map_size, coordinate_min, coordinate_max)
    x_gp1 = _project_to_ground_plane(image, spatial_locs, valid_inputs, s)

    return x_gp1


def _project_to_ground_plane(img_feats, spatial_locs, valid_inputs, s):
    outh, outw = (s, s)
    bs, f, HbyK, WbyK = img_feats.shape
    device = img_feats.device
    eps=-1e16
    K = 1

    # Sub-sample spatial_locs, valid_inputs according to img_feats resolution.
    idxes_ss = ((torch.arange(0, HbyK, 1)*K).long().to(device), \
                (torch.arange(0, WbyK, 1)*K).long().to(device))

    spatial_locs_ss = spatial_locs[:, :, idxes_ss[0][:, None], idxes_ss[1]] # (bs, 2, HbyK, WbyK)
    valid_inputs_ss = valid_inputs[:, :, idxes_ss[0][:, None], idxes_ss[1]] # (bs, 1, HbyK, WbyK)
    valid_inputs_ss = valid_inputs_ss.squeeze(1) # (bs, HbyK, WbyK)
    invalid_inputs_ss = ~valid_inputs_ss

    # Filter out invalid spatial locations
    invalid_spatial_locs = (spatial_locs_ss[:, 1] >= outh) | (spatial_locs_ss[:, 1] < 0 ) | \
                        (spatial_locs_ss[:, 0] >= outw) | (spatial_locs_ss[:, 0] < 0 ) # (bs, H, W)

    invalid_writes = invalid_spatial_locs | invalid_inputs_ss

    # Set the idxes for all invalid locations to (0, 0)
    spatial_locs_ss[:, 0][invalid_writes] = 0
    spatial_locs_ss[:, 1][invalid_writes] = 0

    # Weird hack to account for max-pooling negative feature values
    invalid_writes_f = rearrange(invalid_writes, 'b h w -> b () h w').float()
    img_feats_masked = img_feats * (1 - invalid_writes_f) + eps * invalid_writes_f
    img_feats_masked = rearrange(img_feats_masked, 'b e h w -> b e (h w)')

    # Linearize ground-plane indices (linear idx = y * W + x)
    linear_locs_ss = spatial_locs_ss[:, 1] * outw + spatial_locs_ss[:, 0] # (bs, H, W)
    linear_locs_ss = rearrange(linear_locs_ss, 'b h w -> b () (h w)')
    linear_locs_ss = linear_locs_ss.expand(-1, f, -1) # .contiguous()

    proj_feats, _ = torch_scatter.scatter_max(
                        img_feats_masked,
                        linear_locs_ss,
                        dim=2,
                        dim_size=outh*outw,
                    )
    proj_feats = rearrange(proj_feats, 'b e (h w) -> b e h w', h=outh)

    # Replace invalid features with zeros
    eps_mask = (proj_feats == eps).float()
    proj_feats = proj_feats * (1 - eps_mask) + eps_mask * (proj_feats - eps)

    return proj_feats


def _compute_spatial_locs(depth_inputs, s, global_map_size, coordinate_min, coordinate_max):
    bs, _, imh, imw = depth_inputs.shape
    local_scale = float(coordinate_max - coordinate_min)/float(global_map_size)
    cx, cy = 256./2., 256./2.
    fx = fy =  (256. / 2.) / np.tan(np.deg2rad(79. / 2.))

    #2D image coordinates
    x    = rearrange(torch.arange(0, imw), 'w -> () () () w')
    y    = rearrange(torch.arange(imh, 0, step=-1), 'h -> () () h ()')
    xx   = (x - cx) / fx
    yy   = (y - cy) / fy

    # 3D real-world coordinates (in meters)
    Z            = depth_inputs
    X            = xx * Z
    Y            = yy * Z
    # valid_inputs = (depth_inputs != 0) & ((Y < 1) & (Y > -1))
    valid_inputs = (depth_inputs != 0) & ((Y > -0.5) & (Y < 1))

    # 2D ground projection coordinates (in meters)
    # Note: map_scale - dimension of each grid in meters
    # - depth/scale + (s-1)/2 since image convention is image y downward
    # and agent is facing upwards.
    x_gp            = ( (X / local_scale) + (s-1)/2).round().long() # (bs, 1, imh, imw)
    y_gp            = (-(Z / local_scale) + (s-1)/2).round().long() # (bs, 1, imh, imw)

    return torch.cat([x_gp, y_gp], dim=1), valid_inputs


def rotate_tensor(x_gp, heading):
    sin_t = torch.sin(heading.squeeze(1))
    cos_t = torch.cos(heading.squeeze(1))
    A = torch.zeros(x_gp.size(0), 2, 3)
    A[:, 0, 0] = cos_t
    A[:, 0, 1] = sin_t
    A[:, 1, 0] = -sin_t
    A[:, 1, 1] = cos_t

    grid = F.affine_grid(A, x_gp.size())
    rotated_x_gp = F.grid_sample(x_gp, grid)
    return rotated_x_gp

@baseline_registry.register_trainer(name="oracle")
class PPOTrainerO(BaseRLTrainerOracle):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.envs = None
        self._static_encoder = False
        self._encoder = None
        self.final_actions = None

    def _setup_actor_critic_agent(self, config: Config, type="train") -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        if type=="test":
            logger.add_filehandler(self.config.EVAL_LOG_FILE)
        else:
            logger.add_filehandler(self.config.LOG_FILE)
        ppo_cfg = config.RL.PPO
        
        # Define body_AC
        self.body_AC = None
        self.body_agent = None
        if config.actor_critic.body_AC == "e2e":
            self.body_AC = BaselinePolicyOracle(
                agent_type = self.config.TRAINER_NAME,
                observation_space=self.envs.observation_spaces[0],
                action_size=self.config.RL.PPO.action_size,
                body_action_size = self.config.RL.PPO.action_size,
                hidden_size=ppo_cfg.hidden_size,
                goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
                device=self.device,
                object_category_embedding_size=self.config.RL.OBJECT_CATEGORY_EMBEDDING_SIZE,
                previous_action_embedding_size=self.config.RL.PREVIOUS_ACTION_EMBEDDING_SIZE,
                use_previous_action=self.config.RL.PREVIOUS_ACTION,
                extra_policy_inputs=self.config.body_policy_inputs,
                config = config
            )
            self.body_AC.to(self.device)
            self.body_agent = PPOOracle(
                actor_critic=self.body_AC,
                clip_param=ppo_cfg.clip_param,
                ppo_epoch=ppo_cfg.ppo_epoch,
                num_mini_batch=ppo_cfg.num_mini_batch,
                value_loss_coef=ppo_cfg.value_loss_coef,
                entropy_coef=ppo_cfg.entropy_coef,
                lr=ppo_cfg.lr,
                eps=ppo_cfg.eps,
                max_grad_norm=ppo_cfg.max_grad_norm,
                use_normalized_advantage=ppo_cfg.use_normalized_advantage,
            )
        elif config.actor_critic.body_AC == "slam":
            self.Relative_Goal = Relative_Goal(self.config)
            self.body_AC = SLAMBodyPolicy(self.config)
        else:
            raise ValueError

        # Define camera_AC
        self.camera_AC = None
        self.camera_agent = None
        if config.actor_critic.camera_AC == "none" or config.actor_critic.camera_AC == "e2e":
            self.camera_AC = NoneCameraPolicy(self.config)
        elif config.actor_critic.camera_AC == "random":
            self.camera_AC = RandomCameraPolicy()
        elif config.actor_critic.camera_AC == "onlyleft":
            self.camera_AC = OnlyLeftCameraPolicy()
        elif config.actor_critic.camera_AC == "he":
            self.camera_AC = HeuristicCameraPolicy(self.config)
        elif config.actor_critic.camera_AC == "rl":
            self.camera_AC = BaselinePolicyOracle(
                agent_type = self.config.TRAINER_NAME,
                observation_space=self.envs.observation_spaces[0],
                action_size=self.config.camera_action_size,
                body_action_size = self.config.RL.PPO.action_size,
                hidden_size=ppo_cfg.hidden_size,
                goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
                device=self.device,
                object_category_embedding_size=self.config.RL.OBJECT_CATEGORY_EMBEDDING_SIZE,
                previous_action_embedding_size=self.config.RL.PREVIOUS_ACTION_EMBEDDING_SIZE,
                use_previous_action=self.config.RL.PREVIOUS_ACTION,
                extra_policy_inputs=self.config.camera_policy_inputs,
                config = config
            )
            self.camera_AC.to(self.device)
            self.camera_agent = PPOOracle(
                actor_critic=self.camera_AC,
                clip_param=ppo_cfg.clip_param,
                ppo_epoch=ppo_cfg.ppo_epoch,
                num_mini_batch=ppo_cfg.num_mini_batch,
                value_loss_coef=ppo_cfg.value_loss_coef,
                entropy_coef=ppo_cfg.entropy_coef,
                lr=ppo_cfg.lr,
                eps=ppo_cfg.eps,
                max_grad_norm=ppo_cfg.max_grad_norm,
                use_normalized_advantage=ppo_cfg.use_normalized_advantage,
            )
        else:
            raise ValueError
        
        # logger.info(
        #     "agent number of parameters: {}".format(
        #         sum(param.numel() for param in self.camera_agent.parameters())
        #     )
        # )

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "body_state_dict": self.body_agent.state_dict() if self.body_agent is not None else None,
            "camera_state_dict": self.camera_agent.state_dict() if self.camera_agent is not None else None,
            "config": self.config,
            "body_optimizer": self.body_agent.optimizer if self.body_agent is not None else None,
            "camera_optimizer": self.camera_agent.optimizer if self.camera_agent is not None else None,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:   #TODO: 把所有load ckpt的内容放进来
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            **kwargs: additional keyword args
            *args: additional positional args

        Returns:
            dict containing checkpoint info
        """
        def _check_state_dict_info(pretrained_dict, curr_dict):
            pretrainedKeys_not_match_curr = []
            for k, v in pretrained_dict.items():
                if k not in curr_dict.keys():
                    pretrainedKeys_not_match_curr.append(k)
                elif v.shape != curr_dict[k].shape:
                    pretrainedKeys_not_match_curr.append(k)
            currKeys_not_match_pretrained = []
            for k, v in curr_dict.items():
                if k not in pretrained_dict.keys():
                    currKeys_not_match_pretrained.append(k)
                elif v.shape != pretrained_dict[k].shape:
                    currKeys_not_match_pretrained.append(k)
            logger.warning("Warning!!! These keys exists in pretrained model but not in curr model: {}".format(pretrainedKeys_not_match_curr))
            logger.warning("Warning!!! These keys exists in curr model but not in pretrained model: {}".format(currKeys_not_match_pretrained))

        ckpt_dict = torch.load(checkpoint_path, *args, **kwargs)
        if self.body_agent is not None:
            if "body_state_dict" in ckpt_dict:
                body_state_dict = ckpt_dict["body_state_dict"]
            elif "state_dict" in ckpt_dict:
                body_state_dict = ckpt_dict["state_dict"]
            else:
                raise NotImplementedError("Can not find body state_dict!")
            _check_state_dict_info(body_state_dict, self.body_agent.state_dict())
            self.body_agent.load_state_dict(body_state_dict)

        if self.camera_agent is not None:
            _check_state_dict_info(ckpt_dict["camera_state_dict"], self.camera_agent.state_dict())
            self.camera_agent.load_state_dict(ckpt_dict["camera_state_dict"], strict = False)

        if "optimizer" in ckpt_dict:
            self.agent.optimizer.load_state_dict(ckpt_dict["optimizer"])

        return ckpt_dict

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", "raw_metrics"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results
    
    def _collect_rollout_step(
        self, batch, body_rollouts, camera_rollouts, current_episode_reward,
        current_episode_reward_nav, current_episode_reward_exp, \
        current_episode_reward_exp_area, current_episode_reward_exp_he,\
        running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            
            # create body_AC_input
            step_observation = batch
            time_steps = torch.tensor(self.envs.call(['get_time_step'] * self.config.NUM_PROCESSES)).unsqueeze(-1)   # (num_process, 1)
            policy_input = self.config.body_policy_inputs + self.config.camera_policy_inputs

            
            if self.config.actor_critic.body_AC == "slam":
                multi_outputs = self.envs.call(['get_relative_goal'] * self.config.NUM_PROCESSES)
                relative_goal, body_masks,  planning_path, collision_map, local_goals = concat_planning_outputs(multi_outputs)

                step_observation['relative_goal'] = relative_goal
                step_observation['body_masks'] = body_masks
            (
                body_values,
                body_actions,
                body_actions_log_probs,
                body_recurrent_hidden_states,
            )  = self.body_AC.act(
                step_observation,
                body_rollouts.recurrent_hidden_states[body_rollouts.step],
                body_rollouts.prev_actions[body_rollouts.step],
                body_rollouts.masks[body_rollouts.step],
                )

            if "he_exp_area" in policy_input or "he_relative_angle" in policy_input:
                multi_camera_outputs = self.envs.call(['get_heuristic_goal'] * self.config.NUM_PROCESSES, ([{'curr_body_action': a.item()} for a in body_actions.cpu()]) + \
                                                                                                            ([{'curr_body_action': 0}]) * (self.config.NUM_PROCESSES-len(body_actions.cpu())) )
              
                heuristic_goal, circle_points, one_hot_vector, dis_list = zip(*multi_camera_outputs)                
                step_observation['one_hot_vector'] = torch.tensor(one_hot_vector, dtype=torch.float, device=self.device)
                step_observation['heuristic_goal'] = torch.tensor(heuristic_goal).squeeze(1).to(self.device)

            # create camera_AC_input
            step_observation['curr_body_action'] = body_actions
            camera_values ,camera_actions, camera_actions_log_probs, camera_recurrent_hidden_states = self.camera_AC.act(
                    step_observation,
                    camera_rollouts.recurrent_hidden_states[camera_rollouts.step],
                    camera_rollouts.prev_actions[camera_rollouts.step],
                    camera_rollouts.masks[camera_rollouts.step],
                )
            actions = joint_action(body_actions, camera_actions.to(self.device), self.config.actor_critic, time_steps)
            # self.final_actions[body_rollouts.step].copy_(actions)
            self.final_actions[camera_rollouts.step].copy_(actions)

        pth_time += time.time() - t_sample_action
        t_step_env = time.time()

        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        #额外存储一些变量进rollouts
        batch['curr_body_action'] = step_observation['curr_body_action']
        if "one_hot_vector" in step_observation:
            batch['one_hot_vector'] = step_observation['one_hot_vector']
        if "heuristic_goal" in step_observation:
            batch['heuristic_goal'] = step_observation['heuristic_goal']
        if "all_points_infos" in step_observation:
            batch['all_points_infos'] = step_observation['all_points_infos']
        rewards = torch.tensor(rewards, dtype=torch.float, device=current_episode_reward.device)
        rewards_nav, rewards_exp = rewards[:, [0]], rewards[:, [1]]
        reward_exp_area, reward_exp_he = rewards[:, [2]], rewards[:, [3]]
        rewards = rewards_nav + rewards_exp

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward

        current_episode_reward_nav += rewards_nav
        running_episode_stats["reward_nav"] += (1 - masks) * current_episode_reward_nav
        current_episode_reward_exp += rewards_exp
        running_episode_stats["reward_exp"] += (1 - masks) * current_episode_reward_exp
        current_episode_reward_exp_area += reward_exp_area
        running_episode_stats["reward_exp_area"] += (1 - masks) * current_episode_reward_exp_area
        current_episode_reward_exp_he += reward_exp_he
        running_episode_stats["reward_exp_he"] += (1 - masks) * current_episode_reward_exp_he

        running_episode_stats["count"] += 1 - masks
        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v

        current_episode_reward *= masks
        current_episode_reward_exp *= masks
        current_episode_reward_exp_area *= masks
        current_episode_reward_exp_he *= masks
        current_episode_reward_nav *= masks

        if self.body_agent is not None:
            body_rollouts.insert(
                batch,
                body_recurrent_hidden_states,
                body_actions,
                body_actions_log_probs,
                body_values,
                rewards_nav,
                masks,
            )
        if self.camera_agent is not None:
            camera_rollouts.insert(
                batch,
                camera_recurrent_hidden_states,
                camera_actions,
                camera_actions_log_probs,
                camera_values,
                rewards_exp,
                masks,
            )

        pth_time += time.time() - t_update_stats

        return batch, pth_time, env_time, self.envs.num_envs

    def _update_agents(self, ppo_cfg, body_rollouts, camera_rollouts, body_agent, camera_agent):
        def _update_agent(ppo_cfg, rollouts, agent):
            with torch.no_grad():
                last_observation = {
                    k: v[rollouts.step] for k, v in rollouts.observations.items()   # TODO:这里拿到的的curr_body_action是全零（还未被赋值），因为最后一个rgbd刚被存进来，对应的路径规划还没出来
                }
                next_value = agent.actor_critic.get_value(
                    last_observation,
                    rollouts.recurrent_hidden_states[rollouts.step],
                    rollouts.prev_actions[rollouts.step],
                    rollouts.masks[rollouts.step],
                ).detach()

            rollouts.compute_returns(
                next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
            )

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()

            return (
                value_loss,
                action_loss,
                dist_entropy,
            )

        t_update_model = time.time()
        b_value_loss = b_action_loss = b_dist_entropy = c_value_loss = c_action_loss = c_dist_entropy = None
        if body_agent is not None:
            b_value_loss, b_action_loss, b_dist_entropy = _update_agent(ppo_cfg, body_rollouts, body_agent)
        if camera_agent is not None:
            c_value_loss, c_action_loss, c_dist_entropy = _update_agent(ppo_cfg, camera_rollouts, camera_agent)
        return (
            time.time() - t_update_model,
            b_value_loss, b_action_loss, b_dist_entropy,
            c_value_loss, c_action_loss, c_dist_entropy
        )


    # @profile
    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        logger.info(f"config: {self.config}")
        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )
        logger.info(f"construct_envs finished")

        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(self.config)

        count_steps = 0
        count_steps_start = 0
        count_checkpoints = 0
        count_update = 0
        if len(os.listdir(self.config.CHECKPOINT_FOLDER)) != 0:
            dir_list = sorted(os.listdir(self.config.CHECKPOINT_FOLDER), key=lambda x: os.path.getmtime(os.path.join(self.config.CHECKPOINT_FOLDER, x)))
            ckpt_file = os.path.join(self.config.CHECKPOINT_FOLDER, dir_list[-1])   # load the last saved ckpt
            previous_model = self.load_checkpoint(ckpt_file, map_location="cpu")
            logger.info("Loading previous checkpoint:%s"%ckpt_file)

            prev_n_process = previous_model["config"].NUM_PROCESSES
            count_steps = int(os.path.basename(ckpt_file).split("_")[-1][:-5]) * 1000
            count_checkpoints = int(os.path.basename(ckpt_file).split("_")[0][5:])
            count_update = int(count_steps / prev_n_process / ppo_cfg.num_steps)
            count_steps_start = count_steps

        body_rollouts = RolloutStorageOracle(
            self.config,
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
        )
        camera_rollouts = RolloutStorageOracle(
            self.config,
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
        )
        if self.body_agent is not None:
            body_rollouts.to(self.device)
        if self.camera_agent is not None:
            camera_rollouts.to(self.device)
        self.final_actions = torch.zeros(ppo_cfg.num_steps, self.envs.num_envs, 1)
        self.final_actions = self.final_actions.to(self.device)
        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        for sensor in body_rollouts.observations:
            if sensor != "curr_body_action" and sensor != "heuristic_goal" and sensor != "one_hot_vector" \
                and sensor != "all_points_infos":
                if self.body_AC is not None:
                    body_rollouts.observations[sensor][0].copy_(batch[sensor])
                if self.camera_AC is not None:
                    camera_rollouts.observations[sensor][0].copy_(batch[sensor])

        # # batch and observations may contain shared PyTorch CUDA
        # # tensors.  We must explicitly clear them here otherwise
        # # they will be kept in memory for the entire duration of training!
        # batch = None  # TODO: 不知道会不会导致gpu内存占用上升
        # observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        current_episode_reward_nav = torch.zeros(self.envs.num_envs, 1)
        current_episode_reward_exp = torch.zeros(self.envs.num_envs, 1)
        current_episode_reward_exp_area = torch.zeros(self.envs.num_envs, 1)
        current_episode_reward_exp_he = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
            reward_nav=torch.zeros(self.envs.num_envs, 1),
            reward_exp=torch.zeros(self.envs.num_envs, 1),
            reward_exp_area=torch.zeros(self.envs.num_envs, 1),
            reward_exp_he=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0

        if self.body_agent is not None:
            body_lr_scheduler = LambdaLR(
                optimizer=self.body_agent.optimizer,
                lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
            )
        if self.camera_agent is not None:
            camera_lr_scheduler = LambdaLR(
                optimizer=self.camera_agent.optimizer,
                lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
            )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            if ppo_cfg.use_linear_lr_decay:
                for i in range(count_update):
                    if self.body_agent is not None:
                        body_lr_scheduler.step()
                    if self.camera_agent is not None:
                        camera_lr_scheduler.step()
            for update in range(count_update, self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    if self.body_agent is not None:
                        body_lr_scheduler.step()
                    if self.camera_agent is not None:
                        camera_lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    if self.body_agent is not None:
                        self.body_agent.clip_param = ppo_cfg.clip_param * linear_decay(
                            update, self.config.NUM_UPDATES
                        )
                    if self.camera_agent is not None:
                        self.camera_agent.clip_param = ppo_cfg.clip_param * linear_decay(
                            update, self.config.NUM_UPDATES
                        )

                for step in range(ppo_cfg.num_steps):
                    # print("collecting one step...")
                    (
                        batch,
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(
                        batch, body_rollouts, camera_rollouts, current_episode_reward,
                        current_episode_reward_nav, current_episode_reward_exp, \
                        current_episode_reward_exp_area,current_episode_reward_exp_he,\
                        running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                # update
                (
                    delta_pth_time,
                    b_value_loss, b_action_loss, b_dist_entropy,
                    c_value_loss, c_action_loss, c_dist_entropy
                ) = self._update_agents(ppo_cfg, body_rollouts, camera_rollouts, self.body_agent, self.camera_agent)
                pth_time += delta_pth_time
                if self.config.actor_critic.camera_AC == "rl" and "aux_task" in self.config.camera_policy_inputs:
                    aux_loss = self._update_aux(camera_rollouts)    # TODO: 写成统一格式可同时处理body_policy和camera_policy
                    writer.add_scalar(
                    "train/aux_loss", aux_loss.item(), count_steps
                    )

                # logging
                for k, v in running_episode_stats.items():
                    window_episode_stats[k].append(v.clone())

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "train/reward", deltas["reward"] / deltas["count"], count_steps
                )
                writer.add_scalar(
                    "train/reward_nav", deltas["reward_nav"] / deltas["count"], count_steps
                )
                writer.add_scalar(
                    "train/reward_exp", deltas["reward_exp"] / deltas["count"], count_steps
                )
                writer.add_scalar(
                    "train/reward_exp_area", deltas["reward_exp_area"] / deltas["count"], count_steps
                )
                writer.add_scalar(
                    "train/reward_exp_he", deltas["reward_exp_he"] / deltas["count"], count_steps
                )

                # writer.add_scalar(
                #     "train/body_learning_rate", body_lr_scheduler._last_lr[0], count_steps
                # )

                total_actions = body_rollouts.actions.shape[0] * body_rollouts.actions.shape[1] # TODO: ???
                total_found_actions = int(
                    torch.sum(self.final_actions == 0).cpu().numpy())
                total_BodyForwardCameraNone_actions = int(
                    torch.sum(self.final_actions == 1).cpu().numpy())
                total_BodyForwardCameraLeft_actions = int(
                    torch.sum(self.final_actions == 2).cpu().numpy())
                total_BodyForwardCameraRight_actions = int(
                    torch.sum(self.final_actions == 3).cpu().numpy())
                total_BodyLeftCameraNone_actions = int(
                    torch.sum(self.final_actions == 4).cpu().numpy())
                total_BodyLeftCameraLeft_actions = int(
                    torch.sum(self.final_actions == 5).cpu().numpy())
                total_BodyLeftCameraRight_actions = int(
                    torch.sum(self.final_actions == 6).cpu().numpy())
                total_BodyRightCameraNone_actions = int(
                    torch.sum(self.final_actions == 7).cpu().numpy())
                total_BodyRightCameraLeft_actions = int(
                    torch.sum(self.final_actions == 8).cpu().numpy())
                total_BodyRightCameraRight_actions = int(
                    torch.sum(self.final_actions == 9).cpu().numpy())
                assert total_actions == (
                    total_found_actions + total_BodyForwardCameraNone_actions + total_BodyForwardCameraLeft_actions +
                    total_BodyForwardCameraRight_actions + total_BodyLeftCameraNone_actions + total_BodyLeftCameraLeft_actions +
                    total_BodyLeftCameraRight_actions + total_BodyRightCameraNone_actions + total_BodyRightCameraLeft_actions +
                    total_BodyRightCameraRight_actions
                )
                writer.add_scalar(
                    "train_actions_prob/found_action_prob",
                    total_found_actions / total_actions, count_steps
                )
                writer.add_scalar(
                    "train_actions_prob/BodyForwardCameraNone_prob",
                    total_BodyForwardCameraNone_actions / total_actions, count_steps
                )
                writer.add_scalar(
                    "train_actions_prob/BodyForwardCameraLeft_prob",
                    total_BodyForwardCameraLeft_actions / total_actions, count_steps
                )
                writer.add_scalar(
                    "train_actions_prob/BodyForwardCameraRight_prob",
                    total_BodyForwardCameraRight_actions / total_actions, count_steps
                )
                writer.add_scalar(
                    "train_actions_prob/BodyLeftCameraNone_prob",
                    total_BodyLeftCameraNone_actions / total_actions, count_steps
                )
                writer.add_scalar(
                    "train_actions_prob/BodyLeftCameraLeft_prob",
                    total_BodyLeftCameraLeft_actions / total_actions, count_steps
                )
                writer.add_scalar(
                    "train_actions_prob/BodyLeftCameraRight_prob",
                    total_BodyLeftCameraRight_actions / total_actions, count_steps
                )
                writer.add_scalar(
                    "train_actions_prob/BodyRightCameraNone_prob",
                    total_BodyRightCameraNone_actions / total_actions, count_steps
                )
                writer.add_scalar(
                    "train_actions_prob/BodyRightCameraLeft_prob",
                    total_BodyRightCameraLeft_actions / total_actions, count_steps
                )
                writer.add_scalar(
                    "train_actions_prob/BodyRightCameraRight_prob",
                    total_BodyLeftCameraRight_actions / total_actions, count_steps
                )

                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }

                if len(metrics) > 0:
                    writer.add_scalar("metrics/distance_to_currgoal", metrics["distance_to_currgoal"], count_steps)
                    writer.add_scalar("metrics/success", metrics["success"], count_steps)
                    writer.add_scalar("metrics/sub_success", metrics["sub_success"], count_steps)
                    writer.add_scalar("metrics/episode_length", metrics["episode_length"], count_steps)
                    writer.add_scalar("metrics/distance_to_multi_goal", metrics["distance_to_multi_goal"], count_steps)
                    writer.add_scalar("metrics/percentage_success", metrics["percentage_success"], count_steps)
                    writer.add_scalar("metrics/percentage_spl", metrics["pspl"], count_steps)
                    writer.add_scalar("metrics/multigoal_spl", metrics["mspl"], count_steps)

                if self.body_agent is not None:
                    writer.add_scalar("train/body_losses_value", b_value_loss, count_steps)
                    writer.add_scalar("train/body_losses_policy", b_action_loss, count_steps)
                if self.camera_agent is not None:
                    writer.add_scalar("train/camera_losses_value", c_value_loss, count_steps)
                    writer.add_scalar("train/camera_losses_policy", c_action_loss, count_steps)

                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, (count_steps - count_steps_start) / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}_{int(count_steps/1000)}k.pth", dict(step=count_steps)
                    )
                    count_checkpoints += 1

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = {}
        config = self.config.clone()

        if os.path.isfile(checkpoint_path):
            ckpt_dict = torch.load(checkpoint_path, map_location="cpu")
            if self.config.EVAL.USE_CKPT_CONFIG:
                config = self._setup_eval_config(ckpt_dict["config"])

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()
        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            # config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS") # ICML rebuttal
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("VIS_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TIME_STEP")
            config.TASK_CONFIG.VIDEO_OPTION = self.config.VIDEO_OPTION
            config.freeze()
            video_num = self.config.VIDEO_NUM

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self._setup_actor_critic_agent(self.config, type="test")

        if os.path.isfile(checkpoint_path):
            self.load_checkpoint(checkpoint_path, map_location="cpu")

        body_recurrent_hidden_states = None
        camera_recurrent_hidden_states = None
        if self.body_agent is not None:
            body_recurrent_hidden_states = torch.zeros(
                self.body_AC.net.num_recurrent_layers,
                self.config.NUM_PROCESSES,
                ppo_cfg.hidden_size,
                device=self.device,
            )
        if self.camera_agent is not None:
            camera_recurrent_hidden_states = torch.zeros(
                self.camera_AC.net.num_recurrent_layers,
                self.config.NUM_PROCESSES,
                ppo_cfg.hidden_size,
                device=self.device,
            )

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        if config.TASK_CONFIG.BODY_TRAIN_TYPE == 'slam':
            found_time_info = self.envs.call(['get_searching_info'] * self.config.NUM_PROCESSES)
        self.FP=0
        self.TP=0
        self.FN=0
        self.TN=0
        self.total_neg = 0
        self.total_pos = 0

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        current_episode_reward_exp_area = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        current_episode_reward_exp_he = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        prev_body_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        prev_camera_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )

        stats_episodes = dict()  # dict of dicts that stores stats per episode
        raw_metrics_episodes = dict()

        sup_vis = SupVis(self.config, crop_size=80)
        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]

        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        pbar = tqdm.tqdm(total=sum(self.envs.number_of_episodes))
        self.body_AC.eval()
        self.camera_AC.eval()
        t_start = time.time()
        count_steps = 0
        while (
            len(stats_episodes) < sum(self.envs.number_of_episodes)
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            # 通过两个AC分别决定body和camera动作
            with torch.no_grad():
                time_steps = torch.tensor(self.envs.call(['get_time_step'] * self.config.NUM_PROCESSES)).unsqueeze(-1)   # (num_process, 1)
                policy_input = self.config.body_policy_inputs + self.config.camera_policy_inputs

                if self.config.actor_critic.body_AC == "slam":
                    multi_outputs = self.envs.call(['get_relative_goal'] * self.config.NUM_PROCESSES)
                    relative_goal, body_masks,  planning_path, collision_map, local_goals = concat_planning_outputs(multi_outputs)
                    batch['relative_goal'] = relative_goal
                    batch['body_masks'] = body_masks


                _, body_actions, _, body_recurrent_hidden_states, = self.body_AC.act(
                    batch,
                    body_recurrent_hidden_states,
                    prev_body_actions,
                    not_done_masks,
                    deterministic=False,
                )
                batch['curr_body_action'] = body_actions

                if "he_exp_area" in policy_input or "he_relative_angle" in policy_input \
                    or self.config.actor_critic.camera_AC == 'he':
                    multi_camera_outputs = self.envs.call(['get_heuristic_goal'] * self.config.NUM_PROCESSES, ([{'curr_body_action': a.item()} for a in body_actions.cpu()]) + \
                                                                                                              ([{'curr_body_action': 0}]) * (self.config.NUM_PROCESSES-len(body_actions.cpu())) )

                    heuristic_goal, circle_points, one_hot_vector, dis_list = zip(*multi_camera_outputs)
                    batch['one_hot_vector'] = torch.tensor(one_hot_vector, dtype=torch.float, device=self.device)
                    batch['heuristic_goal'] = torch.tensor(heuristic_goal).squeeze(1).to(self.device)

                _, camera_actions, _, camera_recurrent_hidden_states = self.camera_AC.act(
                    batch,
                    camera_recurrent_hidden_states,
                    prev_camera_actions,
                    not_done_masks,
                    deterministic = False
                )

                if config.test_new_baseline:
                    defrosted_condition = self.envs.call(['record_action'] * self.config.NUM_PROCESSES, ([{'curr_camera_action': a.item()} for a in camera_actions.cpu()]) + \
                                                                                            ([{'curr_camera_action': 0}]) * (self.config.NUM_PROCESSES-len(camera_actions.cpu())) )
                    test_new_baseline = np.array(defrosted_condition)
                else:
                    test_new_baseline = None

                if "aux_task" in config.camera_policy_inputs:
                    c_m = confusion_matrix(batch['one_hot_vector'].cpu().reshape(-1), \
                        (self.camera_AC.net.Aux_Task(batch['semMap']).cpu()>0.5).reshape(-1)+0, labels=[0,1])
                    # FP,TP,FN,TN = compute_aux_acc(self.camera_AC.net.Aux_Task(batch['semMap']), batch['one_hot_vector'])
                    FP,TP,FN,TN = c_m[1,0],c_m[1,1],c_m[0,1],c_m[0,0]
                    self.FP += FP
                    self.TP += TP
                    self.FN += FN
                    self.TN += TN
                    self.total_neg += (FN+TN)
                    self.total_pos += (FP+TP)
        
                try:
                    del batch['relative_goal']
                    del batch['body_masks']
                    del batch['heuristic_goal']
                    del batch['all_points_infos']
                except:
                    pass

                prev_body_actions.copy_(body_actions)               
                prev_camera_actions.copy_(camera_actions)               
                actions = joint_action(body_actions, camera_actions.to(self.device), self.config.actor_critic, time_steps, test_new_baseline)

            outputs = self.envs.step([a[0].item() for a in actions])

            if self.config.TASK_CONFIG.TASK.DEPTH_FOW_MAP.USE_NONORACLE_MAP and len(self.config.VIDEO_OPTION) > 0:
                NonOracle_pt = self.envs.call(['get_NonOracle_pt'] * self.config.NUM_PROCESSES)

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            if config.TASK_CONFIG.BODY_TRAIN_TYPE == 'slam':
                found_time_info = self.envs.call(['get_searching_info'] * self.config.NUM_PROCESSES)

            batch = batch_obs(observations, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float, device=current_episode_reward.device)
            rewards_nav, rewards_exp = rewards[:, [0]], rewards[:, [1]]
            reward_exp_area, reward_exp_he = rewards[:, [2]], rewards[:, [3]]
            rewards = rewards_nav + rewards_exp

            

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            # rewards = torch.tensor(
            #     rewards, dtype=torch.float, device=self.device
            # ).unsqueeze(1)
            current_episode_reward += rewards
            current_episode_reward_exp_area += reward_exp_area
            current_episode_reward_exp_he += reward_exp_he
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats["reward_exp_area"] = current_episode_reward_exp_area[i].item()
                    episode_stats["reward_exp_he"] = current_episode_reward_exp_he[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    current_episode_reward_exp_area[i] = 0
                    current_episode_reward_exp_he[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    
                    if config.TASK_CONFIG.BODY_TRAIN_TYPE == 'slam':
                        infos[i]["raw_metrics"]['found_time_info'] = found_time_info[i]
                    raw_metrics_episodes[
                        current_episodes[i].scene_id + '.' +
                        current_episodes[i].episode_id
                    ] = infos[i]["raw_metrics"]

                    if len(self.config.VIDEO_OPTION) > 0 and video_num > 0:
                        frame = observations_to_image(observations[i], infos[i], actions[i].cpu().numpy())
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )
                        # cv2.imwrite(config.VIDEO_DIR + '/' + current_episodes[i].episode_id + '.jpg', rgb_frames[i][-1])

                        rgb_frames[i] = []
                        video_num -= 1

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0 and video_num > 0:
                    frame = observations_to_image(observations[i], infos[i], actions[i].cpu().numpy())
                    # print('step: {}'.format(infos[i]['time_step']))
                    step_text = 'step: {}'.format(infos[i]['time_step'])
                    cv2.putText(frame, step_text, (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)

                    vis_inputs = {
                        'frame_shape': frame.shape,
                        'vis_map': infos[i]['vis_map']['map'],
                        'map_position': infos[i]['vis_map']['map_pos'],
                        'map_range': infos[i]['vis_map']['map_range'],
                        'body_heading': - observations[i]['body_heading'] + 3 * np.pi / 2,
                        'agent_heading': - observations[i]['heading'] + 3 * np.pi / 2,
                        'time_step': infos[i]['time_step'],
                        'semMap': observations[i]['semMap']
                    }
                    if self.config.TASK_CONFIG.TASK.DEPTH_FOW_MAP.USE_NONORACLE_MAP:
                        vis_inputs['NonOracle_pt'] = NonOracle_pt[i]
                    if self.config.actor_critic.body_AC == "slam":
                        vis_inputs['planning_path'] = planning_path[i]
                        vis_inputs['collision_map'] = collision_map[i].T.numpy()
                        vis_inputs['local_goals'] = local_goals[i]
                    if self.config.actor_critic.camera_AC == 'he':
                        vis_inputs['circle_points'] = circle_points[i]
                        vis_inputs['one_hot_vector'] = one_hot_vector[i]
                        vis_inputs['dis_list'] = dis_list[i]
                    save_dir = os.path.join(self.config.VIDEO_DIR, "id_{}".format(current_episodes[i].episode_id))
                    sup_frame = np.uint8(sup_vis.draw_new(vis_inputs, save_dir))    # TODO: 很耗时，优化一下
                    frame = np.concatenate([frame, sup_frame], axis=0)
                    rgb_frames[i].append(frame)
            
            (
                self.envs,
                body_recurrent_hidden_states,
                camera_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                current_episode_reward_exp_area,
                current_episode_reward_exp_he,
                prev_body_actions,
                prev_camera_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                body_recurrent_hidden_states,
                camera_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                current_episode_reward_exp_area,
                current_episode_reward_exp_he,
                prev_body_actions,
                prev_camera_actions,
                batch,
                rgb_frames,
            )

            count_steps += self.config.NUM_PROCESSES
            if count_steps % 50 == 0:   # 每隔50step打印一次信息
                if "aux_task" in config.camera_policy_inputs:
                    logger.info(f"FN:{self.FN} TN:{self.TN} FP:{self.FP} TP:{self.TP} Total_neg:{self.total_neg} Total_pos:{self.total_pos}")
                logger.info(f"fps: {count_steps/(time.time()-t_start)}")
                num_episodes = len(stats_episodes)
                if num_episodes > 0:
                    aggregated_stats = dict()
                    for stat_key in next(iter(stats_episodes.values())).keys():
                        aggregated_stats[stat_key] = (
                            sum([v[stat_key] for v in stats_episodes.values()])
                            / num_episodes
                        )
                    logger.info("Average state with {} finished episodes".format(num_episodes))
                    for k, v in aggregated_stats.items():
                        logger.info(f"Average episode {k}: {v:.4f}")

        num_episodes = len(stats_episodes)
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")



        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar("eval/average_reward", aggregated_stats["reward"],
            step_id,
        )
        writer.add_scalar("eval/average_reward_exp_area", aggregated_stats["reward_exp_area"],
            step_id,
        )
        writer.add_scalar("eval/average_reward_exp_he", aggregated_stats["reward_exp_he"],
            step_id,
        )
        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        writer.add_scalar("eval/distance_to_currgoal", metrics["distance_to_currgoal"], step_id)
        writer.add_scalar("eval/distance_to_multi_goal", metrics["distance_to_multi_goal"], step_id)
        writer.add_scalar("eval/episode_length", metrics["episode_length"], step_id)
        writer.add_scalar("eval/mspl", metrics["mspl"], step_id)
        writer.add_scalar("eval/pspl", metrics["pspl"], step_id)
        writer.add_scalar("eval/percentage_success", metrics["percentage_success"], step_id)
        writer.add_scalar("eval/success", metrics["success"], step_id)
        writer.add_scalar("eval/sub_success", metrics["sub_success"], step_id)
        writer.add_scalar("eval/pspl", metrics["pspl"], step_id)

        ##Dump metrics JSON
        if 'RAW_METRICS' in config.TASK_CONFIG.TASK.MEASUREMENTS:
            raw_metric_path = os.path.join(config.log, "metrics", os.path.basename(checkpoint_path)+".json")
            os.makedirs(os.path.dirname(raw_metric_path), exist_ok=True)
            with open(raw_metric_path, 'w') as fp:
                json.dump(raw_metrics_episodes, fp)

        self.envs.close()
