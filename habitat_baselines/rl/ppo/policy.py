#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Sigmoid
from habitat_baselines.common.utils import CategoricalNet, Flatten, to_grid
from habitat_baselines.rl.models.projection import Projection, RotateTensor, get_grid
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.models.simple_cnn import RGBCNNNonOracle, RGBCNNOracle, MapCNN
from habitat_baselines.rl.models.projection import Projection

import cv2
from skimage.measure import label


# Angle bucketed embedding
ANGLE_min = 0
ANGLE_max = 2*math.pi
ANGLE_count = 72
ANGLE_dim = 32
ANGLE_use_log_scale = False


class PolicyNonOracle(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        global_map,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states, global_map = self.net(
            observations, rnn_hidden_states, global_map, prev_actions, masks
        )

        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, global_map

    def get_value(self, observations, rnn_hidden_states, global_map, prev_actions, masks):
        features, _, _ = self.net(
            observations, rnn_hidden_states, global_map, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, global_map, prev_actions, masks, action
    ):
        features, rnn_hidden_states, global_map = self.net(
            observations, rnn_hidden_states, global_map, prev_actions, masks, ev=1
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class PolicyOracle(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )

        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class BaselinePolicyNonOracle(PolicyNonOracle):
    def __init__(
        self,
        batch_size,
        observation_space,
        action_space,
        goal_sensor_uuid,
        device,
        object_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
        egocentric_map_size,
        global_map_size,
        global_map_depth,
        coordinate_min,
        coordinate_max,
        hidden_size=512,
    ):
        super().__init__(
            BaselineNetNonOracle(
                batch_size,
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                device=device,
                object_category_embedding_size=object_category_embedding_size,
                previous_action_embedding_size=previous_action_embedding_size,
                use_previous_action=use_previous_action,
                egocentric_map_size=egocentric_map_size,
                global_map_size=global_map_size,
                global_map_depth=global_map_depth,
                coordinate_min=coordinate_min,
                coordinate_max=coordinate_max,
            ),
            action_space.n,
        )


class BaselinePolicyOracle(PolicyOracle):
    def __init__(
        self,
        agent_type,
        observation_space,
        action_size,
        body_action_size,
        goal_sensor_uuid,
        device,
        object_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
        hidden_size=512,
        extra_policy_inputs=[],
        config = None
    ):
        super().__init__(
            BaselineNetOracle(
                agent_type,
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                device=device,
                object_category_embedding_size=object_category_embedding_size,
                previous_action_embedding_size=previous_action_embedding_size,
                use_previous_action=use_previous_action,
                action_size=action_size,
                body_action_size=body_action_size,
                extra_policy_inputs=extra_policy_inputs,
                config = config
            ),
            action_size,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, global_map, prev_actions):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class BucketingEmbedding(nn.Module):
    def __init__(self, min_val, max_val, count, dim, use_log_scale=False):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.count = count
        self.dim = dim
        self.use_log_scale = use_log_scale
        if self.use_log_scale:
            self.min_val = torch.log2(torch.Tensor([self.min_val])).item()
            self.max_val = torch.log2(torch.Tensor([self.max_val])).item()
        self.main = nn.Embedding(count, dim)

    def forward(self, x):
        """
        x - (bs, ) values
        """
        if self.use_log_scale:
            x = torch.log2(x)
        x = self.count * (x - self.min_val) / (self.max_val - self.min_val)
        x = torch.clamp(x, 0, self.count - 1).long()
        return self.main(x)

    def get_class(self, x):
        """
        x - (bs, ) values
        """
        if self.use_log_scale:
            x = torch.log2(x)
        x = self.count * (x - self.min_val) / (self.max_val - self.min_val)
        x = torch.clamp(x, 0, self.count - 1).long()
        return x


class BaselineNetNonOracle(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, batch_size, observation_space, hidden_size, goal_sensor_uuid, device, 
        object_category_embedding_size, previous_action_embedding_size, use_previous_action,
        egocentric_map_size, global_map_size, global_map_depth, coordinate_min, coordinate_max
    ):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space.spaces[
            self.goal_sensor_uuid
        ].shape[0]
        self._hidden_size = hidden_size
        self.device = device
        self.use_previous_action = use_previous_action
        self.egocentric_map_size = egocentric_map_size
        self.global_map_size = global_map_size
        self.global_map_depth = global_map_depth

        self.visual_encoder = RGBCNNNonOracle(observation_space, hidden_size)
        self.map_encoder = MapCNN(51, 256, "non-oracle")        

        self.projection = Projection(egocentric_map_size, global_map_size, 
            device, coordinate_min, coordinate_max
        )

        self.to_grid = to_grid(global_map_size, coordinate_min, coordinate_max)
        self.rotate_tensor = RotateTensor(device)

        self.image_features_linear = nn.Linear(32 * 28 * 28, 512)

        self.flatten = Flatten()

        if self.use_previous_action:
            self.state_encoder = RNNStateEncoder(
                self._hidden_size + 256 + object_category_embedding_size + 
                previous_action_embedding_size, self._hidden_size,
            )
        else:
            self.state_encoder = RNNStateEncoder(
                (0 if self.is_blind else self._hidden_size) + object_category_embedding_size,
                self._hidden_size,   #Replace 2 by number of target categories later
            )
        self.goal_embedding = nn.Embedding(8, object_category_embedding_size)
        self.action_embedding = nn.Embedding(4, previous_action_embedding_size)
        self.full_global_map = torch.zeros(
            batch_size,
            global_map_size,
            global_map_size,
            global_map_depth,
            device=self.device,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    def forward(self, observations, rnn_hidden_states, global_map, prev_actions, masks, ev=0):
        target_encoding = self.get_target_encoding(observations)
        goal_embed = self.goal_embedding((target_encoding).type(torch.LongTensor).to(self.device)).squeeze(1)
        
        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
        # interpolated_perception_embed = F.interpolate(perception_embed, scale_factor=256./28., mode='bilinear')
        projection = self.projection.forward(perception_embed, observations['depth'] * 10, -(observations["compass"]))
        perception_embed = self.image_features_linear(self.flatten(perception_embed))
        grid_x, grid_y = self.to_grid.get_grid_coords(observations['gps'])
        # grid_x_coord, grid_y_coord = grid_x.type(torch.uint8), grid_y.type(torch.uint8)
        bs = global_map.shape[0]
        ##forward pass specific
        if ev == 0:
            self.full_global_map[:bs, :, :, :] = self.full_global_map[:bs, :, :, :] * masks.unsqueeze(1).unsqueeze(1)
            if bs != 18:
                self.full_global_map[bs:, :, :, :] = self.full_global_map[bs:, :, :, :] * 0
            if torch.cuda.is_available():
                with torch.cuda.device(1):
                    agent_view = torch.cuda.FloatTensor(bs, self.global_map_depth, self.global_map_size, self.global_map_size).fill_(0)
            else:
                agent_view = torch.FloatTensor(bs, self.global_map_depth, self.global_map_size, self.global_map_size).to(self.device).fill_(0)
            agent_view[:, :, 
                self.global_map_size//2 - math.floor(self.egocentric_map_size/2):self.global_map_size//2 + math.ceil(self.egocentric_map_size/2), 
                self.global_map_size//2 - math.floor(self.egocentric_map_size/2):self.global_map_size//2 + math.ceil(self.egocentric_map_size/2)
            ] = projection
            st_pose = torch.cat(
                [-(grid_y.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2),
                 -(grid_x.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2), 
                 observations['compass']], 
                 dim=1
            )
            rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)
            rotated = F.grid_sample(agent_view, rot_mat)
            translated = F.grid_sample(rotated, trans_mat)
            self.full_global_map[:bs, :, :, :] = torch.max(self.full_global_map[:bs, :, :, :], translated.permute(0, 2, 3, 1))
            st_pose_retrieval = torch.cat(
                [
                    (grid_y.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2),
                    (grid_x.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2),
                    torch.zeros_like(observations['compass'])
                    ],
                    dim=1
                )
            _, trans_mat_retrieval = get_grid(st_pose_retrieval, agent_view.size(), self.device)
            translated_retrieval = F.grid_sample(self.full_global_map[:bs, :, :, :].permute(0, 3, 1, 2), trans_mat_retrieval)
            translated_retrieval = translated_retrieval[:,:,
                self.global_map_size//2-math.floor(51/2):self.global_map_size//2+math.ceil(51/2), 
                self.global_map_size//2-math.floor(51/2):self.global_map_size//2+math.ceil(51/2)
            ]
            final_retrieval = self.rotate_tensor.forward(translated_retrieval, observations["compass"])

            global_map_embed = self.map_encoder(final_retrieval.permute(0, 2, 3, 1))

            if self.use_previous_action:
                action_embedding = self.action_embedding(prev_actions).squeeze(1)

            x = torch.cat((perception_embed, global_map_embed, goal_embed, action_embedding), dim = 1)
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
            return x, rnn_hidden_states, final_retrieval.permute(0, 2, 3, 1)
        else: 
            global_map = global_map * masks.unsqueeze(1).unsqueeze(1)  ##verify
            with torch.cuda.device(1):
                agent_view = torch.cuda.FloatTensor(bs, self.global_map_depth, 51, 51).fill_(0)
            agent_view[:, :, 
                51//2 - math.floor(self.egocentric_map_size/2):51//2 + math.ceil(self.egocentric_map_size/2), 
                51//2 - math.floor(self.egocentric_map_size/2):51//2 + math.ceil(self.egocentric_map_size/2)
            ] = projection
            
            final_retrieval = torch.max(global_map, agent_view.permute(0, 2, 3, 1))

            global_map_embed = self.map_encoder(final_retrieval)

            if self.use_previous_action:
                action_embedding = self.action_embedding(prev_actions).squeeze(1)

            x = torch.cat((perception_embed, global_map_embed, goal_embed, action_embedding), dim = 1)
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
            return x, rnn_hidden_states, final_retrieval.permute(0, 2, 3, 1) 
            

class BaselineNetOracle(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, agent_type, observation_space, hidden_size, goal_sensor_uuid, device, 
        object_category_embedding_size, previous_action_embedding_size, use_previous_action,
        action_size, body_action_size, extra_policy_inputs=[], config = None
    ):
        super().__init__()
        self.agent_type = agent_type
        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space.spaces[
            self.goal_sensor_uuid
        ].shape[0]
        self._hidden_size = hidden_size
        self.device = device
        self.use_previous_action = use_previous_action
        self.extra_policy_inputs = extra_policy_inputs
        self.config =config
        RNN_input_size = 0
        if "relative_angle" in self.extra_policy_inputs:
            self.angle_encoder = BucketingEmbedding(
                ANGLE_min,
                ANGLE_max,
                ANGLE_count,
                ANGLE_dim,
                ANGLE_use_log_scale,
            )
            RNN_input_size += ANGLE_dim
        if "rgbd" in self.extra_policy_inputs:
            self.visual_encoder = RGBCNNOracle(observation_space, 512)
            RNN_input_size += 512
        # map
        if "occ_map" not in self.extra_policy_inputs and "obj_map" not in self.extra_policy_inputs:
            _num_object = 8
        elif "occ_map" in self.extra_policy_inputs and "obj_map" in self.extra_policy_inputs:
            self.map_encoder = MapCNN(50, 256, "oracle")
            self.occupancy_embedding = nn.Embedding(3, 16)
            self.object_embedding = nn.Embedding(9, 16)
            _num_object = 9
            RNN_input_size += 256
        elif "occ_map" in self.extra_policy_inputs:
            self.map_encoder = MapCNN(50, 256, "oracle-ego")
            self.occupancy_embedding = nn.Embedding(4, 16)
            _num_object = 9
            RNN_input_size += 256
        elif "obj_map" in self.extra_policy_inputs:
            self.map_encoder = MapCNN(50, 256, "oracle-ego")
            self.object_embedding = nn.Embedding(10, 16)
            _num_object = 9
            RNN_input_size += 256
        if "aux_task" in self.extra_policy_inputs:

            self.Auxilary_fc = nn.Sequential(
                                             nn.Linear(256, config.circle_point_num),
                                             nn.Sigmoid()
                                            )

        # if agent_type == "oracle":
        #     self.map_encoder = MapCNN(50, 256, agent_type)
        #     self.occupancy_embedding = nn.Embedding(3, 16)
        #     self.object_embedding = nn.Embedding(9, 16)
        #     _num_object = 9
        #     RNN_input_size += 256
        # elif agent_type == "no-map":
        #     _num_object = 8
        # elif agent_type == "oracle-ego":
        #     self.map_encoder = MapCNN(50, 256, agent_type)
        #     # self.object_embedding = nn.Embedding(10, 16)
        #     _num_object = 9
        #     if "occ_map" in self.extra_policy_inputs:
        #         self.occupancy_embedding = nn.Embedding(4, 16)
        #         if "obj_map" in self.extra_policy_inputs:
        #             self.map_encoder = MapCNN(50, 256, "oracle")
        #     RNN_input_size += 256
        
        if "target" in self.extra_policy_inputs:
            self.goal_embedding = nn.Embedding(_num_object, object_category_embedding_size)
            RNN_input_size += object_category_embedding_size
        if "prev_action" in self.extra_policy_inputs:
            self.action_embedding = nn.Embedding(action_size, previous_action_embedding_size)
            RNN_input_size += previous_action_embedding_size
        if "curr_body_action" in self.extra_policy_inputs:
            self.body_action_embedding = nn.Embedding(body_action_size, previous_action_embedding_size)
            RNN_input_size += previous_action_embedding_size
        if "he_exp_area" in self.extra_policy_inputs:
            self.he_exp_area_embedding = nn.Linear(config.circle_point_num, previous_action_embedding_size) # TODO: 用参数决定
            RNN_input_size += previous_action_embedding_size
        if "he_relative_angle" in self.extra_policy_inputs:
            self.he_relative_angle_embedding = BucketingEmbedding(-math.pi, math.pi, 36, ANGLE_dim, ANGLE_use_log_scale)
            RNN_input_size += ANGLE_dim
        if "all_he_inputs" in self.extra_policy_inputs:
            self.all_he_relative_angles_embedding = BucketingEmbedding(-math.pi, math.pi, 36, ANGLE_dim, ANGLE_use_log_scale)
            self.all_he_states_embedding = nn.Embedding(2, 32)
            RNN_input_size += (ANGLE_dim + 32 +1) * config.circle_point_num

        if config.RNN2MLP:
            self.state_encoder = nn.Sequential(
                                             nn.Linear(RNN_input_size, 2*self._hidden_size),
                                             nn.ReLU(),
                                             nn.Linear(2*self._hidden_size, self._hidden_size),
                                             nn.ReLU()
                                            )
        else:
            self.state_encoder = RNNStateEncoder(
                    RNN_input_size,
                    self._hidden_size,   #Replace 2 by number of target categories later
                )
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]
   
    def Aux_Task(self, semMap):
        aux_global_map_embedding = []
        bs = semMap.shape[0]
        aux_global_map = semMap
        if "occ_map" in self.extra_policy_inputs:
            aux_global_map_embedding.append(self.occupancy_embedding(aux_global_map[:, :, :, 0].type(torch.LongTensor).to(self.device).view(-1)).view(bs, 50, 50 , -1))
        if "obj_map" in self.extra_policy_inputs:
            aux_global_map_embedding.append(self.object_embedding(aux_global_map[:, :, :, 1].type(torch.LongTensor).to(self.device).view(-1)).view(bs, 50, 50, -1))
        aux_global_map_embedding = torch.cat(aux_global_map_embedding, dim=3)
        map_embed = self.map_encoder(aux_global_map_embedding)
        aux_prediction = self.Auxilary_fc(map_embed)

        return aux_prediction
    
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        bs = observations['rgb'].shape[0]
        # target
        if "target" in self.extra_policy_inputs:
            target_encoding = self.get_target_encoding(observations)
            x = [self.goal_embedding((target_encoding).type(torch.LongTensor).to(self.device)).squeeze(1)]
        # rgbd
        if "rgbd" in self.extra_policy_inputs:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x
        # map
        if "occ_map" in self.extra_policy_inputs or "obj_map" in self.extra_policy_inputs:
            global_map_embedding = []
            global_map = observations['semMap']
            if "occ_map" in self.extra_policy_inputs:
                global_map_embedding.append(self.occupancy_embedding(global_map[:, :, :, 0].type(torch.LongTensor).to(self.device).view(-1)).view(bs, 50, 50 , -1))
            if "obj_map" in self.extra_policy_inputs:
                global_map_embedding.append(self.object_embedding(global_map[:, :, :, 1].type(torch.LongTensor).to(self.device).view(-1)).view(bs, 50, 50, -1))
            global_map_embedding = torch.cat(global_map_embedding, dim=3)
            map_embed = self.map_encoder(global_map_embedding)
            x = [map_embed] + x
        # relative angle
        if "relative_angle" in self.extra_policy_inputs:
            relative_angle = (observations['heading']-observations['body_heading']) % (2*math.pi)   # 可以理解为body_heading向正方向（左边）旋转relative_angle得到heading
            relative_angle_embed = self.angle_encoder(relative_angle).squeeze(1)
            x += [relative_angle_embed]
        # previous actions
        if "prev_action" in self.extra_policy_inputs:
            prev_actions_embed = self.action_embedding(prev_actions).squeeze(1)
            x += [prev_actions_embed]
        # body actions
        if "curr_body_action" in self.extra_policy_inputs:
            body_actions_embed = self.body_action_embedding(observations['curr_body_action']).squeeze(1)
            x += [body_actions_embed]
        # heuristic direction
        if "he_exp_area" in self.extra_policy_inputs:
            he_exp_area_embed = self.he_exp_area_embedding(observations['one_hot_vector']).squeeze(1)
            x += [he_exp_area_embed]
        # heuristic relative angle goal
        if "he_relative_angle" in self.extra_policy_inputs:
            relative_goal = observations['heuristic_goal']
            he_relative_angle = torch.atan2(relative_goal[:, 1], relative_goal[:, 0])
            # print(he_relative_angle / math.pi * 180)
            he_relative_angle_embed = self.he_relative_angle_embedding(he_relative_angle).squeeze(1)
            x += [he_relative_angle_embed]
        # Input all the heuristic informations
        if "all_he_inputs" in self.extra_policy_inputs:
            all_points = observations['all_points_infos'][:,:,0:2]
            all_points_relative_angle = torch.atan2(all_points[:, :, 1], all_points[:, :, 0])
            all_relative_angle_embed = self.all_he_relative_angles_embedding(all_points_relative_angle).squeeze(1)

            all_states = observations['all_points_infos'][:,:,2]
            all_states_embed = self.all_he_states_embedding(all_states.long())

            all_infos_embed = torch.cat((all_relative_angle_embed,all_states_embed,observations['all_points_infos'][:,:,3:4].float()), 2)
            all_infos_embed = torch.flatten(all_infos_embed,1,2)

            x += [all_infos_embed]

        x = torch.cat(x, dim=1)
        if self.config.RNN2MLP:
            x = self.state_encoder(x)
        else:
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        return x, rnn_hidden_states  


class SLAMBodyPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    def forward(self, inputs, rnn_hxs, prev_actions, masks):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        relative_goal = observations['relative_goal']
        body_masks = observations['body_masks']

        goal_xy = relative_goal
        goal_phi = torch.atan2(goal_xy[:, 1], goal_xy[:, 0]) # 反正切运算

        turn_angle = math.radians(self.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE)
        fwd_action_flag = torch.abs(goal_phi) <= round(0.5 * turn_angle, 4)
        turn_left_flag = ~fwd_action_flag & (goal_phi < 0)
        turn_right_flag = ~fwd_action_flag & (goal_phi > 0)

        action = torch.zeros_like(goal_xy)[:, 0:1]
        action[fwd_action_flag] = 1
        action[turn_left_flag] = 2
        action[turn_right_flag] = 3
        action = action.long()

        action = action.to(self.device) * body_masks.to(self.device)

        return None, action, None, rnn_hidden_states

    def act_old(self, inputs, rnn_hxs, prev_actions, masks, deterministic=False):
        goal_xy = inputs["goal_at_t"]
        goal_phi = torch.atan2(goal_xy[:, 1], goal_xy[:, 0]) # 反正切运算

        turn_angle = math.radians(self.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE)
        fwd_action_flag = torch.abs(goal_phi) <= round(0.5 * turn_angle, 4)
        turn_left_flag = ~fwd_action_flag & (goal_phi < 0)
        turn_right_flag = ~fwd_action_flag & (goal_phi > 0)

        action = torch.zeros_like(goal_xy)[:, 0:1]
        action[fwd_action_flag] = 1
        action[turn_left_flag] = 2
        action[turn_right_flag] = 3
        action = action.long()

        return None, action, None, None

    def load_state_dict(self, *args, **kwargs):
        pass




class NoneCameraPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.TASK_CONFIG.SIMULATOR.CAMERA_TURN_ANGLE == config.TASK_CONFIG.SIMULATOR.TURN_ANGLE

    def forward(self, inputs, rnn_hxs, prev_actions, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, prev_actions, masks, deterministic=False):
        action = 666    # 代表camera和body相同动作
        action = torch.ones(inputs["gps"][:, 0:1].shape) * 666
        return None, action, None, rnn_hxs

    def load_state_dict(self, *args, **kwargs):
        pass


class RandomCameraPolicy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, rnn_hxs, prev_actions, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, prev_actions, masks, deterministic=False):
        num_actions = 3
        action = torch.randint(0, 3, inputs["gps"][:, 0:1].shape)
        return None, action, None, rnn_hxs

    def load_state_dict(self, *args, **kwargs):
        pass


class OnlyLeftCameraPolicy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, rnn_hxs, prev_actions, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, prev_actions, masks, deterministic=False):
        num_actions = 3
        action = torch.randint(0, 3, inputs["gps"][:, 0:1].shape)
        action = action * 0 + 1
        return None, action, None, rnn_hxs

    def load_state_dict(self, *args, **kwargs):
        pass


class HeuristicCameraPolicyPast(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = np.ones((3, 3), np.float32)

    def largest_connect_component(self, bw_img, ):
        labeled_img, num = label(bw_img, connectivity=1, background=0, return_num=True)
        max_label = 0
        max_num = 0
        for label_id in range(1, num + 1):  # 这里从1开始，防止将背景设置为最大连通域
            if np.sum(labeled_img == label_id) > max_num:
                max_num = np.sum(labeled_img == label_id)
                max_label = label_id
        lcc = (labeled_img == max_label)
        return lcc

    def forward(self, inputs, rnn_hxs, prev_actions, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, prev_actions, masks, deterministic=False):
        action = torch.zeros(inputs['heading'].shape, dtype=torch.int64)
        best_dis = np.inf

        for num_env in range(inputs['semMap'].shape[0]):
            input_map = inputs['semMap'][num_env, :, :, 0].cpu().numpy()

            free = (input_map == 3).astype(np.uint8)
            free_edge = cv2.Canny(free, 1, 1)
            free_edge_dilate = cv2.dilate(free_edge, self.kernel, iterations=1)

            explore = (input_map == 0).astype(np.uint8)
            explore_edge = cv2.Canny(explore, 1, 1)
            explore_edge_dilate = cv2.dilate(explore_edge, self.kernel, iterations=1)

            unexplore_edge = free_edge_dilate & explore_edge_dilate
            labeled_img, num = label(unexplore_edge, connectivity=1, background=0, return_num=True)
            for label_num in range(1, num + 1):
                position = np.where(labeled_img == label_num)
                avg_x = int(position[0].mean())
                avg_y = int(position[1].mean())
                # dis = (map_position[0] - avg_x)**2 + (map_position[1] - avg_y)**2
                dis = (25 - avg_x) ** 2 + (25 - avg_y) ** 2
                if dis < best_dis:
                    nearest_point = [avg_x, avg_y]
                    best_dis = dis

            phi = np.rad2deg(np.arctan2(nearest_point[1], nearest_point[0])) % 360
            if 30 < phi < 180:
                action[num_env] = 1
            elif 180 < phi < 330:
                action[num_env] = 2
            else:
                action[num_env] = 0

        return None, action, None, rnn_hxs

    def load_state_dict(self, *args, **kwargs):
        pass


class HeuristicCameraPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    def forward(self, inputs, rnn_hxs, prev_actions, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, prev_actions, masks, deterministic=False):
        relative_goal = inputs['heuristic_goal']

        goal_xy = relative_goal
        goal_phi = torch.atan2(goal_xy[:, 1], goal_xy[:, 0])

        turn_angle = math.radians(self.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE)
        fwd_action_flag = torch.abs(goal_phi) <= round(0.5 * turn_angle, 4)
        turn_left_flag = ~fwd_action_flag & (goal_phi < 0)
        turn_right_flag = ~fwd_action_flag & (goal_phi > 0)

        action = torch.zeros_like(goal_xy)[:, 0:1]
        action[fwd_action_flag] = 0
        action[turn_left_flag] = 1
        action[turn_right_flag] = 2
        action = action.long()

        action = action.to(self.device)

        return None, action, None, rnn_hxs

    def load_state_dict(self, *args, **kwargs):
        pass
