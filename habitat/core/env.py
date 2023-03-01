#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union
import pickle
from unicodedata import name
import torch
from scipy import ndimage, misc
import gym
import numpy as np
from gym.spaces.dict_space import Dict as SpaceDict
from habitat import config
import copy

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode, EpisodeIterator
from habitat.core.embodied_task import EmbodiedTask, Metrics
from habitat.core.simulator import Observations, Simulator
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks import make_task
# from habitat_baselines.common.utils import quat_from_angle_axis

import matplotlib.pyplot as plt
import cv2
from PIL import Image
from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_rotate_vector,
)
from einops import asnumpy

from Mapper.model import OccupancyAnticipator
from Mapper.Mapper import OccupancyAnticipationWrapper, Mapper, load_state_dict
from Mapper.utils import bottom_row_center_cropping, convert_gt2channel_to_gtrgb, mapper_debugger_plot

def display_sample(rgb_obs):
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    arr = [rgb_img]
    plt.imshow(rgb_img)
    plt.show()



class Env:
    r"""Fundamental environment class for `habitat`.

    :data observation_space: ``SpaceDict`` object corresponding to sensor in
        sim and task.
    :data action_space: ``gym.space`` object corresponding to valid actions.

    All the information  needed for working on embodied tasks with simulator
    is abstracted inside `Env`. Acts as a base for other derived environment
    classes. `Env` consists of three major components: ``dataset`` (`episodes`), ``simulator`` (`sim`) and `task` and connects all the three components
    together.
    """

    observation_space: SpaceDict
    action_space: SpaceDict
    number_of_episodes: Optional[int]
    _config: Config
    _dataset: Optional[Dataset]
    _episodes: List[Type[Episode]]
    _current_episode_index: Optional[int]
    _current_episode: Optional[Type[Episode]]
    _episode_iterator: Optional[Iterator]
    _sim: Simulator
    _task: EmbodiedTask
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool

    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:
        """Constructor

        :param config: config for the environment. Should contain id for
            simulator and ``task_name`` which are passed into ``make_sim`` and
            ``make_task``.
        :param dataset: reference to dataset for task instance level
            information. Can be defined as :py:`None` in which case
            ``_episodes`` should be populated from outside.
        """

        assert config.is_frozen(), (
            "Freeze the config before creating the "

            "environment, use config.freeze()."
        )
        self._config = config
        self._dataset = dataset
        self._current_episode_index = None
        if self._dataset is None and config.DATASET.TYPE:
            self._dataset = make_dataset(
                id_dataset=config.DATASET.TYPE, config=config.DATASET
            )
        self._episodes = self._dataset.episodes if self._dataset else []
        self._current_episode = None
        iter_option_dict = {
            k.lower(): v
            for k, v in config.ENVIRONMENT.ITERATOR_OPTIONS.items()
        }
        iter_option_dict["seed"] = config.SEED
        self._episode_iterator = self._dataset.get_episode_iterator(
            **iter_option_dict
        )

        # load the first scene if dataset is present
        if self._dataset:
            assert (
                len(self._dataset.episodes) > 0
            ), "dataset should have non-empty episodes list"
            self._config.defrost()
            self._config.SIMULATOR.SCENE = self._dataset.episodes[0].scene_id
            self._config.freeze()
            self.number_of_episodes = len(self._dataset.episodes)
        else:
            self.number_of_episodes = None
        self._sim = make_sim(
            id_sim=self._config.SIMULATOR.TYPE, config=self._config.SIMULATOR
        )
        self._task = make_task(
            self._config.TASK.TYPE,
            config=self._config.TASK,
            sim=self._sim,
            dataset=self._dataset,
        )
        self.observation_space = SpaceDict(
            {
                **self._sim.sensor_suite.observation_spaces.spaces,
                **self._task.sensor_suite.observation_spaces.spaces,
            }
        )
        self.action_space = self._task.action_space
        self._max_episode_seconds = (
            self._config.ENVIRONMENT.MAX_EPISODE_SECONDS
        )
        self._max_episode_steps = self._config.ENVIRONMENT.MAX_EPISODE_STEPS
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False
        self.mapCache = {}
        self.wallCache = {}
        self.copy_found_goal_info = {}
        if config.TRAINER_NAME in ["oracle", "oracle-ego"]:
            if config.DATASET.NAME == 'gibson':
                # with open('data/datasets/multiON/oracle_maps/gibson3000.pickle', 'rb') as handle:
                #     self.mapCache = pickle.load(handle)
                for scene in self._dataset.scene_ids:
                    name = scene.split('/')[-1]
                    with open(f'data/datasets/multiON/oracle_maps/gibson_3000/{name}.pkl', 'rb') as handle:
                        self.mapCache[scene] = pickle.load(handle)
            else:
                # with open('data/datasets/multiON/oracle_maps/map3000.pickle', 'rb') as handle:
                #     self.mapCache = pickle.load(handle)
                for scene in self._dataset.scene_ids:
                    name = scene.split('/')[-2]
                    with open(f'data/datasets/multiON/oracle_maps/mp3d_3000/{name}.pickle', 'rb') as handle:
                        self.mapCache[scene] = pickle.load(handle)[scene]
                    # self.wallCache[scene] = np.load(f'data/datasets/multiON/oracle_maps/wall_map/all/{name}.glb.npy')[:, :, 0]

        self.top_down_map = copy.deepcopy(self.mapCache)
        for scene in self._dataset.scene_ids:
            self.top_down_map[scene][self.mapCache[scene][:, :, 0] == 1, 0] = 0
            self.top_down_map[scene][self.mapCache[scene][:, :, 0] == 2, 0] = 1
            self.top_down_map[scene] = self.top_down_map[scene][:, :, 0]

        if config.TRAINER_NAME == "oracle-ego":
            for x, y in self.mapCache.items():
                self.mapCache[x] += 1

        self.object_to_dataset_mapping = {
            'cylinder_red': 0, 'cylinder_green': 1, 'cylinder_blue': 2,
            'cylinder_yellow': 3, 'cylinder_white': 4, 'cylinder_pink': 5,
            'cylinder_black': 6, 'cylinder_cyan': 7
        }
        self.objIndexOffset = 1 if self._config.TRAINER_NAME == "oracle" else 2

        self.patch = np.zeros([3000, 3000, 3], dtype=np.uint8)
        self.semMap_size = self._config.semMap_size # pixel
        if self._config.TASK.DEPTH_FOW_MAP.USE_NONORACLE_MAP:
            self.task.USE_NONORACLE_MAP = True
            self._init_NonOracleMapper()
        else:
            self.task.USE_NONORACLE_MAP = False

    @property
    def current_episode(self) -> Type[Episode]:
        assert self._current_episode is not None
        return self._current_episode

    @current_episode.setter
    def current_episode(self, episode: Type[Episode]) -> None:
        self._current_episode = episode

    @property
    def episode_iterator(self) -> Iterator:
        return self._episode_iterator

    @episode_iterator.setter
    def episode_iterator(self, new_iter: Iterator) -> None:
        self._episode_iterator = new_iter

    @property
    def episodes(self) -> List[Type[Episode]]:
        return self._episodes

    @episodes.setter
    def episodes(self, episodes: List[Type[Episode]]) -> None:
        assert (
            len(episodes) > 0
        ), "Environment doesn't accept empty episodes list."
        self._episodes = episodes

    @property
    def sim(self) -> Simulator:
        return self._sim

    @property
    def episode_start_time(self) -> Optional[float]:
        return self._episode_start_time

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    @property
    def task(self) -> EmbodiedTask:
        return self._task

    @property
    def _elapsed_seconds(self) -> float:
        assert (
            self._episode_start_time
        ), "Elapsed seconds requested before episode was started."
        return time.time() - self._episode_start_time

    def get_metrics(self) -> Metrics:
        return self._task.measurements.get_metrics()

    def _past_limit(self) -> bool:
        if (
            self._max_episode_steps != 0
            and self._max_episode_steps <= self._elapsed_steps
        ):
            return True
        elif (
            self._max_episode_seconds != 0
            and self._max_episode_seconds <= self._elapsed_seconds
        ):
            return True
        return False

    def _reset_stats(self) -> None:
        self._episode_start_time = time.time()
        self._elapsed_steps = 0
        self._episode_over = False


    def conv_grid(
        self,
        realworld_x,
        realworld_y,
        coordinate_min=-120.3241-1e-6,
        coordinate_max=120.0399+1e-6,
        grid_resolution=(300, 300)
    ):
        r"""Return gridworld index of realworld coordinates assuming top-left corner
        is the origin. The real world coordinates of lower left corner are
        (coordinate_min, coordinate_min) and of top right corner are
        (coordinate_max, coordinate_max)

        The real world coordinates of lower left corner are
        (coordinate_min, coordinate_min): 意思是地图的左下角那个点表示世界坐标（realworld coor）的(coordinate_min, coordinate_min)位置的内容
        地图坐标：纵坐标的负方向是x轴的正方向，指向世界坐标系的-z轴；横坐标的正方向的y轴的正方向，指向世界坐标系的+x轴
        """
        # grid_resolution = (self._config.TASK.FOW_MAP.RESOLUTION, self._config.TASK.FOW_MAP.RESOLUTION)
        grid_size = (
            (coordinate_max - coordinate_min) / grid_resolution[0],
            (coordinate_max - coordinate_min) / grid_resolution[1],
        )
        grid_x = int((coordinate_max - realworld_x) / grid_size[0])
        grid_y = int((realworld_y - coordinate_min) / grid_size[1])
        return grid_x, grid_y

    def grid_conv(
        self,
        grid_x,
        grid_y,
        coordinate_min=-120.3241-1e-6,
        coordinate_max=120.0399+1e-6,
        grid_resolution=(300, 300)
    ):
        grid_size = (
            (coordinate_max - coordinate_min) / grid_resolution[0],
            (coordinate_max - coordinate_min) / grid_resolution[1],
        )
        realworld_x = coordinate_max - grid_x * grid_size[0]
        realworld_y = coordinate_min + grid_y * grid_size[1]
        return realworld_x, realworld_y

    def reset(self) -> Observations:
        r"""Resets the environments and returns the initial observations.

        :return: initial observations from the environment.
        """
        self._reset_stats()

        assert len(self.episodes) > 0, "Episodes list is empty"

        self._current_episode = next(self._episode_iterator)
        self.reconfigure(self._config)
        
        # Remove existing objects from last episode
        for objid in self._sim._sim.get_existing_object_ids():
            self._sim._sim.remove_object(objid)

        # Insert object here
        if not self._config.TASK.DEPTH_FOW_MAP.USE_NONORACLE_MAP:
            for i in range(len(self.current_episode.goals)):
                current_goal = self.current_episode.goals[i].object_category
                dataset_index = self.object_to_dataset_mapping[current_goal]
                ind = self._sim._sim.add_object(dataset_index)
                self._sim._sim.set_translation(np.array(self.current_episode.goals[i].position), ind)

        self.action = torch.zeros(1, 1, dtype=torch.long)
        self.task.action = self.action
        self.object_category = []
        observations = self.task.reset(episode=self.current_episode)

        # ---------------------------------------------------------------------
        # SELECT USE OF VIS MAP
        # ---------------------------------------------------------------------
        if len(self._config.VIDEO_OPTION) > 0:
            self.task.timeStep = self._elapsed_steps
            self.task.visMap = torch.as_tensor(np.copy(self.mapCache[self.current_episode.scene_id]))
            self._draw_goal(self.task.visMap, (3000, 3000))
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # SELECT TYPE OF CURR MAP
        # ---------------------------------------------------------------------
        if self._config.BODY_TRAIN_TYPE == 'e2e':
            self.currMap = np.copy(cv2.resize(self.mapCache[self.current_episode.scene_id], (300, 300)))
            self._draw_goal(self.currMap, (300, 300), save_oc = True)
            self.task.sceneMap = self.currMap[:, :, 0]

        if self._config.BODY_TRAIN_TYPE == 'slam':
            self.currMap = np.copy(self.mapCache[self.current_episode.scene_id])
            self._draw_goal(self.currMap, (3000, 3000), save_oc = True)
            self.task.meshMap = self.top_down_map[self.current_episode.scene_id]
            # self.task.wallMap = self.wallCache[self.current_episode.scene_id]
        # ---------------------------------------------------------------------

        self._task.measurements.reset_measures(episode=self.current_episode, task=self.task)

        # ---------------------------------------------------------------------
        # SELECT TYPE OF EXPOSE MAP AND PATCH
        # ---------------------------------------------------------------------
        if self._config.BODY_TRAIN_TYPE == 'e2e':
            grid_resolution = (300, 300)
            if self._config.TRAINER_NAME == "oracle-ego":
                self.expose = self.task.measurements.measures["fow_map"].get_metric()[:, :, np.newaxis]
                patch = self.currMap * self.expose
            elif self._config.TRAINER_NAME == "oracle":
                patch = self.currMap

        if self._config.BODY_TRAIN_TYPE == 'slam':
            grid_resolution = (3000, 3000)
            if 'grown_fow_map' in self.task.measurements.measures.keys():
                self.expose = self.task.measurements.measures["grown_fow_map"].get_metric()[:, :, np.newaxis]
            elif 'depth_fow_map' in self.task.measurements.measures.keys():
                
                if self._config.TASK.DEPTH_FOW_MAP.USE_NONORACLE_MAP:
                    self.ego_map_gt = self.task.measurements.measures["depth_fow_map"].projected_occupancy
                    pt = self.predict_local_map(observations['rgb'])['pt']
                    pt = bottom_row_center_cropping(pt, 61)
                    fow_mask, fow_occ = self.task.measurements.measures["depth_fow_map"]._get_fow_mask(asnumpy(pt[0, :, :, :]), \
                                                                                           self._sim.get_agent_state(), True)

                    self.expose = fow_mask[:, :, np.newaxis]
                    self.currMap[:, :, 0] = fow_occ*-1+3
                    if len(self._config.VIDEO_OPTION) > 0:
                        self.NonOracle_pt = convert_gt2channel_to_gtrgb(asnumpy(pt[0,:,:,:]).transpose((1,2,0)))
                        self.task.visMap = self.currMap
                        self.task.expose = self.expose
                else:
                    self.expose = self.task.measurements.measures["depth_fow_map"].get_metric()[:, :, np.newaxis]

            elif 'wall_fow_map' in self.task.measurements.measures.keys():
                self.expose = self.task.measurements.measures["wall_fow_map"].get_metric()[:, :, np.newaxis]
            elif 'depth_more_fow_map' in self.task.measurements.measures.keys():
                self.expose = self.task.measurements.measures["depth_more_fow_map"].get_metric()[:, :, np.newaxis]
            self.expose = self.expose.numpy()
            y, x, h, w = cv2.boundingRect(self.expose[:, :, 0].astype(np.uint8))
            if h == 0:
               y, x, h, w = 1500, 1500, 20, 20  # 因为视线被墙挡住导致expose面积为0
            # patch = np.zeros_like(self.currMap)
            self.patch.fill(0)
            patch = self.patch
            patch[x: x + w, y: y + h, :] = self.currMap[x: x + w, y: y + h, :] * self.expose[x: x + w, y: y + h, :] 
            self.map_masked = patch
            self.bias = np.array([[x - 10, y - 10, w + 20, h + 20]])
            self.boundingRect = patch[x - 10: x + w + 10, y - 10: y + h + 10]
        # ---------------------------------------------------------------------

        currPix = self.conv_grid(observations["agent_position"][0], observations["agent_position"][2], grid_resolution=grid_resolution)
        patch = patch[currPix[0] - self.semMap_size: currPix[0] + self.semMap_size, currPix[1] - self.semMap_size: currPix[1] + self.semMap_size, :]
        patch = ndimage.interpolation.rotate(patch, -(observations["heading"][0] * 180 / np.pi) + 90, order=0, reshape=False)
        path = patch[self.semMap_size - int(self.semMap_size/2): self.semMap_size + int(self.semMap_size/2), self.semMap_size - int(self.semMap_size/2): self.semMap_size + int(self.semMap_size/2), :]
        observations["semMap"] = cv2.resize(path, (50, 50), interpolation=cv2.INTER_NEAREST)

        # ---------------------------------------------------------------------
        # SELECT TYPE OF OTHER
        # ---------------------------------------------------------------------
        if self._config.BODY_TRAIN_TYPE == 'e2e':
            pass

        if self._config.BODY_TRAIN_TYPE == 'slam':
            self.planning_goal = []
            for i in range(len(self.current_episode.goals)):
                goal = self.conv_grid(
                    self.current_episode.goals[i].position[0],
                    self.current_episode.goals[i].position[2],
                    grid_resolution=(3000, 3000)
                )
                self.planning_goal.append([goal[0], goal[1]])

            self.curr_object_list = np.unique(self.boundingRect[:, :, 1])
            self.prev_object_list = [0,1]
            self.increase_founding_object = len(self.curr_object_list) - len(self.prev_object_list)
            
            self.founding_goal_info = {}
            if self.increase_founding_object > 0:
                for goal in list(set(self.curr_object_list)-set(self.prev_object_list)):
                    self.founding_goal_info[self.object_category.index(goal)] = self._elapsed_steps
            self.prev_object_list = self.curr_object_list
            curr_goal_category = self.object_category[self.task.currGoalIndex]
            self.curr_goal_info = False if curr_goal_category in self.curr_object_list else True
        # ---------------------------------------------------------------------

        self.observations = observations

        return observations

    def _draw_goal(self, map, map_resolution, save_oc = False):
        goal_size = 1
        for i in range(len(self.current_episode.goals)):
            loc0 = self.current_episode.goals[i].position[0]
            loc2 = self.current_episode.goals[i].position[2]
            grid_loc = self.conv_grid(loc0, loc2, grid_resolution=map_resolution)
            objIndexOffset = 1 if self._config.TRAINER_NAME == "oracle" else 2
            map[grid_loc[0]-goal_size:grid_loc[0]+2*goal_size, grid_loc[1]-goal_size:grid_loc[1]+2*goal_size, 1] \
                = self.object_to_dataset_mapping[self.current_episode.goals[i].object_category] + objIndexOffset
            if save_oc:
                self.object_category.append(self.object_to_dataset_mapping[
                    self.current_episode.goals[i].object_category
                ] + self.objIndexOffset)

    def _update_step_stats(self) -> None:
        self._elapsed_steps += 1
        self._episode_over = not self._task.is_episode_active
        if self._past_limit():
            self._episode_over = True

        if self.episode_iterator is not None and isinstance(
            self.episode_iterator, EpisodeIterator
        ):
            self.episode_iterator.step_taken()

    def step(
        self, action: Union[int, str, Dict[str, Any]], **kwargs
    ) -> Observations:
        r"""Perform an action in the environment and return observations.

        :param action: action (belonging to `action_space`) to be performed
            inside the environment. Action is a name or index of allowed
            task's action and action arguments (belonging to action's
            `action_space`) to support parametrized and continuous actions.
        :return: observations after taking action in environment.
        """
        assert (self._episode_start_time is not None), "Cannot call step before calling reset"
        assert (self._episode_over is False), "Episode over, call reset before calling step"

        # Support simpler interface as well
        if isinstance(action, str) or isinstance(action, (int, np.integer)):
            action = {"action": action}
        self.action = torch.tensor(action['action'], dtype=torch.long).reshape([1, 1])
        self.task.action = self.action
        observations = self.task.step(action=action, episode=self.current_episode)

        # ---------------------------------------------------------------------
        # SELECT USE OF VIS MAP
        # ---------------------------------------------------------------------
        if len(self._config.VIDEO_OPTION) > 0:
            # print("step: ", self._elapsed_steps)
            self.task.timeStep = self._elapsed_steps
        # ---------------------------------------------------------------------

        self._task.measurements.update_measures(episode=self.current_episode, action=action, task=self.task)
        if self.task.currGoalIndex >= 3:
            self.task.currGoalIndex = 2

        # ---------------------------------------------------------------------
        # SELECT TYPE OF EXPOSE MAP AND PATCH
        # ---------------------------------------------------------------------
        if self._config.BODY_TRAIN_TYPE == 'e2e':
            grid_resolution = (300, 300)
            if self._config.TRAINER_NAME == "oracle-ego":
                self.expose = self.task.measurements.measures["fow_map"].get_metric()[:, :, np.newaxis]
                patch = self.currMap * self.expose
            elif self._config.TRAINER_NAME == "oracle":
                patch = self.currMap

        if self._config.BODY_TRAIN_TYPE == 'slam':
            grid_resolution = (3000, 3000)
            if 'grown_fow_map' in self.task.measurements.measures.keys():
                self.expose = self.task.measurements.measures["grown_fow_map"].get_metric()[:, :, np.newaxis]
            elif 'depth_fow_map' in self.task.measurements.measures.keys():
                
                if self._config.TASK.DEPTH_FOW_MAP.USE_NONORACLE_MAP:
                    self.ego_map_gt = self.task.measurements.measures["depth_fow_map"].projected_occupancy

                    pt = self.predict_local_map(observations['rgb'])['pt']
                    pt = bottom_row_center_cropping(pt, 61)
                    fow_mask, fow_occ = self.task.measurements.measures["depth_fow_map"]._get_fow_mask(asnumpy(pt[0, :, :, :]), \
                                                                                           self._sim.get_agent_state(), True)

                    # mapper_debugger_plot(self.ego_map_gt,asnumpy(pt[0,:,:,:]).transpose((1,2,0)),observations['rgb'])
                    
                    self.expose = fow_mask[:, :, np.newaxis]
                    self.currMap[:, :, 0] = fow_occ*-1 + 3 # 将0变成3,1变成2

                    if len(self._config.VIDEO_OPTION) > 0:
                        self.NonOracle_pt = convert_gt2channel_to_gtrgb(asnumpy(pt[0,:,:,:]).transpose((1,2,0)))
                        self.task.visMap = self.currMap
                        self.task.expose = self.expose
                else:
                    self.expose = self.task.measurements.measures["depth_fow_map"].get_metric()[:, :, np.newaxis]

            elif 'wall_fow_map' in self.task.measurements.measures.keys():
                self.expose = self.task.measurements.measures["wall_fow_map"].get_metric()[:, :, np.newaxis]
            elif 'depth_more_fow_map' in self.task.measurements.measures.keys():
                self.expose = self.task.measurements.measures["depth_more_fow_map"].get_metric()[:, :, np.newaxis]
            self.expose = self.expose.numpy()
            y, x, h, w = cv2.boundingRect(self.expose[:, :, 0].astype(np.uint8))
            if h == 0:
               y, x, h, w = 1500, 1500, 20, 20  # 因为视线被墙挡住导致expose面积为0
            # patch = np.zeros_like(self.currMap)
            patch = self.patch
            patch[x: x + w, y: y + h, :] = self.currMap[x: x + w, y: y + h, :] * self.expose[x: x + w, y: y + h, :]
            self.map_masked = patch
            self.bias = np.array([[x - 10, y - 10, w + 20, h + 20]])
            self.boundingRect = patch[x - 10: x + w + 10, y - 10: y + h + 10]
        # ---------------------------------------------------------------------

        currPix = self.conv_grid(observations["agent_position"][0], observations["agent_position"][2], grid_resolution=grid_resolution)
        patch = patch[currPix[0] - self.semMap_size: currPix[0] + self.semMap_size, currPix[1] - self.semMap_size: currPix[1] + self.semMap_size, :]
        patch = ndimage.interpolation.rotate(patch, -(observations["heading"][0] * 180 / np.pi) + 90, order=0, reshape=False)
        path = patch[self.semMap_size - int(self.semMap_size/2): self.semMap_size + int(self.semMap_size/2), self.semMap_size - int(self.semMap_size/2): self.semMap_size + int(self.semMap_size/2), :]
        observations["semMap"] = cv2.resize(path, (50, 50), interpolation=cv2.INTER_NEAREST)

        # ---------------------------------------------------------------------
        # SELECT TYPE OF OTHER
        # ---------------------------------------------------------------------
        if self._config.BODY_TRAIN_TYPE == 'e2e':
            pass

        if self._config.BODY_TRAIN_TYPE == 'slam':
            self.curr_object_list = np.unique(self.boundingRect[:, :, 1])
            self.increase_founding_object = len(self.curr_object_list) - len(self.prev_object_list)
            

            # curr_goal_category = self.object_to_dataset_mapping[
            #     self.current_episode.goals[self.task.currGoalIndex].object_category
            # ] + self.objIndexOffset
            if self.increase_founding_object > 0:
                for goal in list(set(self.curr_object_list)-set(self.prev_object_list)):
                    if goal !=0 and goal !=1:
                        self.founding_goal_info[self.object_category.index(goal)] = self._elapsed_steps + 1
            self.copy_found_goal_info = self.founding_goal_info.copy()
            self.prev_object_list = self.curr_object_list
            curr_goal_category = self.object_category[self.task.currGoalIndex]
            self.curr_goal_info = False if curr_goal_category in self.curr_object_list else True
            # self._task.curr_goal_info = self.curr_goal_info
        # ---------------------------------------------------------------------

        # Terminates episode if wrong found is called
        if self.task.is_found_called is True and self.task.measurements.measures["sub_success"].get_metric() == 0:
            self.task._is_episode_active = False

        self._update_step_stats()
        self.observations = observations

        return observations

    def _init_NonOracleMapper(self):
        occ_cfg = self._config.OCCUPANCY_ANTICIPATOR
        occupancy_model = OccupancyAnticipator(occ_cfg)
        occupancy_model = OccupancyAnticipationWrapper(
                occupancy_model, occ_cfg.output_size, (128, 128) # 在这里通过双线性插值使得输出的local map从(2,128,128)到(2,V,V)
            )
        mapper = Mapper(self._config.MAPPER, occupancy_model)
        ckpt_stats = torch.load(self._config.MAPPER.pretrained_model_path, map_location=torch.device('cpu'))
        mapper_dict = {
                k.replace("mapper.", ""): v
                for k, v in ckpt_stats["mapper_state_dict"].items()
            }
        self.mapper = load_state_dict(mapper, mapper_dict)
        self.mapper = self.mapper.to(f'cuda:{self._config.MAPPER.GPU_ID}')
        self.mapper.eval()

    def predict_local_map(self,rgb_obs):
        mapper_inputs = {
            "rgb_at_t": torch.tensor(rgb_obs[np.newaxis, :, :, :]).float().to(f'cuda:{self._config.MAPPER.GPU_ID}'),
            "ego_map_gt_at_t": torch.tensor(self.ego_map_gt[np.newaxis, :, :, :]).float().to(f'cuda:{self._config.MAPPER.GPU_ID}'),
        }
        mapper_output = self.mapper(mapper_inputs)
        return mapper_output

    def seed(self, seed: int) -> None:
        self._sim.seed(seed)
        self._task.seed(seed)

    def reconfigure(self, config: Config) -> None:
        self._config = config

        self._config.defrost()
        self._config.SIMULATOR = self._task.overwrite_sim_config(
            self._config.SIMULATOR, self.current_episode
        )
        self._config.freeze()

        self._sim.reconfigure(self._config.SIMULATOR)

    def render(self, mode="rgb") -> np.ndarray:
        return self._sim.render(mode)

    def close(self) -> None:
        self._sim.close()


class RLEnv(gym.Env):
    r"""Reinforcement Learning (RL) environment class which subclasses ``gym.Env``.

    This is a wrapper over `Env` for RL users. To create custom RL
    environments users should subclass `RLEnv` and define the following
    methods: `get_reward_range()`, `get_reward()`, `get_done()`, `get_info()`.

    As this is a subclass of ``gym.Env``, it implements `reset()` and
    `step()`.
    """

    _env: Env

    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:
        """Constructor

        :param config: config to construct `Env`
        :param dataset: dataset to construct `Env`.
        """
        self._env = Env(config, dataset)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.reward_range = self.get_reward_range()
        self.number_of_episodes = self._env.number_of_episodes

    @property
    def habitat_env(self) -> Env:
        return self._env

    @property
    def episodes(self) -> List[Type[Episode]]:
        return self._env.episodes

    @property
    def current_episode(self) -> Type[Episode]:
        return self._env.current_episode

    @episodes.setter
    def episodes(self, episodes: List[Type[Episode]]) -> None:
        self._env.episodes = episodes

    def reset(self) -> Observations:
        return self._env.reset()

    def get_reward_range(self):
        r"""Get min, max range of reward.

        :return: :py:`[min, max]` range of reward.
        """
        raise NotImplementedError

    def get_reward(self, observations: Observations) -> Any:
        r"""Returns reward after action has been performed.

        :param observations: observations from simulator and task.
        :return: reward after performing the last action.

        This method is called inside the `step()` method.
        """
        raise NotImplementedError

    def get_done(self, observations: Observations) -> bool:
        r"""Returns boolean indicating whether episode is done after performing
        the last action.

        :param observations: observations from simulator and task.
        :return: done boolean after performing the last action.

        This method is called inside the step method.
        """
        raise NotImplementedError

    def get_info(self, observations) -> Dict[Any, Any]:
        r"""..

        :param observations: observations from simulator and task.
        :return: info after performing the last action.
        """
        raise NotImplementedError

    def step(self, *args, **kwargs) -> Tuple[Observations, Any, bool, dict]:
        r"""Perform an action in the environment.

        :return: :py:`(observations, reward, done, info)`
        """

        observations = self._env.step(*args, **kwargs)
        reward = self.get_reward(observations, **kwargs)
        done = self.get_done(observations)
        info = self.get_info(observations)

        return observations, reward, done, info

    def seed(self, seed: Optional[int] = None) -> None:
        self._env.seed(seed)

    def render(self, mode: str = "rgb") -> np.ndarray:
        return self._env.render(mode)

    def close(self) -> None:
        self._env.close()
