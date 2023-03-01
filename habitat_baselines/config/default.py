#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import numpy as np

from habitat import get_config as get_task_config
from habitat.config import Config as CN
import os
import time
import shutil
import math

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.BASE_TASK_CONFIG_PATH = "configs/tasks/pointnav.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "ppo"
_C.ENV_NAME = "NavRLEnv"
_C.SIMULATOR_GPU_ID = 0
_C.SIMULATOR_GPU_IDS = [2,3,4,5,6,7]
_C.TORCH_GPU_ID = 0
_C.VIDEO_OPTION = ["disk", "tensorboard"]
_C.VIDEO_NUM = 1000
_C.TENSORBOARD_DIR = "tb"
_C.VIDEO_DIR = "video_dir"
_C.TEST_EPISODE_COUNT = 2
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints"  # path to ckpt or path to ckpts dir
_C.NUM_PROCESSES = 16
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.NUM_UPDATES = 10000
_C.LOG_INTERVAL = 10
_C.LOG_FILE = "train.log"
_C.CHECKPOINT_INTERVAL = 50
# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val"
_C.EVAL.USE_CKPT_CONFIG = True
# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.REWARD_MEASURE = "distance_to_currgoal"
_C.RL.SUCCESS_MEASURE = "success"
_C.RL.SUBSUCCESS_MEASURE = "sub_success"
_C.RL.SUCCESS_REWARD = 10.0
_C.RL.SLACK_REWARD = -0.01
_C.RL.OBJECT_CATEGORY_EMBEDDING_SIZE = 32
_C.RL.PREVIOUS_ACTION_EMBEDDING_SIZE = 32
_C.RL.PREVIOUS_ACTION = True
_C.RL.USE_FOUND_REWARD = False
_C.RL.USE_AREA_REWARD = True
_C.RL.HE_RELATIVE_ANGLE_REWARD_SCALE = 0.0
_C.RL.HE_RELATIVE_ANGLE_REWARD_SCALE_DECAY = False
_C.RL.FOUND_REWARD = 10.0
_C.RL.RELATIVE_ANGLE_PENALTY = 0.0
_C.RL.TURN_HEAD_PENALTY = 0.0
# =============================================================================
# Planner
# =============================================================================
_C.RL.PLANNER = CN()
_C.RL.PLANNER.nplanners = 1  # Same as the number of processes
_C.RL.PLANNER.allow_diagonal = True  # planning diagonally
# local region around the agent / goal that is set to free space when either
# are classified occupied
_C.RL.PLANNER.local_free_size = 0.25
# Assign weights to graph based on proximity to obstacles?
_C.RL.PLANNER.use_weighted_graph = True
# Weight factors
_C.RL.PLANNER.weight_scale = 4.0
_C.RL.PLANNER.weight_niters = 1
# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 16
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.lr = 7e-4
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 5
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = False
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.reward_window_size = 50
_C.RL.PPO.use_normalized_advantage = True
_C.RL.PPO.hidden_size = 512
_C.RL.PPO.action_size = 4   # 这里指的是body_AC的动作个数，如果e2e输出body和camera动作时此值为10
# -----------------------------------------------------------------------------
# MAPS
# -----------------------------------------------------------------------------
_C.RL.MAPS = CN()
_C.RL.MAPS.egocentric_map_size = 3
_C.RL.MAPS.global_map_size = 250
_C.RL.MAPS.global_map_depth = 32
_C.RL.MAPS.coordinate_min = -62.3241 - 1e-6
_C.RL.MAPS.coordinate_max = 90.0399 + 1e-6
# -----------------------------------------------------------------------------
# DECENTRALIZED DISTRIBUTED PROXIMAL POLICY OPTIMIZATION (DD-PPO)
# -----------------------------------------------------------------------------
_C.RL.DDPPO = CN()
_C.RL.DDPPO.sync_frac = 0.6
_C.RL.DDPPO.distrib_backend = "GLOO"
_C.RL.DDPPO.rnn_type = "LSTM"
_C.RL.DDPPO.num_recurrent_layers = 2
_C.RL.DDPPO.backbone = "resnet50"
_C.RL.DDPPO.pretrained_weights = "data/ddppo-models/gibson-2plus-resnet50.pth"
# Loads pretrained weights
_C.RL.DDPPO.pretrained = False
# Loads just the visual encoder backbone weights
_C.RL.DDPPO.pretrained_encoder = False
# Whether or not the visual encoder backbone will be trained
_C.RL.DDPPO.train_encoder = True
# Whether or not to reset the critic linear layer
_C.RL.DDPPO.reset_critic = True
# -----------------------------------------------------------------------------
# ORBSLAM2 BASELINE
# -----------------------------------------------------------------------------
_C.ORBSLAM2 = CN()
_C.ORBSLAM2.SLAM_VOCAB_PATH = "habitat_baselines/slambased/data/ORBvoc.txt"
_C.ORBSLAM2.SLAM_SETTINGS_PATH = (
    "habitat_baselines/slambased/data/mp3d3_small1k.yaml"
)
_C.ORBSLAM2.MAP_CELL_SIZE = 0.1
_C.ORBSLAM2.MAP_SIZE = 40
_C.ORBSLAM2.CAMERA_HEIGHT = get_task_config().SIMULATOR.DEPTH_SENSOR.POSITION[
    1
]
_C.ORBSLAM2.BETA = 100
_C.ORBSLAM2.H_OBSTACLE_MIN = 0.3 * _C.ORBSLAM2.CAMERA_HEIGHT
_C.ORBSLAM2.H_OBSTACLE_MAX = 1.0 * _C.ORBSLAM2.CAMERA_HEIGHT
_C.ORBSLAM2.D_OBSTACLE_MIN = 0.1
_C.ORBSLAM2.D_OBSTACLE_MAX = 4.0
_C.ORBSLAM2.PREPROCESS_MAP = True
_C.ORBSLAM2.MIN_PTS_IN_OBSTACLE = (
    get_task_config().SIMULATOR.DEPTH_SENSOR.WIDTH / 2.0
)
_C.ORBSLAM2.ANGLE_TH = float(np.deg2rad(15))
_C.ORBSLAM2.DIST_REACHED_TH = 0.15
_C.ORBSLAM2.NEXT_WAYPOINT_TH = 0.5
_C.ORBSLAM2.NUM_ACTIONS = 3
_C.ORBSLAM2.DIST_TO_STOP = 0.05
_C.ORBSLAM2.PLANNER_MAX_STEPS = 500
_C.ORBSLAM2.DEPTH_DENORM = get_task_config().SIMULATOR.DEPTH_SENSOR.MAX_DEPTH


# =============================================================================
# 我们的特殊设置
# =============================================================================
_C.special_requirements = []
_C.extra_policy_inputs = []
_C.ans_planner_debug = False
_C.actor_critic = CN()
_C.actor_critic.body_AC = 'e2e'    # ['e2e', 'slam'].
_C.actor_critic.camera_AC = 'none'    # ['none', 'e2e' 'random' 'onlyleft' 'he']
_C.expose_type = 'grown'  # normal, grown, depth, wall, depthmore
# 常用设置：
# (e2e, none): multion; 
# (e2e, e2e): 直接用RL输入10个动作;
# (slam, none/random/he): 通过slam决定身体，各种其他policy决定转头
_C.body_policy_inputs = ['target','rgbd','relative_angle','prev_action','obj_map']
_C.camera_policy_inputs = ['rgbd','relative_angle','prev_action','obj_map'] # 'occ_map', 'curr_body_action', 'he_exp_area', 'he_relative_angle', 'aux_task'
_C.camera_action_size = 3
_C.run_fast = False # 确定程序没有bug的时候可以用这个flag，提升1倍fps，但是初始化环境的时候很慢
_C.No_Repeat_FBE = False
_C.anchor_length = 30
_C.exp_threshold = 1.2
_C.circle_point_num = 8
_C.semMap_size = 50   # 单位为像素。默认50，即在e2e中表示50*0.8=40m内容，在slam中表示50*0.08=4m的内容（改为125表示10m的内容）
_C.sample_goal_type = "FBE"
_C.measure_exploration = False
_C.NonOracleMap = False
_C.test_new_baseline = False # 用来测试身体跟着头部转的baseline

_C.RNN2MLP = False # 用来测试身体跟着头部转的baseline


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
    model_dir: Optional[str] = None,
    run_type: Optional[str] = None,
    overwrite: bool = False,
    special_exp: bool = False,
    note=""
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    if model_dir is None:
        if run_type == "eval":
            model_dir = os.path.dirname(os.path.dirname(config.EVAL_CKPT_PATH_DIR))
        else:
            model_dir = 'data/models'
    else:
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
    if run_type == 'eval' and special_exp:
        run_times = 1
        for file in os.listdir(model_dir):
            if "sep_run_{}".format(run_type) in file:
                run_times +=1
        model_dir = os.path.join(model_dir,"sep_run_{}_{}".format(run_type, note))
        run_dir = model_dir
    else:
        run_times = 1
        for file in os.listdir(model_dir):
            if "run_{}".format(run_type) in file:
                run_times +=1
        run_dir = os.path.join(model_dir,"run_{}_{}_{}_{}".format(run_type, run_times, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()), note))

    config = make_related_dir(config, model_dir, run_dir)

    if config.TASK_CONFIG.DATASET.NAME == 'gibson':
        config.TASK_CONFIG.DATASET.DATA_PATH = "data/datasets/multiON/multinav/gibson/3_ON_same_height/{split}/{split}.json.gz"
        config.TASK_CONFIG.MAPPER.pretrained_model_path = '/mnt/cephfs/dataset/soundspaces_data/results/OA/pretrained_model/ckpt.10.pth'
        config.TASK_CONFIG.OCCUPANCY_ANTICIPATOR.GP_ANTICIPATION.unet_nsf = 64

    config = refine_config(config, run_type)
    dirs = [config.VIDEO_DIR, config.TENSORBOARD_DIR, config.CHECKPOINT_FOLDER]
    if run_type == 'train':
        if any([os.path.exists(d) for d in dirs]):
            for d in dirs:
                if os.path.exists(d):
                    print('{} exists'.format(d))
            if overwrite or input('Output directory already exists! Overwrite the folder? (y/n)') == 'y':
                for d in dirs:
                    if os.path.exists(d):
                        shutil.rmtree(d)
    config.freeze()
    return config


def make_related_dir(config, model_dir, run_dir):
    config.defrost()
    config.sh_n_codes = os.path.join(run_dir, "sh_n_codes")
    config.configs = os.path.join(run_dir, "configs")
    config.log = os.path.join(run_dir, "log")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(config.sh_n_codes, exist_ok=True)
    os.makedirs(config.configs, exist_ok=True)
    os.makedirs(config.log, exist_ok=True)

    config.VIDEO_DIR = os.path.join(run_dir, 'video_dir')
    run_dir = ""
    config.LOG_FILE = os.path.join(run_dir, '{}/train.log'.format(config.log))
    config.EVAL_LOG_FILE = os.path.join(run_dir, '{}/eval.log'.format(config.log))

    config.TENSORBOARD_DIR = os.path.join(model_dir, 'tb')
    config.CHECKPOINT_FOLDER = os.path.join(model_dir, 'data')

    return config


def refine_config(config, run_type):
    POSSIBLE_REQUIREMENT = []

    # 修改action_size和POSSIBLE_ACTIONS
    config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = [
        "FOUND",
        "BODY_FORWARD_CAMERA_NONE",
        "BODY_FORWARD_CAMERA_LEFT",
        "BODY_FORWARD_CAMERA_RIGHT",
        "BODY_LEFT_CAMERA_NONE",
        "BODY_LEFT_CAMERA_LEFT",
        "BODY_LEFT_CAMERA_RIGHT",
        "BODY_RIGHT_CAMERA_NONE",
        "BODY_RIGHT_CAMERA_LEFT",
        "BODY_RIGHT_CAMERA_RIGHT",
    ]
    if config.actor_critic.body_AC == 'e2e':
        if config.actor_critic.camera_AC == 'e2e': # e2e同时决定body和camera，输出10个动作
            config.RL.PPO.action_size = 10
    config.TASK_CONFIG.actor_critic = CN()
    config.TASK_CONFIG.actor_critic = config.actor_critic

    # 修改dataset iteration
    if config.actor_critic.body_AC == "slam" and run_type == "train":
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = 2000
        config.CHECKPOINT_INTERVAL = 50

    # 可视化时不要连续可视化同一个场景
    if len(config.VIDEO_OPTION) != 0:
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False

    # 确定输入没有写错
    for input in config.body_policy_inputs + config.camera_policy_inputs:
        assert input in ['target','rgbd','relative_angle','prev_action','obj_map', 'occ_map', \
            'curr_body_action', 'he_exp_area', 'he_relative_angle','aux_task', 'all_he_inputs']

    # 如果要惩罚转头动作，则camera policy必须知道curr body action才合理
    if config.RL.TURN_HEAD_PENALTY != 0:
        assert "curr_body_action" in config.camera_policy_inputs

    config.TASK_CONFIG.semMap_size = config.semMap_size
    config.TASK_CONFIG.VIDEO_OPTION = config.VIDEO_OPTION
    config.TASK_CONFIG.BODY_TRAIN_TYPE = config.actor_critic.body_AC
    if config.TASK_CONFIG.BODY_TRAIN_TYPE == 'e2e':
        assert config.expose_type == 'normal', print('please use normal fow map')
    if config.TASK_CONFIG.BODY_TRAIN_TYPE == 'slam':
        assert config.expose_type != 'normal', print('please do not use normal fow map')

    if config.NonOracleMap:
        config.expose_type = 'depth'
        config.TASK_CONFIG.TASK.DEPTH_FOW_MAP.USE_NONORACLE_MAP = True
        config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.HFOV = 90
        config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV = 90
        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV = 90
        config.TASK_CONFIG.TASK.DEPTH_FOW_MAP.MAP_SIZE = 63


    if len(config.special_requirements) == 0:
        return config
    else:
        for requirement in config.special_requirements:
            assert requirement in POSSIBLE_REQUIREMENT, print("Not recognized requirement:{}".format(requirement))

    return config
