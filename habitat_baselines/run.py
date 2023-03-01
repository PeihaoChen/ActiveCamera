#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
sys.path.insert(0, "")
import os
os.environ['GLOG_minloglevel']='2'
os.environ['MAGNUM_LOG']='quiet'
os.environ['MKL_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import cv2
cv2.setNumThreads(0)
import argparse
import random
import numpy as np
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config
from shlex import quote
import socket
from habitat_baselines.common.utils import save_code,save_config
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )

    parser.add_argument(
        "--agent-type",
        choices=["no-map", "oracle", "oracle-ego", "proj-neural", "obj-recog"],
        required=True,
        help="agent type: oracle, oracleego, projneural, objrecog",
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action='store_true',
        help="Modify config options from command line"
    )
    parser.add_argument(
        "--special_exp",
        default=False,
        help="Make directory for special experiment"
    )
    parser.add_argument(
        "--note",
        default="",
        help="Add extra note for running file"
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, model_dir: str, run_type: str, agent_type: str, opts=None, overwrite=False, special_exp=False, note="") -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts, model_dir, run_type, overwrite, special_exp, note)
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)

    config.defrost()
    config.TRAINER_NAME = agent_type
    config.TASK_CONFIG.TRAINER_NAME = agent_type
    config.RL.PLANNER.nplanners = config.NUM_PROCESSES
    config.freeze()

    if agent_type in ["oracle", "oracle-ego", "no-map"]:
        trainer_init = baseline_registry.get_trainer("oracle")
        config.defrost()
        config.RL.PPO.hidden_size = 512 if agent_type=="no-map" else 768
        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.5
        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 5.0
        config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = 1.5
        if agent_type == "oracle-ego":
            if config.expose_type == 'normal':
                config.TASK_CONFIG.TASK.MEASUREMENTS.insert(-1,'FOW_MAP')
            elif config.expose_type == 'grown':
                config.TASK_CONFIG.TASK.MEASUREMENTS.insert(-1,'GROWN_FOW_MAP')
            elif config.expose_type == 'depth':
                config.TASK_CONFIG.TASK.MEASUREMENTS.insert(-1,'DEPTH_FOW_MAP')
            elif config.expose_type == 'wall':
                config.TASK_CONFIG.TASK.MEASUREMENTS.insert(-1,'WALL_FOW_MAP')
            elif config.expose_type == 'depthmore':
                config.TASK_CONFIG.TASK.MEASUREMENTS.insert(-1,'DEPTH_MORE_FOW_MAP')
            config.TASK_CONFIG.TASK.MEASUREMENTS.insert(-1,'COLLISIONS')
        config.freeze()
    else:
        trainer_init = baseline_registry.get_trainer("non-oracle")
        config.defrost()
        config.RL.PPO.hidden_size = 512
        config.freeze()

    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    """保存运行时的指令为'run_{服务器名字}.sh'，您下次可以在任何地方使用 sh run_{服务器名字}.sh 运行当前实验"""
    with open(config.sh_n_codes +'/run_{}_{}_{}.sh'.format(run_type,socket.gethostname(), config.ENV_NAME), 'w') as f:
        f.write(f'cd {quote(os.getcwd())}\n')
        f.write('unzip -d {}/code {}\n'.format(config.sh_n_codes, os.path.join(config.sh_n_codes, 'code.zip')))
        f.write('cp -r -f {} {}\n'.format(os.path.join(config.sh_n_codes, 'code', '*'), quote(os.getcwd())))
        envs = ['CUDA_VISIBLE_DEVICES']
        for env in envs:
            value = os.environ.get(env, None)
            if value is not None:
                f.write(f'export {env}={quote(value)}\n')
        f.write(sys.executable + ' ' + ' '.join(quote(arg) for arg in sys.argv) + '\n')

    """保存本次运行的所有代码"""
    save_code(os.path.join(config.sh_n_codes, 'code.zip'),
              ignore_dir=['habitat-lab', 'habitat-sim', 'data', 'result', 'results'])

    """保存config"""
    save_config(config, run_type)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()


if __name__ == "__main__":
    print(os.getcwd())
    main()

    #MIN_DEPTH: 0.5
    #MAX_DEPTH: 5.0
