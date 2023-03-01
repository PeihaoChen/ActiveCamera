from habitat import config
import time
import numpy as np
import torch
import copy

from habitat.core.registry import registry
from habitat.core.embodied_task import SimulatorTaskAction

from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_sim.utils.common import *
from habitat.utils.geometry_utils import *
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)


def turn_angle(rotation_o, angle_delta):
    '''输入一个四元数rotation_o，计算旋转angle_delta（弧度制，右转为正，左为负）后得到的四元数
    '''
    theta = compute_heading_from_quaternion(
        rotation_o
    )  # counter-clockwise rotation about Y from -Z to X (???)
    thetap = theta + angle_delta
    rotation_n = compute_quaternion_from_heading(thetap)
    return rotation_n


class BaseAction(SimulatorTaskAction):

    def _get_uuid(self, *args, **kwargs) -> str:
        raise NotImplementedError

    def step(self, collided, **kwargs):
        """reference from habitat-sim/habitat_sim/simulator.py L251
        """
        self._sim._sim._num_total_frames += 1

        self._sim._sim._last_state = self._sim._sim.get_agent(0).get_state()

        # step physics by dt=1.0 / 60.0
        step_start_Time = time.time()
        # self._sim._sim.step_physics(1.0 / 60.0)  # 这个是关于对物体施加作用力，应该与我们无关
        _previous_step_time = time.time() - step_start_Time

        sim_obs = self._sim._sim.get_sensor_observations()
        sim_obs["collided"] = collided

        self._sim._prev_sim_obs = sim_obs
        observations = self._sim._sensor_suite.get_observations(sim_obs)

        return observations

    def _get_body_camera_angle(self):
        """
        获取当前body和camera角度（0~360°）
        """
        agent_state = self._sim._sim.get_agent(0).get_state()
        body_rotation = agent_state.rotation
        body_angle = compute_heading_from_quaternion(agent_state.rotation)

        sensor_name = list(agent_state.sensor_states.keys())[0]  # 第0个sensor的名字
        sensor_rotation = agent_state.sensor_states[sensor_name].rotation
        sensor_angle = compute_heading_from_quaternion(sensor_rotation)

        return np.rad2deg(body_angle), np.rad2deg(sensor_angle), body_rotation, sensor_rotation

    def _cal_camera_angle(self, target_turn_angle, limited_angle=180):
        '''限制camera与body的角度不能超过一定度数
        limited_angle: 机器人的camera和body夹角不大于limited_angle (可设置0~180)
        '''
        body_angle_new, camera_angle, body_rotation_new, camera_rotation = self._get_body_camera_angle()
        camera_rotation_new = turn_angle(camera_rotation, np.deg2rad(target_turn_angle))
        camera_angle_new = np.rad2deg(compute_heading_from_quaternion(camera_rotation_new))

        if angle_between_quaternions(body_rotation_new, camera_rotation_new) > np.deg2rad(limited_angle):
            if (camera_angle_new % 360 - body_angle_new % 360) % 360 < 180: # camera在body右边
                camera_rotation_new = turn_angle(body_rotation_new, np.pi/2)
            elif (camera_angle_new % 360 - body_angle_new % 360) % 360 >= 180: # camera在body左边
                camera_rotation_new = turn_angle(body_rotation_new, -np.pi/2)
            camera_angle_new = np.rad2deg(compute_heading_from_quaternion(camera_rotation_new))
            # print("trigger")

        # print("action_name:", self._get_uuid())
        # print("body_angle_new:{} camera_angle_new:{} relative_angle:{} ".format(body_angle_new, camera_angle_new, (camera_angle_new - body_angle_new) % 360))
        return camera_rotation_new

    def _implement_camera_action(self, camera_rotation_new):
        '''把camera设置到某个角度
        camera_angle_new: 需要设置到的角度（单位°）
        '''
        agent_state = self._sim._sim.get_agent(0).get_state()
        camera_rotation_new = camera_rotation_new
        for sensor in agent_state.sensor_states:
            agent_state.sensor_states[sensor].rotation = camera_rotation_new

        self._sim._sim.get_agent(0).set_state(agent_state)
        # self._sim._sim.get_agent(0).set_state(agent_state, infer_sensor_states=False)


@registry.register_task_action
class BodyForwardCameraNoneAction(BaseAction):

    def _get_uuid(self, *args, **kwargs) -> str:
        return "body_forward_camera_none"

    def step(self, *args, **kwargs):
        kwargs['task'].is_found_called = False
        collided = self._sim._sim.get_agent(0).act(HabitatSimActions.MOVE_FORWARD)
        a = self._cal_camera_angle(0)
        observations = super().step(collided)

        return observations


@registry.register_task_action
class BodyForwardCameraLeftAction(BaseAction):

    def _get_uuid(self, *args, **kwargs) -> str:
        return "body_forward_camera_left"

    def step(self, *args, **kwargs):
        kwargs['task'].is_found_called = False
        collided = self._sim._sim.get_agent(0).act(HabitatSimActions.MOVE_FORWARD)   # 就这一句有用，其他全是抄过来
        agent_state = self._sim._sim.get_agent(0).get_state()  # 读取当前的agent状态，根据camera的旋转需求改变这个状态，最后用set_state使agent位于新的状态

        delta = kwargs['task']._sim.config.TURN_ANGLE
        camera_delta = kwargs['task']._sim.config.CAMERA_TURN_ANGLE
        relative_angle_limit = kwargs['task']._sim.config.RELATIVE_ANGLE_LIMIT
        camera_rotation_new = self._cal_camera_angle(-camera_delta, relative_angle_limit)
        self._implement_camera_action(camera_rotation_new)

        observations = super().step(collided)

        return observations


@registry.register_task_action
class BodyForwardCameraRightAction(BaseAction):

    def _get_uuid(self, *args, **kwargs) -> str:
        return "body_forward_camera_right"

    def step(self, *args, **kwargs):
        kwargs['task'].is_found_called = False
        collided = self._sim._sim.get_agent(0).act(HabitatSimActions.MOVE_FORWARD)
        agent_state = self._sim._sim.get_agent(0).get_state()

        delta = kwargs['task']._sim.config.TURN_ANGLE
        camera_delta = kwargs['task']._sim.config.CAMERA_TURN_ANGLE
        relative_angle_limit = kwargs['task']._sim.config.RELATIVE_ANGLE_LIMIT
        camera_rotation_new = self._cal_camera_angle(camera_delta, relative_angle_limit)
        self._implement_camera_action(camera_rotation_new)

        observations = super().step(collided)

        return observations


@registry.register_task_action
class BodyLeftCameraNoneAction(BaseAction):

    def _get_uuid(self, *args, **kwargs) -> str:
        return "body_left_camera_none"

    def step(self, *args, **kwargs):
        kwargs['task'].is_found_called = False
        collided = self._sim._sim.get_agent(0).act(HabitatSimActions.TURN_LEFT)
        agent_state = self._sim._sim.get_agent(0).get_state()

        delta = kwargs['task']._sim.config.TURN_ANGLE
        camera_delta = kwargs['task']._sim.config.CAMERA_TURN_ANGLE
        relative_angle_limit = kwargs['task']._sim.config.RELATIVE_ANGLE_LIMIT
        camera_rotation_new = self._cal_camera_angle(delta, relative_angle_limit)
        self._implement_camera_action(camera_rotation_new)

        observations = super().step(collided)

        return observations


@registry.register_task_action
class BodyLeftCameraLeftAction(BaseAction):

    def _get_uuid(self, *args, **kwargs) -> str:
        return "body_left_camera_left"

    def step(self, *args, **kwargs):
        kwargs['task'].is_found_called = False
        collided = self._sim._sim.get_agent(0).act(HabitatSimActions.TURN_LEFT)
        agent_state = self._sim._sim.get_agent(0).get_state()

        delta = kwargs['task']._sim.config.TURN_ANGLE
        camera_delta = kwargs['task']._sim.config.CAMERA_TURN_ANGLE
        relative_angle_limit = kwargs['task']._sim.config.RELATIVE_ANGLE_LIMIT
        camera_rotation_new = self._cal_camera_angle(-(camera_delta - delta), relative_angle_limit)
        self._implement_camera_action(camera_rotation_new)

        observations = super().step(collided)

        return observations


@registry.register_task_action
class BodyLeftCameraRightAction(BaseAction):

    def _get_uuid(self, *args, **kwargs) -> str:
        return "body_left_camera_right"

    def step(self, *args, **kwargs):
        kwargs['task'].is_found_called = False
        collided = self._sim._sim.get_agent(0).act(HabitatSimActions.TURN_LEFT)
        agent_state = self._sim._sim.get_agent(0).get_state()

        delta = kwargs['task']._sim.config.TURN_ANGLE
        camera_delta = kwargs['task']._sim.config.CAMERA_TURN_ANGLE
        relative_angle_limit = kwargs['task']._sim.config.RELATIVE_ANGLE_LIMIT
        camera_rotation_new = self._cal_camera_angle((delta + camera_delta), relative_angle_limit)
        self._implement_camera_action(camera_rotation_new)

        observations = super().step(collided)

        return observations


@registry.register_task_action
class BodyRightCameraNoneAction(BaseAction):

    def _get_uuid(self, *args, **kwargs) -> str:
        return "body_right_camera_none"

    def step(self, *args, **kwargs):
        kwargs['task'].is_found_called = False
        collided = self._sim._sim.get_agent(0).act(HabitatSimActions.TURN_RIGHT)
        agent_state = self._sim._sim.get_agent(0).get_state()

        delta = kwargs['task']._sim.config.TURN_ANGLE
        camera_delta = kwargs['task']._sim.config.CAMERA_TURN_ANGLE
        relative_angle_limit = kwargs['task']._sim.config.RELATIVE_ANGLE_LIMIT
        camera_rotation_new = self._cal_camera_angle(-delta, relative_angle_limit)
        self._implement_camera_action(camera_rotation_new)

        observations = super().step(collided)

        return observations


@registry.register_task_action
class BodyRightCameraLeftAction(BaseAction):

    def _get_uuid(self, *args, **kwargs) -> str:
        return "body_right_camera_left"

    def step(self, *args, **kwargs):
        kwargs['task'].is_found_called = False
        collided = self._sim._sim.get_agent(0).act(HabitatSimActions.TURN_RIGHT)
        agent_state = self._sim._sim.get_agent(0).get_state()

        delta = kwargs['task']._sim.config.TURN_ANGLE
        camera_delta = kwargs['task']._sim.config.CAMERA_TURN_ANGLE
        relative_angle_limit = kwargs['task']._sim.config.RELATIVE_ANGLE_LIMIT
        camera_rotation_new = self._cal_camera_angle(-(delta + camera_delta), relative_angle_limit)
        self._implement_camera_action(camera_rotation_new)

        observations = super().step(collided)

        return observations


@registry.register_task_action
class BodyRightCameraRightAction(BaseAction):

    def _get_uuid(self, *args, **kwargs) -> str:
        return "body_right_camera_right"

    def step(self, *args, **kwargs):
        kwargs['task'].is_found_called = False
        collided = self._sim._sim.get_agent(0).act(HabitatSimActions.TURN_RIGHT)
        agent_state = self._sim._sim.get_agent(0).get_state()

        delta = kwargs['task']._sim.config.TURN_ANGLE
        camera_delta = kwargs['task']._sim.config.CAMERA_TURN_ANGLE
        relative_angle_limit = kwargs['task']._sim.config.RELATIVE_ANGLE_LIMIT
        camera_rotation_new = self._cal_camera_angle((camera_delta - delta), relative_angle_limit)
        self._implement_camera_action(camera_rotation_new)

        observations = super().step(collided)

        return observations


def joint_action(body_action, camera_action, cfg_AC: config, time_steps, test_new_baseline = None):
    fourAction2tenAction = {0:0, 1:1, 2:5, 3:9}

    if cfg_AC.camera_AC == 'none':   # multiON的设置，输出4个动作
        b_action = torch.zeros_like(body_action)
        for i in range(len(body_action)):
            b_action[i] = fourAction2tenAction[body_action[i].item()]
        action = b_action
    elif cfg_AC.body_AC == 'e2e' and cfg_AC.camera_AC == 'e2e': # e2e同时决定body和camera，输出10个动作
        action = body_action
    else:
        mask = (body_action-1)>=0   # 判断是否执行found

        if test_new_baseline is not None:
            for idx, flag in enumerate(test_new_baseline):
                if flag:
                    body_action[idx] = camera_action[idx] + 1 # only for another baseline testing

        action = (body_action-1)*3 + camera_action + 1
        action *= mask

    action[time_steps < 3] = 5   # 前三步左转左看
    body_action[time_steps < 3] = 2
    camera_action[time_steps < 3] = 1
    return action
