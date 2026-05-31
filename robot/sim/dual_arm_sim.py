"""
双臂仿真管理器
Dual-arm simulation manager.

管理两个机器人在同一 MuJoCo 场景中的同步仿真。
Manages synchronized simulation of two robots in a single MuJoCo scene.
"""

import threading
import warnings
from typing import Optional, Union

import numpy as np
from spatialmath import SE3

from .mujoco_sim import MuJoCoSim


class DualArmSim:
    """
    双臂仿真管理器
    Dual-arm simulation manager.

    在一个 MuJoCo 场景中协调两个机器人的同步运动控制。
    Coordinates synchronized motion control of two robots in one MuJoCo scene.
    """

    def __init__(self, left_arm, right_arm, sim: MuJoCoSim):
        """
        初始化双臂仿真。
        Initialize dual-arm simulation.

        参数 / Parameters:
            left_arm: 左臂机器人仿真器 / Left arm robot simulator
            right_arm: 右臂机器人仿真器 / Right arm robot simulator
            sim: 共享的 MuJoCo 仿真实例 / Shared MuJoCo simulation instance
        """
        self.left_arm = left_arm
        self.right_arm = right_arm
        self.sim = sim

    # ================================================================
    # 同步控制 / Synchronized Control
    # ================================================================

    def step(self, q_left: np.ndarray, q_right: np.ndarray) -> None:
        """
        同时设置两个臂的关节位置并推进仿真一步。
        Set joint positions for both arms and advance simulation by one step.

        参数 / Parameters:
            q_left: 左臂关节位置 / Left arm joint positions
            q_right: 右臂关节位置 / Right arm joint positions
        """
        # 先设置两个臂的关节位置(原子操作)
        # Set both arm joint positions first (atomic operation)
        if hasattr(self.left_arm, 'mujoco_set_joint'):
            self.left_arm.mujoco_set_joint(q_left)
        if hasattr(self.right_arm, 'mujoco_set_joint'):
            self.right_arm.mujoco_set_joint(q_right)

        # 推进仿真 / Advance simulation
        self.sim.step()

    def sync_viewer(self) -> None:
        """
        同步查看器。
        Sync viewer.
        """
        self.sim.sync_viewer()

    # ================================================================
    # 运动控制 / Motion Control
    # ================================================================

    def MoveJ_dual_arm(self, q_left: np.ndarray, q_right: np.ndarray,
                       v_max: float = 1.0, a_max: float = 1.0,
                       dt: float = 0.001) -> None:
        """
        双臂同步关节运动(并发执行)。
        Dual-arm synchronized joint motion (concurrent execution).

        参数 / Parameters:
            q_left: 左臂目标关节位置 / Left arm target joint positions
            q_right: 右臂目标关节位置 / Right arm target joint positions
            v_max: 最大速度 / Maximum velocity
            a_max: 最大加速度 / Maximum acceleration
            dt: 控制周期 / Control period
        """
        left_error = [None]
        right_error = [None]

        def move_left():
            try:
                if hasattr(self.left_arm, 'MoveJ'):
                    self.left_arm.MoveJ(q_left, v_max=v_max, a_max=a_max, dt=dt)
            except Exception as e:
                left_error[0] = e

        def move_right():
            try:
                if hasattr(self.right_arm, 'MoveJ'):
                    self.right_arm.MoveJ(q_right, v_max=v_max, a_max=a_max, dt=dt)
            except Exception as e:
                right_error[0] = e

        # 并发执行两个臂的运动 / Execute both arm motions concurrently
        t_left = threading.Thread(target=move_left)
        t_right = threading.Thread(target=move_right)

        t_left.start()
        t_right.start()

        t_left.join()
        t_right.join()

        if left_error[0] is not None:
            warnings.warn(f"Left arm MoveJ error: {left_error[0]}")
        if right_error[0] is not None:
            warnings.warn(f"Right arm MoveJ error: {right_error[0]}")

    def MoveJ_left_arm(self, q_left: np.ndarray,
                       v_max: float = 1.0, a_max: float = 1.0,
                       dt: float = 0.001) -> None:
        """
        仅左臂关节运动。
        Left arm only joint motion.

        参数 / Parameters:
            q_left: 左臂目标关节位置 / Left arm target joint positions
            v_max: 最大速度 / Maximum velocity
            a_max: 最大加速度 / Maximum acceleration
            dt: 控制周期 / Control period
        """
        if hasattr(self.left_arm, 'MoveJ'):
            self.left_arm.MoveJ(q_left, v_max=v_max, a_max=a_max, dt=dt)

    def MoveJ_right_arm(self, q_right: np.ndarray,
                        v_max: float = 1.0, a_max: float = 1.0,
                        dt: float = 0.001) -> None:
        """
        仅右臂关节运动。
        Right arm only joint motion.

        参数 / Parameters:
            q_right: 右臂目标关节位置 / Right arm target joint positions
            v_max: 最大速度 / Maximum velocity
            a_max: 最大加速度 / Maximum acceleration
            dt: 控制周期 / Control period
        """
        if hasattr(self.right_arm, 'MoveJ'):
            self.right_arm.MoveJ(q_right, v_max=v_max, a_max=a_max, dt=dt)

    # ================================================================
    # 清理 / Cleanup
    # ================================================================

    def close(self) -> None:
        """
        关闭仿真，释放资源。
        Close simulation, release resources.
        """
        if self.sim is not None:
            self.sim.close_viewer()
