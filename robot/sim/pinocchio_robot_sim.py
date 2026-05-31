"""
Pinocchio 机器人仿真基类
Pinocchio robot simulation base class.

继承 robot.robot_pinocchio.Robot，添加可选的 MuJoCo 物理仿真支持。
现有的 Meshcat 可视化由 Pinocchio 基类管理。
Inherits robot.robot_pinocchio.Robot, adding optional MuJoCo physics simulation.
Existing Meshcat visualization is managed by the Pinocchio base class.
"""

import warnings
from typing import Optional

import numpy as np
from spatialmath import SE3

from .. import robot_pinocchio
from .mujoco_sim import MuJoCoSim


class PinocchioRobotSim(robot_pinocchio.Robot):
    """
    Pinocchio 机器人仿真基类
    Pinocchio robot simulation base class.

    继承 Pinocchio Robot，添加可选的 MuJoCo 仿真后端。
    MoveJ/MoveL/servoJ 中的 servoJ 在此重写以同步 MuJoCo 状态。
    Inherits Pinocchio Robot, adding optional MuJoCo simulation backend.
    The servoJ method is overridden to sync MuJoCo state.
    """

    def __init__(self, urdf_path: str, mesh_dir: str,
                 mujoco_xml_path: Optional[str] = None,
                 visualizer: bool = False,
                 target_frame: str = "link_7"):
        """
        初始化 Pinocchio 机器人仿真。
        Initialize Pinocchio robot simulation.

        参数 / Parameters:
            urdf_path: URDF 文件路径 / URDF file path
            mesh_dir: 网格资源目录 / Mesh resource directory
            mujoco_xml_path: MuJoCo 场景 XML 路径(可选) / MuJoCo scene XML path (optional)
            visualizer: 是否启动 Meshcat 可视化 / Whether to launch Meshcat visualizer
            target_frame: 目标末端执行器帧名 / Target end-effector frame name
        """
        super().__init__(urdf_path, mesh_dir, vizualizer=visualizer, target_frame=target_frame)

        # --- MuJoCo 仿真后端(可选) ---
        # --- MuJoCo simulation backend (optional) ---
        self.mj_sim: Optional[MuJoCoSim] = None
        if mujoco_xml_path is not None:
            self.mj_sim = MuJoCoSim(mujoco_xml_path)

    # ================================================================
    # MuJoCo 仿真接口(委托给 MuJoCoSim)
    # MuJoCo simulation interface (delegate to MuJoCoSim)
    # ================================================================

    def mujoco_step(self) -> None:
        """
        推进 MuJoCo 仿真一步。
        Advance MuJoCo simulation by one step.
        """
        if self.mj_sim is not None:
            self.mj_sim.step()

    def mujoco_set_joint(self, q: np.ndarray) -> None:
        """
        设置 MuJoCo 中所有机器人关节位置。
        Set all robot joint positions in MuJoCo.

        参数 / Parameters:
            q: 关节位置数组 / Joint position array
        """
        if self.mj_sim is not None:
            n = min(len(q), self.mj_sim.model.njnt - 1)
            for i in range(n):
                self.mj_sim.set_joint_q(i + 1, q[i])

    def mujoco_get_joint(self) -> np.ndarray:
        """
        获取 MuJoCo 中所有机器人关节位置。
        Get all robot joint positions from MuJoCo.

        返回 / Returns:
            关节位置数组 / Joint position array
        """
        if self.mj_sim is None:
            return np.array([])
        n = self.mj_sim.model.njnt - 1
        q = np.zeros(n)
        for i in range(n):
            q[i] = self.mj_sim.get_joint_q(i + 1)
        return q

    def mujoco_get_body_pose(self, name: str) -> SE3:
        """
        获取 MuJoCo 中刚体位姿。
        Get body pose from MuJoCo.

        参数 / Parameters:
            name: 刚体名称 / Body name

        返回 / Returns:
            刚体位姿 SE3 / Body pose SE3
        """
        if self.mj_sim is None:
            return SE3()
        return self.mj_sim.get_body_pose(name)

    # ================================================================
    # servoJ 重写 / servoJ Override
    # ================================================================

    def servoJ(self, q_target: np.ndarray, delta_t: float = 0.001) -> int:
        """
        关节伺服控制：调用父类 servoJ(碰撞检测+位置/速度检查+Meshcat 显示)
        后同步 MuJoCo 状态。
        Joint servo control: call parent servoJ (collision check + position/velocity
        check + Meshcat display), then sync MuJoCo state.

        参数 / Parameters:
            q_target: 目标关节角 / Target joint angles
            delta_t: 控制周期(秒) / Control period (seconds)

        返回 / Returns:
            0 成功, -1 失败 / 0 success, -1 failure
        """
        # 调用父类 servoJ：碰撞检测 + 位置/速度检查 + 更新 self.q + Meshcat 显示
        # Call parent servoJ: collision check + position/velocity check + update self.q + Meshcat
        result = super().servoJ(q_target, delta_t)

        # 同步 MuJoCo / Sync MuJoCo
        if result == 0 and self.mj_sim is not None:
            self.mujoco_set_joint(self.q)
            self.mj_sim.step()

        return result
