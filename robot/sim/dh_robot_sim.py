"""
DH 参数机器人仿真基类
DH-parameter robot simulation base class.

继承 robot.robot.Robot，添加可选的 MuJoCo 物理仿真和 Meshcat Web 可视化支持。
所有运动学/动力学方法(fkine, ikine, get_inertia, inv_dynamics 等)全部继承。
Inherits robot.robot.Robot, adding optional MuJoCo physics simulation and Meshcat
web visualization support. All kinematics/dynamics methods are inherited.
"""

import time
import warnings
from typing import List, Optional, Union

import numpy as np
from spatialmath import SE3

from ..robot import Robot
from .mujoco_sim import MuJoCoSim

try:
    from pinocchio.visualize import MeshcatVisualizer as _MeshcatVisualizer
    _HAS_MESHCAT = True
except ImportError:
    _HAS_MESHCAT = False


class DHRobotSim(Robot):
    """
    DH 参数机器人仿真基类
    DH-parameter robot simulation base class.

    继承 Robot 基类，添加可选的 MuJoCo 仿真后端和 Meshcat 可视化后端。
    MoveJ/MoveL/servoJ 中的 servoJ 在此实现，MoveJ/MoveL 留给子类或未来扩展。
    Inherits Robot base class, adding optional MuJoCo simulation backend and
    Meshcat visualization backend.
    """

    def __init__(self, mujoco_xml_path: Optional[str] = None,
                 meshcat_port: Optional[int] = None):
        """
        初始化 DH 机器人仿真。
        Initialize DH robot simulation.

        参数 / Parameters:
            mujoco_xml_path: MuJoCo 场景 XML 路径(可选) / MuJoCo scene XML path (optional)
            meshcat_port: Meshcat 服务器端口(可选) / Meshcat server port (optional)
        """
        super().__init__()

        # --- MuJoCo 仿真后端(可选) ---
        # --- MuJoCo simulation backend (optional) ---
        self.mj_sim: Optional[MuJoCoSim] = None
        if mujoco_xml_path is not None:
            self.mj_sim = MuJoCoSim(mujoco_xml_path)

        # --- Meshcat 可视化(可选) ---
        # --- Meshcat visualization (optional) ---
        self.meshcat = None
        if meshcat_port is not None:
            self._init_meshcat(meshcat_port)

    # ================================================================
    # Meshcat 初始化 / Meshcat Initialization
    # ================================================================

    def _init_meshcat(self, port: int) -> None:
        """
        初始化 Meshcat 可视化器。
        Initialize Meshcat visualizer.

        参数 / Parameters:
            port: Meshcat 服务器端口 / Meshcat server port
        """
        if not _HAS_MESHCAT:
            warnings.warn(
                "pinocchio is required for Meshcat visualization but not installed. "
                "Meshcat will be unavailable."
            )
            self.meshcat = None
            return
        try:
            import meshcat
            self.meshcat = meshcat.Visualizer(zmq_url=f"tcp://127.0.0.1:{port}")
        except Exception as e:
            warnings.warn(f"Failed to initialize Meshcat on port {port}: {e}")
            self.meshcat = None

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
            n = min(len(q), self.mj_sim.model.njnt - 1)  # 减去 world joint
            for i in range(n):
                self.mj_sim.set_joint_q(i + 1, q[i])  # +1 跳过 world joint

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
    # Meshcat 可视化接口 / Meshcat Visualization Interface
    # ================================================================

    def meshcat_display(self, q: np.ndarray = None) -> None:
        """
        在 Meshcat 中显示机器人。
        Display robot in Meshcat.

        参数 / Parameters:
            q: 关节位置(默认 self.q0) / Joint positions (default self.q0)
        """
        if self.meshcat is None:
            return
        if q is None:
            q = np.array(self.q0)
        try:
            # 对于 DH 机器人，在 Meshcat 中显示正运动学末端位姿
            # For DH robot, display end-effector pose via forward kinematics
            T = self.fkine(q)
            # 将 SE3 转换为 4x4 矩阵并设置到 Meshcat
            # Convert SE3 to 4x4 matrix and set to Meshcat
            self.meshcat.set_transform(T.A)
        except Exception as e:
            warnings.warn(f"meshcat_display failed: {e}")

    def meshcat_display_trajectory(self, q_list: List[np.ndarray]) -> None:
        """
        在 Meshcat 中显示轨迹动画。
        Display trajectory animation in Meshcat.

        参数 / Parameters:
            q_list: 关节位置序列 / Sequence of joint positions
        """
        if self.meshcat is None:
            return
        for q in q_list:
            self.meshcat_display(q)
            time.sleep(0.01)

    def meshcat_display_frame(self, T: SE3, name: str) -> None:
        """
        在 Meshcat 中显示坐标系。
        Display coordinate frame in Meshcat.

        参数 / Parameters:
            T: 坐标系位姿 SE3 / Frame pose SE3
            name: 坐标系名称 / Frame name
        """
        if self.meshcat is None:
            return
        try:
            import meshcat.geometry as g
            self.meshcat[name].set_object(
                g.LineSegments(
                    g.PointsGeometry(
                        position=np.array([[0, 0, 0], [0.1, 0, 0],
                                           [0, 0, 0], [0, 0.1, 0],
                                           [0, 0, 0], [0, 0, 0.1]]).T,
                        color=np.array([[1, 0, 0], [1, 0, 0],
                                        [0, 1, 0], [0, 1, 0],
                                        [0, 0, 1], [0, 0, 1]]).T
                    ),
                    g.LineBasicMaterial(linewidth=2)
                )
            )
            self.meshcat[name].set_transform(T.A)
        except Exception as e:
            warnings.warn(f"meshcat_display_frame failed: {e}")

    # ================================================================
    # servoJ 重写 / servoJ Override
    # ================================================================

    def servoJ(self, q_target: np.ndarray, delta_t: float = 0.001) -> int:
        """
        关节伺服控制：位置/速度检查 + MuJoCo 同步 + Meshcat 显示。
        Joint servo control: position/velocity check + MuJoCo sync + Meshcat display.

        参数 / Parameters:
            q_target: 目标关节角 / Target joint angles
            delta_t: 控制周期(秒) / Control period (seconds)

        返回 / Returns:
            0 成功, -1 失败 / 0 success, -1 failure
        """
        q_target = np.array(q_target)

        # 位置限制检查 / Position limit check
        if hasattr(self, "_q_lim_low") and hasattr(self, "_q_lim_up"):
            for i in range(len(q_target)):
                if q_target[i] < self._q_lim_low[i] or q_target[i] > self._q_lim_up[i]:
                    print(f"[servoJ] Joint {i} position limit exceeded: "
                          f"{np.degrees(q_target[i]):.2f} deg")
                    return -1

        # 速度限制检查 / Velocity limit check
        q_current = np.array(self.q0)
        q_offset = np.abs(q_target - q_current)
        if hasattr(self, "_q_lim_low"):
            # 估算最大速度 / Estimate max velocity
            v_max_est = 3.0  # rad/s 默认估计值 / default estimate
            for i in range(len(q_target)):
                if q_offset[i] > v_max_est * delta_t:
                    print(f"[servoJ] Joint {i} velocity exceeded: "
                          f"{np.degrees(q_offset[i] / delta_t):.2f} deg/s")
                    return -1

        # 更新关节位置 / Update joint positions
        self.q0 = q_target.tolist() if isinstance(q_target, np.ndarray) else q_target[:]

        # 同步 MuJoCo / Sync MuJoCo
        if self.mj_sim is not None:
            self.mujoco_set_joint(q_target)
            self.mj_sim.step()

        # 同步 Meshcat / Sync Meshcat
        if self.meshcat is not None:
            self.meshcat_display(q_target)

        return 0
