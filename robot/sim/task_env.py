"""
操作任务仿真环境
Manipulation task simulation environment.

组合机器人仿真器与 MuJoCo 仿真场景，提供物体管理、抓取逻辑和观测生成。
Combines robot simulator with MuJoCo simulation scene, providing object
management, grasp logic, and observation generation.
"""

import warnings
from typing import Dict, List, Optional, Union

import numpy as np
from spatialmath import SE3

from .mujoco_sim import MuJoCoSim


class TaskEnv:
    """
    操作任务环境
    Manipulation task environment.

    管理场景物体、抓取/释放操作、观测生成和场景重置。
    Manages scene objects, grasp/release operations, observation generation,
    and scene reset.
    """

    def __init__(self, robot, mujoco_sim: Optional[MuJoCoSim] = None):
        """
        初始化操作任务环境。
        Initialize manipulation task environment.

        参数 / Parameters:
            robot: 机器人仿真器实例(DHRobotSim 或 PinocchioRobotSim)
                   Robot simulator instance (DHRobotSim or PinocchioRobotSim)
            mujoco_sim: MuJoCo 仿真实例(可选，也可从 robot.mj_sim 获取)
                        MuJoCo simulation instance (optional, can also get from robot.mj_sim)
        """
        self.robot = robot
        self.sim = mujoco_sim if mujoco_sim is not None else getattr(robot, 'mj_sim', None)

        # 记录已生成的物体 / Track spawned objects
        self._spawned_objects: Dict[str, dict] = {}
        # 记录抓取约束 / Track grasp constraints
        self._grasp_constraints: Dict[str, bool] = {}

    # ================================================================
    # 物体管理 / Object Management
    # ================================================================

    def spawn_object(self, name: str, pos: np.ndarray, shape: str = "box",
                     size: np.ndarray = None, rgba: np.ndarray = None) -> None:
        """
        向场景中添加物体。注意：MuJoCo 场景通常需要在 XML 中预定义物体。
        此方法记录物体信息，若场景中已有同名物体则设置其位置。
        Add object to scene. Note: MuJoCo scenes typically require pre-defined
        objects in XML. This method records object info, and if a same-named
        object already exists in the scene, sets its position.

        参数 / Parameters:
            name: 物体名称 / Object name
            pos: 初始位置(x, y, z) / Initial position (x, y, z)
            size: 物体尺寸(可选) / Object size (optional)
            shape: 形状类型("box", "sphere", "cylinder") / Shape type
            rgba: 颜色(可选) / Color (optional)
        """
        if size is None:
            size = np.array([0.05, 0.05, 0.05])
        if rgba is None:
            rgba = np.array([1.0, 0.0, 0.0, 1.0])

        # 记录物体信息 / Record object info
        self._spawned_objects[name] = {
            "pos": np.array(pos),
            "shape": shape,
            "size": np.array(size),
            "rgba": np.array(rgba),
        }

        # 如果 MuJoCo 场景中已有该物体，设置其位置
        # If the MuJoCo scene already has this object, set its position
        if self.sim is not None:
            try:
                self.sim.set_body_pose(name, np.array(pos))
            except (ValueError, Exception):
                warnings.warn(
                    f"Object '{name}' not found in MuJoCo scene. "
                    f"Pre-define it in the XML or use mujoco_set_joint to "
                    f"place pre-existing objects."
                )

    def remove_object(self, name: str) -> None:
        """
        从场景中移除物体(将其移到远处)。
        Remove object from scene (moves it far away).

        参数 / Parameters:
            name: 物体名称 / Object name
        """
        if name in self._spawned_objects:
            del self._spawned_objects[name]

        # 释放相关抓取 / Release related grasp
        if name in self._grasp_constraints:
            self.release(name)

        # 将物体移到远处 / Move object far away
        if self.sim is not None:
            try:
                self.sim.set_body_pose(name, np.array([100.0, 100.0, 100.0]))
            except (ValueError, Exception):
                pass

    def reset_scene(self) -> None:
        """
        重置场景：重置 MuJoCo 仿真，将所有物体恢复到初始位置。
        Reset scene: reset MuJoCo simulation, restore all objects to initial positions.
        """
        # 重置 MuJoCo / Reset MuJoCo
        if self.sim is not None:
            self.sim.reset()

        # 恢复物体位置 / Restore object positions
        for name, info in self._spawned_objects.items():
            if self.sim is not None:
                try:
                    self.sim.set_body_pose(name, info["pos"])
                except (ValueError, Exception):
                    pass

        # 释放所有抓取 / Release all grasps
        for name in list(self._grasp_constraints.keys()):
            self.release(name)

        # 重置机器人到初始位姿 / Reset robot to initial pose
        if hasattr(self.robot, 'q0'):
            n = len(self.robot.q0) if hasattr(self.robot.q0, '__len__') else 0
            if n > 0:
                self.robot.q0 = [0.0] * n

    # ================================================================
    # 抓取操作 / Grasping Operations
    # ================================================================

    def grasp(self, object_name: str,
              equality_name: str = None,
              free_joint_name: str = None) -> None:
        """
        抓取物体：激活 MuJoCo 等式约束将物体附着到抓手。
        Grasp object: activate MuJoCo equality constraint to attach object to gripper.

        参数 / Parameters:
            object_name: 物体名称 / Object name
            equality_name: 等式约束名称(默认: "{object_name}_grasp_eq")
                           Equality constraint name (default: "{object_name}_grasp_eq")
            free_joint_name: 自由关节名称(默认: "{object_name}_free_joint")
                             Free joint name (default: "{object_name}_free_joint")
        """
        if self.sim is None:
            warnings.warn("No MuJoCo simulation available for grasping.")
            return

        if equality_name is None:
            equality_name = f"{object_name}_grasp_eq"
        if free_joint_name is None:
            free_joint_name = f"{object_name}_free_joint"

        # 获取当前抓手位姿 / Get current gripper pose
        try:
            ee_pose = self.robot.fkine(self.robot.q0) if hasattr(self.robot, 'fkine') else SE3()
        except Exception:
            ee_pose = SE3()

        try:
            self.sim.attach(equality_name, free_joint_name, ee_pose)
            self._grasp_constraints[object_name] = True
        except (ValueError, Exception) as e:
            warnings.warn(f"Failed to grasp '{object_name}': {e}")

    def release(self, object_name: str, equality_name: str = None) -> None:
        """
        释放物体：停用等式约束。
        Release object: deactivate equality constraint.

        参数 / Parameters:
            object_name: 物体名称 / Object name
            equality_name: 等式约束名称(默认: "{object_name}_grasp_eq")
                           Equality constraint name (default: "{object_name}_grasp_eq")
        """
        if self.sim is None:
            return

        if equality_name is None:
            equality_name = f"{object_name}_grasp_eq"

        try:
            self.sim.detach(equality_name)
        except (ValueError, Exception) as e:
            warnings.warn(f"Failed to release '{object_name}': {e}")

        if object_name in self._grasp_constraints:
            del self._grasp_constraints[object_name]

    # ================================================================
    # 观测 / Observation
    # ================================================================

    def get_observation(self, camera: Union[str, int] = 0,
                        width: int = 256, height: int = 256) -> dict:
        """
        获取观测数据。
        Get observation data.

        参数 / Parameters:
            camera: 相机名称或 ID(用于渲染) / Camera name or ID (for rendering)
            width: 图像宽度 / Image width
            height: 图像高度 / Image height

        返回 / Returns:
            观测字典，包含:
            Observation dict containing:
                - "joint_pos": 关节位置 / Joint positions
                - "joint_vel": 关节速度(若可用) / Joint velocities (if available)
                - "ee_pose": 末端执行器位姿 SE3 / End-effector pose SE3
                - "rgb": RGB 图像(若渲染器已配置) / RGB image (if renderer configured)
                - "depth": 深度图像(若渲染器已配置) / Depth image (if renderer configured)
        """
        obs = {}

        # 关节位置 / Joint positions
        if hasattr(self.robot, 'q0'):
            obs["joint_pos"] = np.array(self.robot.q0)
        elif hasattr(self.robot, 'q'):
            obs["joint_pos"] = np.array(self.robot.q)
        else:
            obs["joint_pos"] = np.array([])

        # 关节速度 / Joint velocities
        if hasattr(self.robot, 'dq'):
            obs["joint_vel"] = np.array(self.robot.dq)
        else:
            obs["joint_vel"] = np.zeros_like(obs["joint_pos"])

        # 末端执行器位姿 / End-effector pose
        try:
            if hasattr(self.robot, 'fkine'):
                obs["ee_pose"] = self.robot.fkine(obs["joint_pos"])
            else:
                obs["ee_pose"] = SE3()
        except Exception:
            obs["ee_pose"] = SE3()

        # 渲染图像(可选) / Rendered images (optional)
        if self.sim is not None:
            try:
                obs["rgb"] = self.sim.render_rgb(camera, width, height)
            except Exception:
                pass

            try:
                obs["depth"] = self.sim.render_depth(camera, width, height)
            except Exception:
                pass

        return obs
