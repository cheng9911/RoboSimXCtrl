"""
MuJoCo 物理仿真环境封装类
MuJoCo physics simulation environment wrapper class.

从 utils/mj.py 和测试脚本中提取的 MuJoCo 交互逻辑，提供统一的仿真接口。
Extracted MuJoCo interaction logic from utils/mj.py and test scripts,
providing a unified simulation interface.
"""

import warnings
from typing import Optional, Union

import numpy as np
from spatialmath import SE3
import spatialmath.base as smb

try:
    import mujoco
    import mujoco.viewer
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False


def _check_mujoco():
    """检查 mujoco 是否已安装 / Check if mujoco is installed."""
    if not _HAS_MUJOCO:
        raise ImportError(
            "mujoco is required for MuJoCoSim but not installed. "
            "Install it via: pip install mujoco"
        )


class MuJoCoSim:
    """
    MuJoCo 物理仿真环境
    MuJoCo physics simulation environment.

    封装 MjModel/MjData，提供仿真控制、关节操作、刚体位姿查询、
    约束抓取、可视化管理和渲染功能。
    Wraps MjModel/MjData, providing simulation control, joint manipulation,
    body pose queries, constraint-based grasping, viewer management, and rendering.
    """

    def __init__(self, xml_path: str, timestep: float = 0.002):
        """
        从 XML 文件加载 MuJoCo 模型并创建仿真数据。
        Load MuJoCo model from XML and create simulation data.

        参数 / Parameters:
            xml_path: MuJoCo XML 场景文件路径 / Path to MuJoCo XML scene file
            timestep: 仿真时间步长(秒) / Simulation timestep in seconds
        """
        _check_mujoco()

        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(xml_path)
        self.data: mujoco.MjData = mujoco.MjData(self.model)
        self.model.opt.timestep = timestep

        self.viewer = None
        self.renderer = None
        self._renderer_width = 0
        self._renderer_height = 0

    # ================================================================
    # 仿真控制 / Simulation Control
    # ================================================================

    def step(self, ctrl: np.ndarray = None) -> None:
        """
        推进一步仿真。若提供 ctrl 则先设置控制信号。
        Advance simulation by one timestep. If ctrl is provided, set it first.

        参数 / Parameters:
            ctrl: 控制信号数组(可选) / Control signal array (optional)
        """
        if ctrl is not None:
            self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        self.sync_viewer()

    def forward(self) -> None:
        """
        调用 mj_forward 更新仿真状态。
        Call mj_forward to update simulation state.
        """
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        """
        重置仿真数据到初始状态。
        Reset simulation data to initial state.
        """
        mujoco.mj_resetData(self.model, self.data)

    # ================================================================
    # 关节控制 / Joint Control
    # ================================================================

    def _resolve_joint_id(self, name_or_id: Union[str, int]) -> int:
        """
        将关节名称或 ID 解析为整数 ID。
        Resolve joint name or ID to integer ID.
        """
        if isinstance(name_or_id, int):
            return name_or_id
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name_or_id)

    def get_joint_qpos_addr(self, joint_id: int) -> int:
        """
        获取关节在 qpos 数组中的起始地址。
        Get the start address of a joint in the qpos array.
        """
        return self.model.jnt_qposadr[joint_id]

    def get_joint_dim(self, joint_id: int) -> int:
        """
        获取关节的自由度维度。
        Get the DOF dimension of a joint.
        """
        return len(self.data.joint(joint_id).qpos)

    def get_joint_qpos_inds(self, joint_id: int) -> np.ndarray:
        """
        获取关节在 qpos 数组中的索引范围。
        Get the index range of a joint in the qpos array.
        """
        addr = self.get_joint_qpos_addr(joint_id)
        dim = self.get_joint_dim(joint_id)
        return np.arange(addr, addr + dim)

    def set_joint_q(self, name_or_id: Union[str, int], q: Union[np.ndarray, float],
                    unit: str = "rad") -> None:
        """
        设置关节位置。
        Set joint position.

        参数 / Parameters:
            name_or_id: 关节名称或 ID / Joint name or ID
            q: 关节位置值 / Joint position value
            unit: 单位，"rad" 或 "deg" / Unit, "rad" or "deg"
        """
        joint_id = self._resolve_joint_id(name_or_id)
        if unit == 'deg':
            q = np.deg2rad(q)
        q_inds = self.get_joint_qpos_inds(joint_id)
        self.data.qpos[q_inds] = q

    def get_joint_q(self, name_or_id: Union[str, int]) -> np.ndarray:
        """
        获取关节当前位置。
        Get current joint position.

        参数 / Parameters:
            name_or_id: 关节名称或 ID / Joint name or ID

        返回 / Returns:
            关节位置数组 / Joint position array
        """
        joint_id = self._resolve_joint_id(name_or_id)
        q_inds = self.get_joint_qpos_inds(joint_id)
        return self.data.qpos[q_inds].copy()

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        """
        设置控制信号。
        Set control signal.

        参数 / Parameters:
            ctrl: 控制信号数组 / Control signal array
        """
        self.data.ctrl[:] = ctrl

    # ================================================================
    # 刚体位姿 / Body Pose
    # ================================================================

    def _resolve_body_id(self, name_or_id: Union[str, int]) -> int:
        """
        将刚体名称或 ID 解析为整数 ID。
        Resolve body name or ID to integer ID.
        """
        if isinstance(name_or_id, int):
            return name_or_id
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name_or_id)

    def get_body_pose(self, name_or_id: Union[str, int]) -> SE3:
        """
        获取刚体位姿(位置 + 姿态)，返回 SE3 对象。
        Get body pose (position + orientation) as SE3 object.

        参数 / Parameters:
            name_or_id: 刚体名称或 ID / Body name or ID

        返回 / Returns:
            刚体位姿 SE3 / Body pose as SE3
        """
        body_id = self._resolve_body_id(name_or_id)
        t = self.data.body(body_id).xpos.copy()
        q = self.data.body(body_id).xquat.copy()
        # mujoco quaternion: (w, x, y, z) -> spatialmath expects same order
        R = smb.q2r(q)
        return SE3.Rt(R=R, t=t, check=False)

    def get_body_pose_xyz(self, name_or_id: Union[str, int]) -> np.ndarray:
        """
        获取刚体位置(x, y, z)。
        Get body position (x, y, z).

        参数 / Parameters:
            name_or_id: 刚体名称或 ID / Body name or ID

        返回 / Returns:
            3 元素位置数组 / 3-element position array
        """
        body_id = self._resolve_body_id(name_or_id)
        return self.data.body(body_id).xpos.copy()

    def set_body_pose(self, name_or_id: Union[str, int], xpos: np.ndarray) -> None:
        """
        设置刚体位置。
        Set body position.

        参数 / Parameters:
            name_or_id: 刚体名称或 ID / Body name or ID
            xpos: 位置数组(x, y, z) / Position array (x, y, z)
        """
        body_id = self._resolve_body_id(name_or_id)
        self.data.body(body_id).xpos[:] = xpos[:3]

    # ================================================================
    # 自由关节位姿 / Free Joint Pose
    # ================================================================

    def set_free_joint_pose(self, joint_name: Union[str, int], T: SE3) -> None:
        """
        设置自由关节的位置和姿态(从 SE3 对象)。
        Set free joint position and orientation from an SE3 object.

        参数 / Parameters:
            joint_name: 关节名称或 ID / Joint name or ID
            T: 目标位姿 SE3 / Target pose SE3
        """
        joint_id = self._resolve_joint_id(joint_name)
        t = T.t
        q = smb.r2q(T.R)  # (w, x, y, z)
        T_new = np.concatenate([t, q])
        q_inds = self.get_joint_qpos_inds(joint_id)
        self.data.qpos[q_inds] = T_new

    # ================================================================
    # 约束/抓取 / Constraints / Grasping
    # ================================================================

    def attach(self, equality_name: str, free_joint_name: str, T: SE3,
               eq_data: np.ndarray = None,
               eq_solimp: np.ndarray = None,
               eq_solref: np.ndarray = None) -> None:
        """
        激活 MuJoCo 等式约束以将物体附着到抓手。
        Activate MuJoCo equality constraint to attach object to gripper.

        参数 / Parameters:
            equality_name: 等式约束名称 / Equality constraint name
            free_joint_name: 自由关节名称 / Free joint name
            T: 附着位姿 SE3 / Attachment pose SE3
            eq_data: 等式约束数据(可选) / Equality constraint data (optional)
            eq_solimp: 求解器阻抗参数(可选) / Solver impedance parameters (optional)
            eq_solref: 求解器参考参数(可选) / Solver reference parameters (optional)
        """
        eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name)
        if eq_id == -1:
            raise ValueError(f"Equality constraint '{equality_name}' not found.")

        if eq_data is None:
            eq_data = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        if eq_solimp is None:
            eq_solimp = np.array([[0.99, 0.99, 0.001, 0.5, 1]])
        if eq_solref is None:
            eq_solref = np.array([0.0001, 1])

        # 先禁用约束，设置自由关节位姿，再启用
        # Disable constraint first, set free joint pose, then enable
        self.data.eq_active[eq_id] = 0
        self.set_free_joint_pose(free_joint_name, T)

        self.model.equality(equality_name).data[:] = eq_data
        self.model.equality(equality_name).solimp[:] = eq_solimp
        self.model.equality(equality_name).solref[:] = eq_solref

        self.data.eq_active[eq_id] = 1

    def detach(self, equality_name: str) -> None:
        """
        停用等式约束(解除抓取)。
        Deactivate equality constraint (release grasp).

        参数 / Parameters:
            equality_name: 等式约束名称 / Equality constraint name
        """
        eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name)
        if eq_id == -1:
            raise ValueError(f"Equality constraint '{equality_name}' not found.")
        self.data.eq_active[eq_id] = 0

    # ================================================================
    # 可视化 / Viewer
    # ================================================================

    def launch_viewer(self) -> None:
        """
        启动被动 MuJoCo 查看器。
        Launch passive MuJoCo viewer.
        """
        if self.viewer is not None:
            warnings.warn("Viewer already launched.")
            return
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def close_viewer(self) -> None:
        """
        关闭 MuJoCo 查看器。
        Close MuJoCo viewer.
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def sync_viewer(self) -> None:
        """
        同步查看器状态(在 step 之后调用)。
        Sync viewer state (call after step).
        """
        if self.viewer is not None:
            try:
                self.viewer.sync()
            except Exception:
                pass

    # ================================================================
    # 渲染 / Rendering
    # ================================================================

    def _ensure_renderer(self, width: int, height: int) -> None:
        """
        确保渲染器已创建且尺寸正确。
        Ensure renderer exists with correct size.
        """
        if (self.renderer is None or
                self._renderer_width != width or
                self._renderer_height != height):
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderer_width = width
            self._renderer_height = height

    def render_rgb(self, camera: Union[str, int], width: int, height: int) -> np.ndarray:
        """
        渲染 RGB 图像。
        Render RGB image.

        参数 / Parameters:
            camera: 相机名称或 ID / Camera name or ID
            width: 图像宽度 / Image width
            height: 图像高度 / Image height

        返回 / Returns:
            RGB 图像数组 (height, width, 3), dtype=uint8
            RGB image array (height, width, 3), dtype=uint8
        """
        self._ensure_renderer(width, height)
        self.renderer.update_scene(self.data, camera)
        return self.renderer.render()

    def render_depth(self, camera: Union[str, int], width: int, height: int) -> np.ndarray:
        """
        渲染深度图像。
        Render depth image.

        参数 / Parameters:
            camera: 相机名称或 ID / Camera name or ID
            width: 图像宽度 / Image width
            height: 图像高度 / Image height

        返回 / Returns:
            深度图像数组 (height, width), dtype=float32
            Depth image array (height, width), dtype=float32
        """
        self._ensure_renderer(width, height)
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera)
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        return depth

    def get_point_cloud(self, camera: Union[str, int], width: int, height: int,
                        depth_limit: float = 3.0) -> np.ndarray:
        """
        从深度和 RGB 图像生成点云。
        Generate point cloud from depth and RGB images.

        参数 / Parameters:
            camera: 相机名称或 ID / Camera name or ID
            width: 图像宽度 / Image width
            height: 图像高度 / Image height
            depth_limit: 最大深度阈值 / Maximum depth threshold

        返回 / Returns:
            点云数组 (N, 6)，列为 [x, y, z, r, g, b]
            Point cloud array (N, 6), columns [x, y, z, r, g, b]
        """
        rgb = self.render_rgb(camera, width, height)
        depth = self.render_depth(camera, width, height)

        fovy = self.model.vis.global_.fovy
        f = height / (2.0 * np.tan(fovy * np.pi / 180.0 / 2.0))
        cx, cy = width / 2.0, height / 2.0

        points = []
        for v in range(height):
            for u in range(width):
                z = depth[v, u]
                if z <= 0 or z > depth_limit:
                    continue
                x = (u - cx) * z / f
                y = (v - cy) * z / f
                r, g, b = rgb[v, u]
                points.append([x, y, z, r, g, b])

        if len(points) == 0:
            return np.zeros((0, 6))
        return np.array(points)

    # ================================================================
    # 场景物体管理 / Scene Object Management
    # ================================================================

    def get_box_size(self, box_name: str = "Box") -> np.ndarray:
        """
        获取盒子几何体的尺寸。
        Get box geometry size.
        """
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, box_name)
        if geom_id == -1:
            raise ValueError(f"Geometry '{box_name}' not found.")
        return self.model.geom(geom_id).size.copy()

    def get_box_position(self, box_name: str = "Box") -> np.ndarray:
        """
        获取盒子几何体的位置。
        Get box geometry position.
        """
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, box_name)
        if geom_id == -1:
            raise ValueError(f"Geometry '{box_name}' not found.")
        return self.model.geom_pos[geom_id].copy()

    def set_box_position(self, box_name: str, pos: np.ndarray) -> None:
        """
        设置盒子几何体的位置。
        Set box geometry position.
        """
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, box_name)
        if geom_id == -1:
            raise ValueError(f"Geometry '{box_name}' not found.")
        self.data.geom_xpos[geom_id] = pos

    def set_body_position(self, body_name: str, pos: np.ndarray) -> None:
        """
        设置刚体位置。
        Set body position.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found.")
        self.data.body(body_id).xpos[:] = pos
