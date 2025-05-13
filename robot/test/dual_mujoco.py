import numpy as np
import time
from scipy.spatial.transform import Rotation
import pinocchio
from pinocchio.visualize import MeshcatVisualizer
from pyroboplan.core.utils import extract_cartesian_pose
from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions
from pyroboplan.planning.cartesian_planner import CartesianPlanner, CartesianPlannerOptions
from pyroboplan.visualization.meshcat_utils import visualize_frames
import matplotlib.pyplot as plt
from pathlib import Path
import mujoco
import mujoco.viewer
import sys
import os
import csv
import queue
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)
import open3d as o3d  # 请确保你已安装 open3d
# 检查 sys.path
print("sys.path:", sys.path)

from utils import mj
DELTA_T = 0.001
class Robot:
    def __init__(
        self, 
        urdf_path: str, 

       
        mesh_dir: str  ,
        vizualizer: bool = False,
        target_frame: str = "link_7",

       
    ):
        """
        Initialize the robot with the given URDF path.
        """
        self.model, self.collision_model, self.visual_model = self.load_robot(urdf_path, mesh_dir)
        self.data = self.model.createData()
        self.collision_data = self.collision_model.createData()
        # 初始化当前关节角度为零（或其他方式）
        self.q = np.zeros(self.model.nq)
        self.target_frame = target_frame
        self.q_vel_limits = np.array([1.5] * self.model.nq)  # 最大速度 (rad/s)
        self.q_limits = np.array([[-np.pi]*self.model.nq, [np.pi]*self.model.nq]).T  # 最小最大位置限制（可选）
        self.q_acc_limits = np.array([10.0] * self.model.nq)  # 最大加速度 (rad/s²)
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model, data=self.data)
        self.viz.initViewer(open=vizualizer)
        self.viz.loadViewerModel()
        if(vizualizer):
            self.viz.display(self.q)
            time.sleep(2)
        
        
    def load_robot(self, urdf_path: str,mesh_dir ):
        """
        Load the robot model from a URDF file.
        """
        model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(urdf_path,mesh_dir)
        return model, collision_model, visual_model
    
    def get_cartesian_pose(self, q: np.ndarray):
        """
        Get the Cartesian pose of the robot's end-effector for the given joint configuration.
        """
        return extract_cartesian_pose(self.model, self.target_frame, q, data=self.data)
    
    def inverse_kinematics(self,  target_position: np.ndarray, q_start: np.ndarray, max_retries: int = 5):
        """
        Solve the inverse kinematics problem using Differential IK.
        """
        ik = DifferentialIk(self.model, data=self.data, collision_model=self.collision_model, 
                            options=DifferentialIkOptions(max_retries=max_retries))
        q_sol = ik.solve(self.target_frame, target_position, q_start)
        return q_sol
    
    def cartesian_planning(self, q_start: np.ndarray, tforms: list, dt: float = 0.05,max_retries: int = 10):
        """
        Perform Cartesian motion planning.
        """
        options = CartesianPlannerOptions(
            use_trapezoidal_scaling=True,
            max_linear_velocity=0.1,
            max_linear_acceleration=0.5,
            max_angular_velocity=1.0,
            max_angular_acceleration=1.0,
        )
        ik = DifferentialIk(self.model, data=self.data, collision_model=self.collision_model, 
                            options=DifferentialIkOptions(max_retries=max_retries))
        planner = CartesianPlanner(self.model, self.target_frame, tforms, ik, options=options)
        success, t_vec, q_vec = planner.generate(q_start, dt)
        tforms_to_show = planner.generated_tforms[::5]
        if not success:
            # 抛出异常失败
            print("Failed to generate Cartesian path.")
            return None, None

        return tforms_to_show,t_vec, q_vec,success

    def display_trajectory(self, q_vec: np.ndarray, t_vec: np.ndarray):
        """
        Visualize the joint trajectories in a Matplotlib plot.
        """
        plt.ion()
        plt.figure()
        plt.title("Robot Joint Position Trajectories")

        for joint_idx in range(q_vec.shape[1]):
            plt.plot(t_vec, q_vec[:, joint_idx], label=f"Joint {joint_idx}")

        plt.xlabel("Time [s]")
        plt.ylabel("Joint Position [rad]")
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_frames(self,  tforms: list,q_start: np.ndarray):
        """
        Visualize the frames along the Cartesian path.
        """
        self.viz.displayFrames(True, frame_ids=[self.model.getFrameId(self.target_frame)])
        self.viz.display(np.zeros(self.model.nq))  # Display initial configuration
        self.viz.display(q_start)
        visualize_frames(self.viz, "cartesian_plan", tforms, line_length=0.05, line_width=1)
        time.sleep(4)
    def servoJ(self, q_target: np.ndarray, delta_t: float = DELTA_T):
        """
        Python版本的servoJ接口，限制速度并更新关节位置状态self.q

        参数:
            q_target: 目标关节角 (np.ndarray)
            delta_t: 控制周期（单位：秒），默认1ms
        返回:
            0 成功，-1 失败
        """
        # **位置限制检查（可选，如果你定义了self.q_limits）**
        if hasattr(self, "q_limits"):  # self.q_limits: (n_joints, 2)
            for i in range(len(q_target)):
                qmin, qmax = self.q_limits[i]
                if not (qmin <= q_target[i] <= qmax):
                    print(f"[servoJ] 目标位置超限: Joint {i} = {np.degrees(q_target[i]):.2f} deg")
                    return -1

        # **速度限制检查**
        q_offset = np.abs(q_target - self.q)
        for i in range(len(q_target)):
            v_max = self.q_vel_limits[i]  # rad/s
            if q_offset[i] > v_max * delta_t:
                print(q_offset, q_target, self.q)
                print(f"[servoJ] 关节速度超限: Joint {i} = {np.degrees(q_offset[i]/delta_t):.2f} deg/s")
                return -1

        # **位置伺服更新**
        self.q = q_target.copy()

        # **可选：触发可视化或仿真接口**
        if hasattr(self, "viz"):
            self.viz.display(self.q)
        time.sleep(delta_t)

        return 0
    def servol(self, pose_target: pinocchio.SE3, q_start: np.ndarray, duration: float = DELTA_T):
        """
        输入目标位姿（SE3），求解逆解并下发。
        """
        q_sol = self.inverse_kinematics(self.target_frame, pose_target.translation, q_start)
        if q_sol is not None:
            self.servoJ(q_sol, duration)
        else:
            print("[Robot] servol failed: no valid IK solution.")
    
    def MoveJ(self, q_target: np.ndarray, v_max: float = 1.0, a_max: float = 2.0, dt: float = DELTA_T,traj_rviz: bool = False):
            """
            使用T型速度规划执行关节空间插值运动，调用servoj逐步下发。
            并记录位置、速度、加速度轨迹用于可视化。
            """
            q_start = self.q.copy()
            delta_q = q_target - q_start
            max_delta = np.max(np.abs(delta_q))
            # print(f"max_delta: {max_delta}, delta_q: {delta_q}")

            for i in range(len(q_target)):
                v_max_i = self.q_vel_limits[i]
                a_max_i = self.q_acc_limits[i]
                if np.abs(v_max) > v_max_i:
                    print(f"[Robot] MoveJ failed: velocity limit exceeded for joint {i}.")
                    return
                if np.abs(a_max) > a_max_i:
                    print(f"[Robot] MoveJ failed: acceleration limit exceeded for joint {i}.")
                    return

            if max_delta < 1e-9:
                print("[Robot] Already at target position.")
                return
            v_max_norm = v_max / max_delta  # 归一化速度
            a_max_norm = a_max / max_delta  # 归一化加速度

            t_steady=0.0
            t_total=0.0
            t_acc=0.0
            # 轨迹规划阶段计算 --------------------------------------------------
            # 计算临界速度（判断能否达到设定速度）
            v_critical = np.sqrt(a_max_norm * 1.0)  # 最大可能达到的速度
            t_acc = v_max_norm / a_max_norm        # 加速阶段时间
            s_acc = 0.5 * a_max_norm * t_acc**2    # 加速阶段位移
            # t_steady=0.0
            # t_total=0.0
            # t_acc=0.0
            # 判断轨迹类型（梯形/三角形）
            if s_acc * 2 >  1.0:  # 无法达到设定速度，使用三角形轨迹
                t_acc = np.sqrt(1.0 / a_max_norm)
                t_total = 2 * t_acc
                has_steady = False
            else:                # 使用梯形轨迹
                t_steady = (1.0 - 2*s_acc) / v_max_norm  # 匀速阶段时间
                t_total = 2*t_acc + t_steady
                has_steady = True
            # print("tacc:",t_acc, "tsteady:", t_steady, "ttotal:", t_total)
            # print("0.5 * a * current_time**2",0.5 * a_max_norm * t_acc**2)

            # 轨迹生成 ----------------------------------------------------------
            time_traj = []
            s_traj, v_traj, a_traj = [], [], []
            q_traj, qd_traj, qdd_traj = [], [], []

            current_time = 0.0
            while current_time <= t_total:
                # 计算当前阶段参数
                if has_steady:
                    if current_time < t_acc:  # 加速阶段
                        a = a_max_norm
                        v = a_max_norm * current_time
                        s = 0.5 * a * current_time**2
                    elif current_time < t_acc + t_steady:  # 匀速阶段
                        a = 0.0
                        v = v_max_norm
                        s = s_acc + v_max_norm*(current_time - t_acc)
                    else:  # 减速阶段
                        dec_time = current_time - (t_acc + t_steady)
                        a = -a_max_norm
                        v = v_max_norm +a *dec_time
                        s = s_acc + v_max_norm*t_steady + v_max_norm*dec_time + 0.5*a*dec_time**2
                        # print("dec_time",dec_time)
                        # print("s",s)
                        # print("v_max_norm*dec_time + 0.5*a*dec_time**2",v*dec_time + 0.5*a*dec_time**2)
                        # print("tacc:",t_acc, "tsteady:", t_steady, "ttotal:", t_total)
                else:  # 三角形轨迹
                    
                    if current_time < t_acc:  # 加速阶段
                        a = a_max_norm
                        v = a * current_time
                        s = 0.5 * a * current_time**2
                    else:  # 减速阶段
                        dec_time = current_time - t_acc
                        a = -a_max_norm
                        v = a_max_norm*t_acc + a*dec_time
                        s = 0.5*a_max_norm*t_acc**2 + a_max_norm*t_acc*dec_time + 0.5*a*dec_time**2
                        
                # 边界保护
                # s = np.clip(s, 0.0, 1.0)
                # v = v if s < 1.0 else 0.0
                # a = a if s < 1.0 else 0.0

                # 转换到实际关节空间
                q_current = q_start + s * delta_q
                qd_current = v * delta_q
                qdd_current = a * delta_q
                

                time_traj.append(current_time)
                s_traj.append(s)
                v_traj.append(v)
                a_traj.append(a)
                q_traj.append(q_current.copy())
                qd_traj.append(qd_current.copy())
                qdd_traj.append(qdd_current.copy())
                result=self.servoJ(q_current, dt)
                if result != 0:
                    print("[MoveJ] servoJ失败，中止MoveJ")
                    break    
                current_time += dt
                


            
            # for i in range(len(q_traj)):
            #     result=self.servoJ(q_traj[i], dt)
            #     if result != 0:
            #         print("[MoveJ] servoJ失败，中止MoveJ")
            #         break    
            # print("q_current",q_current,"q_target",s)
            # 可视化
            if traj_rviz:
                self.plot_trajectory(time_traj, q_traj, qd_traj, qdd_traj)
            
            
    def MoveL(self, tforms: list, dt: float = DELTA_T, max_retries: int = 10):
        """
        基于笛卡尔路径规划执行 moveL（线性轨迹运动），逐点调用 servol。
        
        参数：
            tforms: list of pinocchio.SE3
                目标末端位姿序列（通常为线性插值生成）
            dt: float
                每个点的时间间隔，单位：秒
            max_retries: int
                IK 求解最大尝试次数
        """
        # 当前起始关节角
        q_start = self.q.copy()
        
        # 调用笛卡尔路径规划器
        tforms_to_show, t_vec, q_vec, success = self.cartesian_planning(
            q_start=q_start,
            tforms=tforms,
            dt=dt,
            max_retries=max_retries
        )

        if not success:
            print("[Robot] moveL 失败：笛卡尔路径规划不成功")
            return

        # 按照轨迹中的位姿逐点追踪执行
        for idx in range(1, q_vec.shape[1]):
            self.servoJ(q_vec[:, idx], dt)
        # for i, pose in enumerate(tforms_to_show):
        #     self.servol(pose_target=pose, q_start=self.q, duration=dt)
    def plot_trajectory(self, time_series, q_traj, qd_traj, qdd_traj):
        q_traj = np.array(q_traj)
        qd_traj = np.array(qd_traj)
        qdd_traj = np.array(qdd_traj)

        num_joints = q_traj.shape[1]

        plt.figure(figsize=(12, 8))

        for j in range(num_joints):
            plt.subplot(3, 1, 1)
            plt.plot(time_series, q_traj[:, j], label=f'Joint {j+1}')
            plt.ylabel("Position (rad)")
            plt.title("Joint Position")

            plt.subplot(3, 1, 2)
            plt.plot(time_series, qd_traj[:, j], label=f'Joint {j+1}')
            plt.ylabel("Velocity (rad/s)")
            plt.title("Joint Velocity")

            plt.subplot(3, 1, 3)
            plt.plot(time_series, qdd_traj[:, j], label=f'Joint {j+1}')
            plt.ylabel("Acceleration (rad/s²)")
            plt.title("Joint Acceleration")
            plt.xlabel("Time (s)")

        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()
class DianaRobot(Robot):
    def __init__(self,target_frame, visualizer: bool = True):
        pinocchio_model_dir = Path(__file__).parent.parent.parent/ "assets"/"urdf"
        print("pinocchio_model_dir:", pinocchio_model_dir.as_posix())
        model_path = pinocchio_model_dir  
        print(model_path)
        # 构建URDF绝对路径并转换为字符串
        urdf_model_path = (
            pinocchio_model_dir 
            
            / "diana7_description" 
            / "urdf" 
            / "diana_v2.urdf"
        ).resolve()  # 解析符号链接和相对路径
        urdf_path = urdf_model_path.as_posix()  # 转换为POSIX路径字符串
        if not urdf_model_path.exists():
            raise FileNotFoundError(f"URDF file not found at: {urdf_path}")
        # 2. 模型加载修正
        # 获取mesh资源目录（转换为字符串）
        mesh_dir =model_path.resolve().as_posix()

        # 初始化父类（加载模型、初始化Meshcat）
        super().__init__(urdf_path, mesh_dir, vizualizer=visualizer,target_frame=target_frame)

        # 设置关节约束（从机器人控制器或官方SDK读取）
        dblMinPos = np.array([-3.124139, -1.570796, -3.124139, 0.000000, -3.124139, -3.124139, -3.124139])
        dblMaxPos = np.array([3.124139, 1.570796, 3.124139, 3.054326, 3.124139, 3.124139, 3.124139])

        # 最大关节速度 (rad/s)
        dblMaxVel = np.array([2.967060, 2.617994, 2.617994, 2.617994, 3.141593, 3.141593, 3.839724])*100

        # 最大关节加速度 (rad/s²)
        dblMaxAcc = np.array([10.780899, 8.733977, 8.931373, 8.794889, 14.885564, 14.762169, 14.803359])*100
        # 1. 位置限位
        self.model.lowerPositionLimit = dblMinPos  # 设置关节下限
        self.model.upperPositionLimit = dblMaxPos  # 设置关节上限
        self.q_vel_limits  = dblMaxVel
        self.q_acc_limits = dblMaxAcc
        # 1. 位置限位
        self.q_limits=np.array([dblMinPos, dblMaxPos]).T  # 最小最大位置限制（可选）

        # 2. 速度限位 (需手动扩展模型属性)
        self.model.velocityLimit = dblMaxVel       # 设置关节速度限位
        self.model.accelerationLimit = dblMaxAcc
class DianaMujocoEnv(Robot):
    def __init__(self,target_frame, visualizer: bool = True):
        pinocchio_model_dir = Path(__file__).parent.parent.parent/ "assets"/"urdf"
        print("pinocchio_model_dir:", pinocchio_model_dir.as_posix())
        model_path = pinocchio_model_dir  
        print(model_path)
        # 构建URDF绝对路径并转换为字符串
        urdf_model_path = (
            pinocchio_model_dir 
            
            / "diana7_description" 
            / "urdf" 
            / "diana_v2.urdf"
        ).resolve()  # 解析符号链接和相对路径
        urdf_path = urdf_model_path.as_posix()  # 转换为POSIX路径字符串
        if not urdf_model_path.exists():
            raise FileNotFoundError(f"URDF file not found at: {urdf_path}")
        # 2. 模型加载修正
        # 获取mesh资源目录（转换为字符串）
        mesh_dir =model_path.resolve().as_posix()

        # 初始化父类（加载模型、初始化Meshcat）
        super().__init__(urdf_path, mesh_dir, vizualizer=visualizer,target_frame=target_frame)

        # 设置关节约束（从机器人控制器或官方SDK读取）
        dblMinPos = np.array([-3.124139, -1.570796, -3.124139, 0.000000, -3.124139, -3.124139, -3.124139])
        dblMaxPos = np.array([3.124139, 1.570796, 3.124139, 3.054326, 3.124139, 3.124139, 3.124139])

        # 最大关节速度 (rad/s)
        dblMaxVel = np.array([2.967060, 2.617994, 2.617994, 2.617994, 3.141593, 3.141593, 3.839724])

        # 最大关节加速度 (rad/s²)
        dblMaxAcc = np.array([10.780899, 8.733977, 8.931373, 8.794889, 14.885564, 14.762169, 14.803359])
        # 1. 位置限位
        self.model.lowerPositionLimit = dblMinPos  # 设置关节下限
        self.model.upperPositionLimit = dblMaxPos  # 设置关节上限
        self.q_vel_limits  = dblMaxVel
        self.q_acc_limits = dblMaxAcc
        # 1. 位置限位
        self.q_limits=np.array([dblMinPos, dblMaxPos]).T  # 最小最大位置限制（可选）

        # 2. 速度限位 (需手动扩展模型属性)
        self.model.velocityLimit = dblMaxVel       # 设置关节速度限位
        self.model.accelerationLimit = dblMaxAcc

        # MuJoCo环境初始化
        self.sim_hz = 500
        self.control_hz = 25
        self.latest_action = None
        self.render_cache = None
        
        # 加载MuJoCo模型
        mujoco_model_dir = Path(__file__).parent.parent.parent / "assets"/"mujoco"
        mujoco_model_path = (mujoco_model_dir / "diana7" / "scene.xml").resolve()
        self.mj_model = mujoco.MjModel.from_xml_path(mujoco_model_path.as_posix())
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # 关节名称列表（需与MuJoCo模型中的关节顺序一致）
        self.diana_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4",
                                 "joint_5", "joint_6","joint_7"]
        [mj.set_joint_q(self.mj_model, self.mj_data, jn, self.q[i]) for i, jn in enumerate(self.diana_joint_names)]
        
        # 初始化关节位置
        
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        # 查看器句柄
       
        self.height = 256
        self.width = 256
        self.fovy = np.pi / 4
        self.camera_matrix = np.eye(3)
        self.camera_matrix_inv = np.eye(3)
        self.num_points = 4096

        self.mj_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_renderer_depth = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_renderer_depth.update_scene(self.mj_data, 0)
        self.mj_renderer_depth.enable_depth_rendering()
        self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.camera_matrix = np.array([
            [self.height / (2.0 * np.tan(self.fovy / 2.0)), 0.0, self.width / 2.0],
            [0.0, self.height / (2.0 * np.tan(self.fovy / 2.0)), self.height / 2.0],
            [0.0, 0.0, 1.0]
        ])
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        self.step_num = 0
        # self.observation = self._get_obs()


    def _get_obs(self):
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_renderer_depth.update_scene(self.mj_data, 0)
        img = self.mj_renderer.render()
        depth = self.mj_renderer_depth.render()

        point_cloud = np.zeros((self.height * self.width, 6))
        for h in range(self.height):
            for w in range(self.width):
                point_cloud[h * self.width + w, :3] = self.camera_matrix_inv @ np.array([w * 1.0, h * 1.0, 1.0]) * \
                                                      depth[h, w]
                point_cloud[h * self.width + w, 3:] = img[h, w, :]
        sampled_points = self.uniform_sampling(point_cloud)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(sampled_points[:, :3])
        # pcd.colors = o3d.utility.Vector3dVector(sampled_points[:, 3:] / 255.0)
        # o3d.visualization.draw_geometries([pcd])

        for i in range(len(self.diana_joint_names)):
            self.q[i] = mj.get_joint_q(self.mj_model, self.mj_data, self.diana_joint_names[i])
        self.robot_T = self.robot.fkine(self.robot_q)
        agent_pos = self.robot.fkine(self.robot_q).t
        obs = {
            'agent_pos': agent_posset_joint_positions,
            'point_cloud': sampled_points
        }
        self.render_cache = img
        return obs
    def render(self, mode):
        if self.render_cache is None:
            self._get_obs()
        return self.render_cache
    # def close(self):
    #     if self.mj_viewer is not None:
    #         self.mj_viewer.close()
    #     if self.mj_renderer is not None:
    #         self.mj_renderer.close()
    #     if self.mj_renderer_depth is not None:
    #         self.mj_renderer_depth.close()
    

    def servoJ(self, q_target: np.ndarray, delta_t: float = DELTA_T):
        """
        Python版本的servoJ接口，限制速度并更新关节位置状态 self.q，同时调用 MuJoCo 仿真。

        参数:
            q_target: 目标关节角 (np.ndarray)
            delta_t: 控制周期（单位：秒），默认1ms
        返回:
            0 成功，-1 失败
        """
        # 检查关节位置限制（如果有定义）
        if hasattr(self, "q_limits"):  # self.q_limits: (n_joints, 2)
            for i in range(len(q_target)):
                qmin, qmax = self.q_limits[i]
                if not (qmin <= q_target[i] <= qmax):
                    print(f"[servoJ] 目标位置超限: Joint {i} = {np.degrees(q_target[i]):.2f} deg")
                    return -1

        # 检查关节速度限制
        q_offset = np.abs(q_target - self.q)
        for i in range(len(q_target)):
            v_max = self.q_vel_limits[i]  # rad/s
            if q_offset[i] > v_max * delta_t:
                print(q_offset, q_target, self.q)
                print(f"[servoJ] 关节速度超限: Joint {i} = {np.degrees(q_offset[i]/delta_t):.2f} deg/s")
                return -1

        # 更新控制命令
        self.q = q_target.copy()

        if hasattr(self, "mj_data") and hasattr(self, "mj_model"):
            # 更新模拟控制信号（假设前7个为目标关节）
            self.mj_data.ctrl[:len(self.q)] = self.q

            # 执行一次仿真步进
            mujoco.mj_step(self.mj_model, self.mj_data)

        # 可视化更新
        if hasattr(self, "viz"):
            self.viz.display(self.q)
        if hasattr(self, "mj_renderer") and hasattr(self, "mj_viewer"):
            self.mj_renderer.update_scene(self.mj_data, 0)
            self.mj_viewer.sync()
        time.sleep(delta_t)
        return 0

class _DualArmMujocoEnv(Robot):
    def __init__(self, target_frame, visualizer: bool = True):
        # URDF 路径设置
        pinocchio_model_dir = Path(__file__).parent.parent.parent/ "assets"/"urdf"
        print("pinocchio_model_dir:", pinocchio_model_dir.as_posix())
        model_path = pinocchio_model_dir  
        print(model_path)
        # 构建URDF绝对路径并转换为字符串
        urdf_model_path = (
            pinocchio_model_dir 
            
            / "diana7_description" 
            / "urdf" 
            / "diana_v2.urdf"
        ).resolve()  # 解析符号链接和相对路径
        urdf_path = urdf_model_path.as_posix()  # 转换为POSIX路径字符串
        if not urdf_model_path.exists():
            raise FileNotFoundError(f"URDF file not found at: {urdf_path}")
        # 2. 模型加载修正
        # 获取mesh资源目录（转换为字符串）
        mesh_dir =model_path.resolve().as_posix()

        # 初始化父类
        super().__init__(urdf_path, mesh_dir, vizualizer=visualizer, target_frame=target_frame)

        # 左右臂关节名
        self.left_joint_names = [f"L_joint_{i}" for i in range(1, 8)]
        self.right_joint_names = [f"R_joint_{i}" for i in range(1, 8)]


        # 关节限制参数
        self.q_limits = {
            "L": np.array([[-3.124139, 3.124139], [-1.570796, 1.570796], [-3.124139, 3.124139],
                           [0.000000, 3.054326], [-3.124139, 3.124139], [-3.124139, 3.124139],
                           [-3.124139, 3.124139]]),
            "R": np.array([[-3.124139, 3.124139], [-1.570796, 1.570796], [-3.124139, 3.124139],
                           [0.000000, 3.054326], [-3.124139, 3.124139], [-3.124139, 3.124139],
                           [-3.124139, 3.124139]])
        }
        self.q_vel_limits = np.array([2.967060, 2.617994, 2.617994,
                                      2.617994, 3.141593, 3.141593, 3.839724])
        self.q_acc_limits = np.array([10.780899, 8.733977, 8.931373,
                                      8.794889, 14.885564, 14.762169, 14.803359])

        # 设置模型限制（假设左右臂限制一致）
        self.model.lowerPositionLimit = np.concatenate(
            (self.q_limits["L"][:, 0], self.q_limits["R"][:, 0]))
        self.model.upperPositionLimit = np.concatenate(
            (self.q_limits["L"][:, 1], self.q_limits["R"][:, 1]))
        self.model.velocityLimit = np.concatenate((self.q_vel_limits, self.q_vel_limits))
        self.model.accelerationLimit = np.concatenate((self.q_acc_limits, self.q_acc_limits))

        

        # 控制频率参数
        self.sim_hz = 500
        self.control_hz = 25
        self.latest_action = None
        self.render_cache = None

        # 加载 MuJoCo 模型
        mujoco_model_dir = Path(__file__).parent.parent.parent / "assets" / "mujoco"
        mujoco_model_path = (mujoco_model_dir / "diana7" / "scene_dual.xml").resolve()
        self.mj_model = mujoco.MjModel.from_xml_path(mujoco_model_path.as_posix())
        self.mj_data = mujoco.MjData(self.mj_model)

        # 初始化左右臂
        self.q_init_left = np.zeros(7)
        self.q_init_right = np.zeros(7)
        self.reset_arms(self.q_init_left, self.q_init_right)

        mujoco.mj_forward(self.mj_model, self.mj_data)
   
    def reset_arms(self, q_left, q_right):
        """初始化双臂关节位置"""
        # for i, name in enumerate(self.left_joint_names):
        #     mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        #     mujoco.set_joint_qpos(self.mj_model, self.mj_data, name, q_left[i])

        # for i, name in enumerate(self.right_joint_names):
        #     mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        #     mujoco.set_joint_qpos(self.mj_model, self.mj_data, name, q_right[i])
        for i, name in enumerate(self.left_joint_names):
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = self.mj_model.jnt_qposadr[joint_id]
            self.mj_data.qpos[qpos_adr] = q_left[i]

        for i, name in enumerate(self.right_joint_names):
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = self.mj_model.jnt_qposadr[joint_id]
            self.mj_data.qpos[qpos_adr] = q_right[i]

        mujoco.mj_forward(self.mj_model, self.mj_data)  # 刷新 forward kinematics

    def step(self, action_left, action_right):
        """控制左右臂动作"""
        assert len(action_left) == 7 and len(action_right) == 7
        # 拼接左右臂动作
        full_action = np.concatenate([action_left, action_right])

        # 设置控制输入（假设 ctrl 顺序和左右臂顺序一致）
        self.mj_data.ctrl[:len(full_action)] = full_action

        # 执行一次仿真步进
        mujoco.mj_step(self.mj_model, self.mj_data)

        # 可视化更新
        if hasattr(self, "mj_renderer") and hasattr(self, "mj_viewer"):
            self.mj_renderer.update_scene(self.mj_data, 0)
            self.mj_viewer.sync()


    def get_joint_positions(self):
        """返回左右臂当前关节位置"""
        q_left = np.array([mujoco.get_joint_qpos(self.mj_model, self.mj_data, name)
                           for name in self.left_joint_names])
        q_right = np.array([mujoco.get_joint_qpos(self.mj_model, self.mj_data, name)
                            for name in self.right_joint_names])
        return q_left, q_right
    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer.close()
        if self.mj_renderer is not None:
            self.mj_renderer.close()
        if self.mj_renderer_depth is not None:
            self.mj_renderer_depth.close()
import threading
class DualArmMujocoEnv():
    def __init__(self, target_frame, visualizer=False):
        self.left_arm = DianaRobot(target_frame=target_frame, visualizer=visualizer)
        self.right_arm = DianaRobot(target_frame=target_frame, visualizer=visualizer)
        # 左右臂关节名
        self.left_joint_names = [f"L_joint_{i}" for i in range(1, 8)]
        self.right_joint_names = [f"R_joint_{i}" for i in range(1, 8)]
        # 控制频率参数
        self.sim_hz = 500
        self.control_hz = 25
        self.latest_action = None
        self.render_cache = None

        # 加载 MuJoCo 模型
        mujoco_model_dir = Path(__file__).parent.parent.parent / "assets" / "mujoco"
        mujoco_model_path = (mujoco_model_dir / "diana7" / "scene_dual.xml").resolve()
        self.mj_model = mujoco.MjModel.from_xml_path(mujoco_model_path.as_posix())
        self.mj_data = mujoco.MjData(self.mj_model)

        # 初始化左右臂
        self.reset_arms(self.left_arm.q, self.right_arm.q)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.height = 256
        self.width = 256
        self.fovy = np.pi / 4
        self.camera_matrix = np.eye(3)
        self.camera_matrix_inv = np.eye(3)
        self.num_points = 4096

        self.mj_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_renderer_depth = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_renderer_depth.update_scene(self.mj_data, 0)
        self.mj_renderer_depth.enable_depth_rendering()
        self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.camera_matrix = np.array([
            [self.height / (2.0 * np.tan(self.fovy / 2.0)), 0.0, self.width / 2.0],
            [0.0, self.height / (2.0 * np.tan(self.fovy / 2.0)), self.height / 2.0],
            [0.0, 0.0, 1.0]
        ])
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        self.step_num = 0
        self.running = True  # 控制线程的变量
        self.delta_t = 1.0 / 60.0  # 可视化帧率
        self.t_left=None
        self.t_right=None
        # self.vis_thread = None
        self.point_cloud_queue = queue.Queue()


        self.vis_thread = threading.Thread(target=self._visual_loop)
        self.vis_thread.daemon = True
        self.vis_thread.start()
        # self._visual_point_loop()  # 主线程运行采集与控制逻辑
        
    def reset_arms(self, q_left, q_right):
        """初始化双臂关节位置"""
        # for i, name in enumerate(self.left_joint_names):
        #     mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        #     mujoco.set_joint_qpos(self.mj_model, self.mj_data, name, q_left[i])

        # for i, name in enumerate(self.right_joint_names):
        #     mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        #     mujoco.set_joint_qpos(self.mj_model, self.mj_data, name, q_right[i])
        for i, name in enumerate(self.left_joint_names):
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = self.mj_model.jnt_qposadr[joint_id]
            self.mj_data.qpos[qpos_adr] = q_left[i]

        for i, name in enumerate(self.right_joint_names):
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = self.mj_model.jnt_qposadr[joint_id]
            self.mj_data.qpos[qpos_adr] = q_right[i]

        mujoco.mj_forward(self.mj_model, self.mj_data)  # 刷新 forward kinematics
    
    def _generate_point_cloud(self, rgb_image, depth_image):
        h, w = np.indices((self.height, self.width))
        coordinates = np.stack([w, h, np.ones_like(w)], axis=-1)
        point_cloud = coordinates @ self.camera_matrix_inv.T * depth_image[..., np.newaxis]
        point_cloud = point_cloud.reshape(-1, 3)
        rgb = rgb_image.reshape(-1, 3) / 255.0
        mask = np.isfinite(point_cloud).all(axis=1) & (point_cloud[:, 2] > 0)
        return point_cloud[mask], rgb[mask]
    def _visual_loop(self):
        # point_cloud = np.zeros((self.height * self.width, 6))
        
        
        while self.running:
            self.step(self.left_arm.q, self.right_arm.q)
            self.mj_renderer_depth.update_scene(self.mj_data, 0)

            # 获取图像
            rgb_image = self.mj_renderer.render()
            depth_image = self.mj_renderer_depth.render()
            import cv2
            cv2.imshow("RGB", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = normalized_depth.astype(np.uint8)
            cv2.imshow("Depth", depth_uint8)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # ========== 向量化点云生成 ==========
            h, w = np.indices((self.height, self.width))
            coordinates = np.stack([w, h, np.ones_like(w)], axis=-1)
            point_cloud = coordinates @ self.camera_matrix_inv.T * depth_image[..., np.newaxis]
            point_cloud = point_cloud.reshape(-1, 3)
            rgb = rgb_image.reshape(-1, 3) / 255.0
            # print("RGB min:", rgb.min(), "max:", rgb.max())

            point_cloud = np.hstack([point_cloud, rgb])
            sampled_points = self.uniform_sampling(point_cloud)
            # mask = np.isfinite(point_cloud).all(axis=1) & (point_cloud[:, 2] > 0)
            # point_cloud = point_cloud[mask]
            # rgb = rgb[mask]
            # 放到queue中
            # points, colors = self._generate_point_cloud(rgb_image, depth_image)
            points = sampled_points[:, :3]
            colors = sampled_points[:, 3:]
            # print("Sample colors:", colors[:5])


            self.point_cloud_queue.put((points, colors))
            

            # time.sleep(0.099)

            time.sleep(DELTA_T)
        
        # cv2.destroyAllWindows()
    def uniform_sampling(self, point_cloud):

        condition = point_cloud[:, 2] < 3.0
        filtered_points = point_cloud[condition, :]
        indices = np.random.permutation(filtered_points.shape[0])[:self.num_points]
        sampled_points = filtered_points[indices, :]
        return sampled_points
    def _visual_point_loop(self):
        vis = o3d.visualization.Visualizer()
        # vis.create_window()
        vis.create_window(width=800, height=600)
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)
        # vis.run()
        while self.running:
            try:
                points, colors = self.point_cloud_queue.get(timeout=0.05)
                pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
                vis.add_geometry(pcd)
                # vis.update_geometry(pcd)
            except queue.Empty:
                print("Queue empty")

                pass
            # 可视化更新


            # vis.poll_events()
            # vis.update_renderer()
            

            vis.run()
        vis.destroy_window()
    def stop(self):

        self.running = False
        time.sleep(5)
        self.close()
        # self.t_left.join()
        # self.t_right.join()
        if hasattr(self, "mj_renderer"):
            self.mj_renderer.close()
            self.mj_renderer = None
        if hasattr(self, "mj_viewer"):
            self.mj_viewer = None  # 或调用关闭方法
        if hasattr(self, "vis_thread"):
            self.vis_thread.join()

    def step(self, action_left, action_right):
        """控制左右臂动作"""
        assert len(action_left) == 7 and len(action_right) == 7
        # 拼接左右臂动作
        full_action = np.concatenate([action_left, action_right])

        # 设置控制输入（假设 ctrl 顺序和左右臂顺序一致）
        self.mj_data.ctrl[:len(full_action)] = full_action

        # 执行一次仿真步进
        mujoco.mj_step(self.mj_model, self.mj_data)

        # 可视化更新
        if hasattr(self, "mj_renderer") and hasattr(self, "mj_viewer"):
            self.mj_renderer.update_scene(self.mj_data, 0)
            self.mj_viewer.sync()


    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer.close()
        if self.mj_renderer is not None:
            self.mj_renderer.close()
        if self.mj_renderer_depth is not None:
            self.mj_renderer_depth.close()
    def MoveJ_dual_arm(self, q_left, q_right, v_max=1.0, a_max=1.0, dt=0.01):
        """同步移动双臂到指定位置"""
        def move_left():
            self.left_arm.MoveJ(q_left, v_max=v_max, a_max=a_max, dt=dt)

        def move_right():
            self.right_arm.MoveJ(q_right, v_max=v_max, a_max=a_max, dt=dt)

        # 创建线程
        self.t_left = threading.Thread(target=move_left)
        self.t_right = threading.Thread(target=move_right)

        # 启动线程
        self.t_left.start()
        self.t_right.start()

        # 等待两个线程都完成
        self.t_left.join()
        self.t_right.join()
    def MoveJ_left_arm(self, q_left, v_max=1.0, a_max=1.0):
        """移动左臂到指定位置"""
        
        def move_left():
            self.left_arm.MoveJ(q_left, v_max=v_max, a_max=a_max)
        t_left = threading.Thread(target=move_left)
        t_left.start()
        t_left.join()
    def MoveJ_right_arm(self, q_right, v_max=1.0, a_max=1.0):
        """移动右臂到指定位置"""
        def move_right():
            self.right_arm.MoveJ(q_right, v_max=v_max, a_max=a_max)
        t_right = threading.Thread(target=move_right)
        t_right.start()
        t_right.join()
    def read_csv(self,filename):
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            return [list(map(float, row)) for row in reader]
    def getImage(self):
        rgb_image = self.mj_renderer.render()
        return rgb_image


    def wave(self):
        print("Start waving....")

        jointid0 = [0.302601, 0.798305, -0.821039, 1.55999, 0.349412, 0.270819, 0.612382]
        jointid1 = [-1.85347, 1.12013, 1.08138, 1.26592, -0.647975, 0.23362, 0.0481165]
        jointid2 = [-2.38536, 0.839411, 1.04073, 2.733, -0.0721692, -0.19043, 0.0479487]

        vel = 0.4
        acc = 0.4
        self.MoveJ_dual_arm(jointid1, jointid0, vel, acc)
       

        
        
        time.sleep(0.05)
        print("Start to read csv file")
        filename = "/home/rocos/Documents/GitHub/RoboSimXCtrl/robot/test/point.csv"
        points = self.read_csv(filename)
        joints = points[0]
        # 变成numpy数组
        joints = np.array(joints)
        self.MoveJ_dual_arm( jointid2, joints,vel, acc)

        # ret = robot.left_arm.MoveJ(joints, vel, acc)
        # robot.right_arm.MoveJ(jointid2, vel, acc)

        
        
        
        while self.running:
            # 正向遍历
            for point in points:
                if not (self.running):
                    return
                point=np.array(point)
                robot.right_arm.servoJ(point)
                time.sleep(0.008)

            # 反向遍历
            for point in reversed(points):
                if not (self.running):
                    return
                point=np.array(point)
                robot.right_arm.servoJ(point)
                time.sleep(0.008)
    def run(self):
        self.vis_thread = threading.Thread(target=self._visual_loop)
        self.vis_thread.daemon = True
        self.vis_thread.start()
        self._visual_point_loop()  # 主线程运行采集与控制逻辑
        self.vis_thread.join()

# 使用示例
import faulthandler

if __name__ == "__main__":
    faulthandler.enable()
    robot = DualArmMujocoEnv(target_frame="link_7")
    time.sleep(10)
    jointid0 = [0.302601, 0.798305, -0.821039, 1.55999, 0.349412, 0.270819, 0.612382]
    jointid1 = [-1.85347, 1.12013, 1.08138, 1.26592, -0.647975, 0.23362, 0.0481165]
    # q_start_left = np.array([-1.94852,	1.11936,	0.975168,	2.43056	,-0.758793	,-0.0871495	,0.0481404])
    # q_start_right = np.array([0.243771	,1.15619	,-0.787016	,2.48789	,0.347578,	0.230049,	0.612406])
    robot.MoveJ_dual_arm(jointid1, jointid0, v_max=1.0, a_max=1.0, dt=DELTA_T)


    # robot.wave()

    # robot.left_arm.MoveJ(q_start, v_max=1.0, a_max=1.0, dt=DELTA_T)
    # robot.left_arm.MoveJ(q_start, v_max=1.8, a_max=1.0, dt=DELTA_T)
    
    
    
    
    
    
    
    # robot.run()
    # # robot.stop()
    robot._visual_point_loop()
    robot.stop()






    # env = DualArmMujocoEnv(target_frame="base", visualizer=False)

    # # 打印初始状态
    # print("初始关节角（左臂）:", env.q_init_left)
    # print("初始关节角（右臂）:", env.q_init_right)

    # # 设定一个动作（简单地抬高每个关节 0.1 rad）
    # delta_q = np.ones(7) * 0.1

    # # 更新关节位置（例如做一次动作控制）
    # new_q_l = env.q_init_left + delta_q
    # new_q_r = env.q_init_right - delta_q  # 假设右臂反方向移动

    # # 应用到 MuJoCo 模拟中
    # env.step(new_q_l, new_q_r)


    # 向前模拟一步
    # mujoco.mj_step(env.mj_model, env.mj_data)

    # 可视化更新（如果开启了 MeshCat 可视化）
    # if env.visualizer:
    #     env.meshcat_visualizer.display(env.q)  # 如果你有 pinocchio 机器人可视化设置

    # 打印新的关节角

    # print("更新后关节角（左臂）:", new_q_l)
    # print("更新后关节角（右臂）:", new_q_r)
   
    # 示例2：使用MuJoCo查看器
    # env = DianaMujocoEnv(target_frame="link_7")
    
    # with mujoco.viewer.launch_passive(env.mj_model, env.mj_data) as env.viewer:
    #     for _ in range(10000):
    #         mujoco.mj_step(env.mj_model, env.mj_data)
    #         env.viewer.sync()
    #         time.sleep(0.01)