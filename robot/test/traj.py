# IDLE → MoveJ() → MOVING → 空格键 → PAUSING → PAUSED
#           ↑                                     ↓
#           ←────────────── r 键 ←───────────────┘
#           ↓                                     ↑
#           └→ MOVING(重启) → 空格键 → PAUSING → PAUSED
#                               ↓
#             ←────────────── r 键 ←─────────────┘


import numpy as np
import time
import threading
from scipy.spatial.transform import Rotation
import pinocchio
from pinocchio.visualize import MeshcatVisualizer
from pyroboplan.core.utils import extract_cartesian_pose
from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions
from pyroboplan.planning.cartesian_planner import CartesianPlanner, CartesianPlannerOptions
from pyroboplan.visualization.meshcat_utils import visualize_frames
import matplotlib.pyplot as plt
from pathlib import Path
from ruckig import InputParameter, OutputParameter, Result, Ruckig
from pynput import keyboard  # 用于键盘监听

DELTA_T = 0.001

class Robot:
    def __init__(
        self, 
        urdf_path: str, 
        mesh_dir: str,
        vizualizer: bool = False,
        target_frame: str = "link_7",
    ):
        """
        初始化机器人，并添加状态管理和中断标志。
        """
        self.model, self.collision_model, self.visual_model = self.load_robot(urdf_path, mesh_dir)
        self.data = self.model.createData()
        self.collision_data = self.collision_model.createData()
        self.q = np.zeros(self.model.nq)
        self.target_frame = target_frame
        self.q_vel_limits = np.array([1.5] * self.model.nq)
        self.q_limits = np.array([[-np.pi]*self.model.nq, [np.pi]*self.model.nq]).T
        self.q_acc_limits = np.array([10.0] * self.model.nq)
        
        # 新增：状态和中断标志
        self.state = "IDLE"  # 状态: "IDLE", "MOVING", "PAUSING", "PAUSED","RESTARTING"
        self.pause_requested = False
        self.restart_requested = False  # 新增：重启请求标志
        self.original_target = None  # 新增：保存原始目标位置
        self.current_velocity = np.zeros(self.model.nq)  # 记录当前指令速度，用于Ruckig
        self._time_traj = []
        self._q_traj = []
        self._qd_traj = []
        self._qdd_traj = []

        # 可视化初始化
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model, data=self.data)
        self.viz.initViewer(open=vizualizer)
        self.viz.loadViewerModel()
        if(vizualizer):
            self.viz.display(self.q)
            time.sleep(2)
        
    def load_robot(self, urdf_path: str, mesh_dir: str):
        """加载机器人URDF模型。"""
        model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(urdf_path, mesh_dir)
        return model, collision_model, visual_model
    
    def get_cartesian_pose(self, q: np.ndarray):
        """获取末端笛卡尔位姿。"""
        return extract_cartesian_pose(self.model, self.target_frame, q, data=self.data)
    
    def inverse_kinematics(self, target_position: np.ndarray, q_start: np.ndarray, max_retries: int = 5):
        """使用Differential IK求解逆运动学。"""
        ik = DifferentialIk(self.model, data=self.data, collision_model=self.collision_model, 
                            options=DifferentialIkOptions(max_retries=max_retries))
        q_sol = ik.solve(self.target_frame, target_position, q_start)
        return q_sol
    
    def cartesian_planning(self, q_start: np.ndarray, tforms: list, dt: float = 0.05, max_retries: int = 10):
        """笛卡尔路径规划。"""
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
            print("笛卡尔路径规划失败。")
            return None, None, None, False
        return tforms_to_show, t_vec, q_vec, success

    def display_trajectory(self, q_vec: np.ndarray, t_vec: np.ndarray):
        """可视化关节轨迹。"""
        plt.ion()
        plt.figure()
        plt.title("关节位置轨迹")

        for joint_idx in range(q_vec.shape[1]):
            plt.plot(t_vec, q_vec[:, joint_idx], label=f"关节 {joint_idx}")

        plt.xlabel("时间 [秒]")
        plt.ylabel("关节位置 [弧度]")
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_frames(self, tforms: list, q_start: np.ndarray):
        """可视化路径上的坐标系。"""
        self.viz.displayFrames(True, frame_ids=[self.model.getFrameId(self.target_frame)])
        self.viz.display(np.zeros(self.model.nq))
        self.viz.display(q_start)
        visualize_frames(self.viz, "cartesian_plan", tforms, line_length=0.05, line_width=1)
        time.sleep(4)
        
    def servoJ(self, q_target: np.ndarray, delta_t: float = DELTA_T):
        """
        关节位置伺服控制。
        返回: 0 成功, -1 失败
        """
        if not isinstance(q_target, np.ndarray):
            q_target = np.array(q_target, dtype=np.float64)
        if hasattr(self, "q_limits"):
            for i in range(len(q_target)):
                qmin, qmax = self.q_limits[i]
                if not (qmin <= q_target[i] <= qmax):
                    print(f"[servoJ] 目标位置超限: 关节 {i} = {np.degrees(q_target[i]):.2f} 度")
                    return -1

        q_offset = np.abs(q_target - self.q)
        for i in range(len(q_target)):
            v_max = self.q_vel_limits[i]
            if q_offset[i] > v_max * delta_t:
                print(f"[servoJ] 关节速度超限: 关节 {i} = {np.degrees(q_offset[i]/delta_t):.2f} 度/秒")
                return -1

        self.q = q_target.copy()

        if hasattr(self, "viz"):
            self.viz.display(self.q)
        time.sleep(delta_t)
        return 0

    def servol(self, pose_target: pinocchio.SE3, q_start: np.ndarray, duration: float = DELTA_T):
        """基于笛卡尔位姿的伺服控制。"""
        q_sol = self.inverse_kinematics(self.target_frame, pose_target.translation, q_start)
        if q_sol is not None:
            self.servoJ(q_sol, duration)
        else:
            print("[Robot] servol 失败: 无有效逆解。")

    def MoveJ(self, q_target: np.ndarray, v_max: float = 1.0, a_max: float = 2.0, dt: float = DELTA_T, traj_rviz: bool = False):
        """
        支持键盘中断的关节空间运动。
        按下空格键触发平滑暂停。
        """
        if self.state == "MOVING":
            print("[MoveJ] 机器人已在运动中，请等待完成或暂停。")
            return
            
        self.state = "MOVING"
        self.pause_requested = False
        self.restart_requested = False
        self.original_target = q_target.copy()  # 保存原始目标位置

        self._time_traj.clear()
        self._q_traj.clear()
        self._qd_traj.clear()
        self._qdd_traj.clear()
        self._should_plot = traj_rviz
        q_start = self.q.copy()
        delta_q = q_target - q_start
        max_delta = np.max(np.abs(delta_q))
        print(f"[MoveJ] 开始运动: 最大关节位移 = {max_delta:.4f} rad")

        # 关节极限检查
        for i in range(len(q_target)):
            v_max_i = self.q_vel_limits[i]
            a_max_i = self.q_acc_limits[i]
            if np.abs(v_max) > v_max_i:
                print(f"[MoveJ] 失败: 关节 {i} 速度超限。")
                self.state = "IDLE"
                return
            if np.abs(a_max) > a_max_i:
                print(f"[MoveJ] 失败: 关节 {i} 加速度超限。")
                self.state = "IDLE"
                return
                
        if max_delta < 1e-9:
            print("[MoveJ] 已在目标位置。")
            self.state = "IDLE"
            return

        # 归一化处理
        v_max_norm = v_max / max_delta
        a_max_norm = a_max / max_delta

        # 轨迹阶段计算
        v_critical = np.sqrt(a_max_norm * 1.0)
        t_acc = v_max_norm / a_max_norm
        s_acc = 0.5 * a_max_norm * t_acc**2

        if s_acc * 2 > 1.0:  # 三角轨迹
            t_acc = np.sqrt(1.0 / a_max_norm)
            t_total = 2 * t_acc
            has_steady = False
        else:  # 梯形轨迹
            t_steady = (1.0 - 2 * s_acc) / v_max_norm
            t_total = 2 * t_acc + t_steady
            has_steady = True

        # 初始化Ruckig用于紧急停止
        otg = Ruckig(self.model.nq, dt)
        inp = InputParameter(self.model.nq)
        out = OutputParameter(self.model.nq)
        
        # 设置Ruckig约束
        inp.max_velocity = self.q_vel_limits
        inp.max_acceleration = self.q_acc_limits
        inp.max_jerk = np.full(self.model.nq, 100.0)  # 加加速度限制，可根据需要调整

        # 主运动循环
        current_time = 0.0
        # time_traj, q_traj, qd_traj, qdd_traj = [], [], [], []
        
        while current_time <= t_total and self.state == "MOVING":
            # 检查暂停请求
            if self.pause_requested:
                print(f"[MoveJ] 暂停请求于 t={current_time:.3f}s，开始平滑停止...")
                self.state = "PAUSING"
                
                # 紧急停止 准备Ruckig输入：从当前状态平滑停止到零速
                # inp.current_position = self.q.copy()
                # inp.current_velocity = self.current_velocity.copy()
                # inp.current_acceleration = self.current_acceleration.copy()
                # inp.target_position = self.q.copy()  # 目标位置 = 当前位置
                # inp.target_velocity = np.zeros(self.model.nq)
                # inp.target_acceleration = np.zeros(self.model.nq)
                inp.current_position = self.current_velocity.copy()
                inp.current_velocity = self.current_acceleration.copy()
                inp.current_acceleration = np.zeros(self.model.nq)
                inp.target_position = np.zeros(self.model.nq) # 目标位置 = 当前位置
                inp.target_velocity = np.zeros(self.model.nq)
                inp.target_acceleration = np.zeros(self.model.nq)
                
                # 执行停止轨迹
                res = Result.Working
                current_q = self.q.copy()  # 记录暂停开始的们置，用于积分
                while res == Result.Working:
                    res = otg.update(inp, out)
                    if res == Result.Error:
                        print("[MoveJ] Ruckig 停止轨迹规划出错。")
                        break
                    v_desired = np.array(out.new_position, dtype=np.float64)
                    current_q += v_desired * dt 
                    current_time += dt
                    result = self.servoJ(current_q, dt)
                    if result != 0:
                        print("[MoveJ] 停止过程中 servoJ 失败。")
                        break
                    if self._should_plot:
                        # 记录时间 (从当前时间开始累加)
                        self._time_traj.append(current_time)
                        # 记录Ruckig计算出的状态
                        # 注意：out.new_position 等可能是list，需确保为numpy array
                        self._q_traj.append(np.array(current_q.copy(), dtype=np.float64))  
                        self._qd_traj.append(np.array(v_desired.copy(), dtype=np.float64))
                        self._qdd_traj.append(np.array(out.new_velocity, dtype=np.float64))
                    # 为下一次迭代更新输入
                    inp.current_position = out.new_position
                    inp.current_velocity = out.new_velocity
                    inp.current_acceleration = out.new_acceleration
                    out.pass_to_input(inp)
                
                print(f"[MoveJ] 已暂停在位置: {np.degrees(self.q)} 度")
                self.state = "PAUSED"
                self.pause_requested = False
                if self._should_plot and self._time_traj:
                    self.plot_trajectory(self._time_traj, self._q_traj, self._qd_traj, self._qdd_traj)
                return

            # 正常轨迹计算
            if has_steady:
                if current_time < t_acc:  # 加速
                    a = a_max_norm
                    v = a_max_norm * current_time
                    s = 0.5 * a * current_time**2
                elif current_time < t_acc + t_steady:  # 匀速
                    a = 0.0
                    v = v_max_norm
                    s = s_acc + v_max_norm * (current_time - t_acc)
                else:  # 减速
                    dec_time = current_time - (t_acc + t_steady)
                    a = -a_max_norm
                    v = v_max_norm + a * dec_time
                    s = s_acc + v_max_norm * t_steady + v_max_norm * dec_time + 0.5 * a * dec_time**2
            else:  # 三角轨迹
                if current_time < t_acc:  # 加速
                    a = a_max_norm
                    v = a * current_time
                    s = 0.5 * a * current_time**2
                else:  # 减速
                    dec_time = current_time - t_acc
                    a = -a_max_norm
                    v = a_max_norm * t_acc + a * dec_time
                    s = 0.5 * a_max_norm * t_acc**2 + a_max_norm * t_acc * dec_time + 0.5 * a * dec_time**2

            # 计算关节空间值
            q_current = q_start + s * delta_q
            qd_current = v * delta_q
            qdd_current = a * delta_q
            
            # 记录当前速度，供可能的暂停使用
            self.current_velocity = qd_current.copy()
            self.current_acceleration = qdd_current.copy()

            # 执行伺服
            result = self.servoJ(q_current, dt)
            if result != 0:
                print("[MoveJ] servoJ 失败，运动终止。")
                self.state = "IDLE"
                break
            if self._should_plot:  # 使用实例属性判断
                self._time_traj.append(current_time)
                self._q_traj.append(q_current.copy())
                self._qd_traj.append(qd_current.copy())
                self._qdd_traj.append(qdd_current.copy())
            # 记录轨迹（用于可视化）
            # if traj_rviz:
            #     time_traj.append(current_time)
            #     q_traj.append(q_current.copy())
            #     qd_traj.append(qd_current.copy())
            #     qdd_traj.append(qdd_current.copy())
                
            current_time += dt

        if self.state == "MOVING":
            print("[MoveJ] 运动完成。")
            self.state = "IDLE"
        if self._should_plot and self._time_traj:
            self.plot_trajectory(self._time_traj, self._q_traj, self._qd_traj, self._qdd_traj)
        # if traj_rviz and time_traj:
        #     self.plot_trajectory(time_traj, q_traj, qd_traj, qdd_traj)

    def MoveL(self, tforms: list, dt: float = DELTA_T, max_retries: int = 10):
        """笛卡尔直线运动（此版本暂不支持中断）。"""
        q_start = self.q.copy()
        tforms_to_show, t_vec, q_vec, success = self.cartesian_planning(
            q_start=q_start,
            tforms=tforms,
            dt=dt,
            max_retries=max_retries
        )
        if not success:
            print("[Robot] MoveL 失败：笛卡尔路径规划不成功")
            return
        for idx in range(1, q_vec.shape[1]):
            self.servoJ(q_vec[:, idx], dt)

    def plot_trajectory(self, time_series, q_traj, qd_traj, qdd_traj):
        """绘制轨迹曲线。"""
        q_traj = np.array(q_traj)
        qd_traj = np.array(qd_traj)
        qdd_traj = np.array(qdd_traj)
        num_joints = q_traj.shape[1]

        plt.figure(figsize=(12, 8))
        for j in range(num_joints):
            plt.subplot(3, 1, 1)
            plt.plot(time_series, q_traj[:, j], label=f'joint {j+1}')
            plt.ylabel("pose (rad)")
            plt.title("joint position")

            plt.subplot(3, 1, 2)
            plt.plot(time_series, qd_traj[:, j], label=f'joint {j+1}')
            plt.ylabel("vel (rad/s)")
            plt.title("joint velocity")

            plt.subplot(3, 1, 3)
            plt.plot(time_series, qdd_traj[:, j], label=f'joint {j+1}')
            plt.ylabel("acc (rad/s²)")
            plt.title("joint acceleration")
            plt.xlabel("t (s)")

        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()


class DianaRobot(Robot):
    def __init__(self, target_frame, visualizer: bool = True):
        pinocchio_model_dir = Path(__file__).parent.parent.parent / "assets" / "urdf"
        model_path = pinocchio_model_dir
        print(f"模型路径: {model_path}")
        
        urdf_model_path = (
            pinocchio_model_dir 
            / "diana7_description" 
            / "urdf" 
            / "diana_v2.urdf"
        ).resolve()
        urdf_path = urdf_model_path.as_posix()
        
        if not urdf_model_path.exists():
            raise FileNotFoundError(f"未找到URDF文件: {urdf_path}")
            
        mesh_dir = model_path.resolve().as_posix()
        super().__init__(urdf_path, mesh_dir, vizualizer=visualizer)

        # 设置Diana7的实际关节约束
        dblMinPos = np.array([-3.124139, -1.570796, -3.124139, 0.000000, -3.124139, -3.124139, -3.124139])
        dblMaxPos = np.array([3.124139, 1.570796, 3.124139, 3.054326, 3.124139, 3.124139, 3.124139])
        dblMaxVel = np.array([2.967060, 2.617994, 2.617994, 2.617994, 3.141593, 3.141593, 3.839724])
        dblMaxAcc = np.array([10.780899, 8.733977, 8.931373, 8.794889, 14.885564, 14.762169, 14.803359])

        self.model.lowerPositionLimit = dblMinPos
        self.model.upperPositionLimit = dblMaxPos
        self.q_vel_limits = dblMaxVel
        self.q_acc_limits = dblMaxAcc
        self.q_limits = np.array([dblMinPos, dblMaxPos]).T
        self.model.velocityLimit = dblMaxVel
        self.model.accelerationLimit = dblMaxAcc


# 键盘监听线程函数
def keyboard_listener(robot):
    """
    独立线程：监听键盘事件，按下空格键触发暂停。
    """
    def on_press(key):
        try:
            if key == keyboard.Key.space:
                if robot.state == "MOVING" and not robot.pause_requested:
                    robot.pause_requested = True
                    print("\n[键盘监听] 空格键按下，暂停请求已发送。")
            elif hasattr(key, 'char') and key.char == 'r':  # 按下'r'键重启
                if robot.state == "PAUSED":
                    print("\n[键盘监听] 'r'键按下，重启运动。")
                    robot.resumeMoveJ(traj_rviz=True)
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


# 主程序
if __name__ == "__main__":
    # 初始化机器人
    target_frame = "link_7"
    robot = DianaRobot(target_frame, visualizer=True)
    
    # 启动键盘监听线程
    listener_thread = threading.Thread(target=keyboard_listener, args=(robot,), daemon=True)
    listener_thread.start()
    print("键盘监听已启动。按下空格键可暂停运动。")
    print("按下空格键可暂停运动。")
    print("暂停后，按下'r'键可重启运动到目标位置。")
    time.sleep(1)  # 等待线程启动

    # 定义起始关节角
    q_start = np.array([0.0, 0.564, 0, 1.84, 0.089, -0.504, 0])
    
    # 运动到起始点
    print("正在运动到起始点...")
    robot.MoveJ(q_start, v_max=0.8, a_max=8.0, dt=DELTA_T)
    print(f"已到达起始点: {np.degrees(q_start)} 度")
    time.sleep(1)
    
    # 定义目标点（一个明显的运动）
    q_target = np.array([1.0, 0.8, 0.5, 2.0, 0.5, -0.8, 0.3])
    print(f"目标点: {np.degrees(q_target)} 度")
    
    # 执行可中断的MoveJ运动
    print("开始执行MoveJ运动。尝试在运动过程中按下空格键...")
    print("暂停后，按下'r'键可重启运动到目标位置。")
    robot.MoveJ(q_target, v_max=0.5, a_max=6.0, dt=DELTA_T, traj_rviz=True)
    
    # 检查最终状态
    print(f"最终状态: {robot.state}")
    print(f"最终关节位置: {np.degrees(robot.q)} 度")
    
    # 保持程序运行
    print("程序运行结束。按Ctrl+C退出。")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n程序退出。")