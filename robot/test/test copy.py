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

class Robot:
    def __init__(
        self, 
        urdf_path: str, 

        end_effector_name: str,
        mesh_dir: str  ,
        collision_model=None,
        ik_options: DifferentialIkOptions = None,
        planner_options: CartesianPlannerOptions = None
    ):
        """
        Initialize the robot with the given URDF path and mesh directory.
        """
        self.model, self.collision_model, self.visual_model = self.load_robot(urdf_path, mesh_dir)
        self.data = self.model.createData()
        self.collision_data = self.collision_model.createData()

        # Initialize visualizer
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model, data=self.data)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()
    
    def load_robot(self, urdf_path: str, mesh_dir: str):
        """
        Load the robot model and geometries from a URDF file.
        """
        # Load the model
        model = pinocchio.buildModelsFromUrdf(urdf_path,mesh_dir)
        # Load collision and visual geometries
        collision_model = pinocchio.buildGeomFromUrdf(model, urdf_path, [mesh_dir], pinocchio.GeometryType.COLLISION)
        visual_model = pinocchio.buildGeomFromUrdf(model, urdf_path, [mesh_dir], pinocchio.GeometryType.VISUAL)
        return model, collision_model, visual_model
    
    def get_cartesian_pose(self, target_frame: str, q: np.ndarray):
        """
        Get the Cartesian pose of the robot's end-effector for the given joint configuration.
        """
        return extract_cartesian_pose(self.model, target_frame, q, data=self.data)
    
    def inverse_kinematics(self, target_frame: str, target_position: np.ndarray, q_start: np.ndarray, max_retries: int = 5):
        """
        Solve the inverse kinematics problem using Differential IK.
        """
        ik = DifferentialIk(self.model, data=self.data, collision_model=self.collision_model, 
                            options=DifferentialIkOptions(max_retries=max_retries))
        q_sol = ik.solve(target_frame, target_position, q_start)
        return q_sol
    
    def cartesian_planning(self, target_frame: str, q_start: np.ndarray, tforms: list, dt: float = 0.05):
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
        planner = CartesianPlanner(self.model, target_frame, tforms, ik=None, options=options)
        success, t_vec, q_vec = planner.generate(q_start, dt)
        return t_vec, q_vec

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

    def visualize_frames(self, target_frame: str, tforms: list):
        """
        Visualize the frames along the Cartesian path.
        """
        self.viz.display(np.zeros(self.model.nq))  # Display initial configuration
        visualize_frames(self.viz, "cartesian_plan", tforms, line_length=0.05, line_width=1)
        time.sleep(0.5)

# Example usage
from pathlib import Path

# Correct paths
pinocchio_model_dir = Path(__file__).parent.parent.parent / "assets" / "urdf"
urdf_model_path = pinocchio_model_dir / "diana7_description" / "urdf" / "diana_v2.urdf"
urdf_path = urdf_model_path.resolve().as_posix()
# Correct mesh directory to the parent of the package
mesh_dir = (pinocchio_model_dir).resolve().as_posix()

if not Path(urdf_path).exists():
    raise FileNotFoundError(f"URDF file not found at: {urdf_path}")

robot = Robot(urdf_path, mesh_dir)

# Define the Cartesian path from a start joint configuration
target_frame = "link_7"
q_start = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
init = robot.get_cartesian_pose(target_frame, q_start)
rot = Rotation.from_euler("z", 60, degrees=True).as_matrix()
rot_neg = Rotation.from_euler("z", -60, degrees=True).as_matrix()
tforms = [
    init,
    init * pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.2])),
    init * pinocchio.SE3(rot, np.array([0.0, 0.25, 0.2])),
    init * pinocchio.SE3(rot_neg, np.array([0.0, -0.25, 0.2])),
    init * pinocchio.SE3(np.eye(3), np.array([0.2, 0.0, 0.0])),
    init,
]

# Plan and display trajectory
t_vec, q_vec = robot.cartesian_planning(target_frame, q_start, tforms)
robot.display_trajectory(q_vec, t_vec)

# Visualize frames in Meshcat
robot.visualize_frames(target_frame, tforms)