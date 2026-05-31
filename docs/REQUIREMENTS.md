# RoboSimXCtrl Refactoring Requirements Document

**Version:** 1.0
**Date:** 2026-05-31
**Status:** Draft

---

## 1. Project Overview

### 1.1 Current Architecture

RoboSimXCtrl contains two parallel Robot class hierarchies for robot arm kinematics, dynamics, and simulation:

- **DH-based path** (`robot/robot.py`): An abstract `Robot` base class using `roboticstoolbox` (rtb) for DH-parameter-based forward/inverse kinematics and dynamics. Concrete subclasses include `UR5e` (`robot/ur5e.py`), `IIWA14` (`robot/iiwa14.py`), and `Diana` (`robot/diana.py`).

- **Pinocchio-based path** (`robot/robot_pinocchio.py`): A separate `Robot` class using Pinocchio for URDF-based kinematics, collision checking, and Meshcat visualization. This class has its own `DianaRobot` and `DianaMujocoEnv` subclasses defined inline in test scripts.

### 1.2 Current Problems

**P1: Massive code duplication across test scripts.** The files `robot/test/test_mujoco.py` (785 lines), `robot/test/dual_mujoco.py` (1088 lines), `robot/test/franka_dual.py` (679 lines), `robot/test/traj.py`, `robot/test/traj_resume.py`, and `robot/test/traj_resume_replaced.py` each contain full copies of the Pinocchio `Robot` class (~350 lines), `DianaRobot` (~40 lines), `DianaMujocoEnv` (~180 lines), `MoveJ` (~120 lines), `MoveL` (~30 lines), and `servoJ` (~40 lines). The total duplicated code across these scripts exceeds 3000 lines.

**P2: Two Robot systems are completely independent.** `robot/robot.py` (DH-based) and `robot/robot_pinocchio.py` (Pinocchio-based) share no common interface, base class, or utility code. They define different method names for the same operations (e.g., `ikine` vs `inverse_kinematics`, `move_cartesian` vs `MoveL`).

**P3: No MuJoCo abstraction.** Every test script that uses MuJoCo manually loads `MjModel`, creates `MjData`, initializes renderers, manages viewers, and implements the servoJ-to-MuJoCo sync loop. The existing `utils/mj.py` provides helper functions but they are bare functions, not a cohesive environment class.

**P4: Meshcat visualization only available in Pinocchio path.** DH-based robots (`UR5e`, `IIWA14`, `Diana`) have no web-based visualization capability.

**P5: Known bugs in existing code** (detailed in Section 5).

### 1.3 Refactoring Goals

1. Extract a unified MuJoCo simulation environment class from scattered test scripts.
2. Create simulation-aware base classes for both DH-based and Pinocchio-based robots that optionally integrate MuJoCo and Meshcat.
3. Eliminate code duplication by having subclasses inherit simulation capabilities rather than reimplementing them.
4. Preserve full backward compatibility: existing `Robot` base classes and their public APIs remain unchanged.
5. Fix all identified bugs.
6. MuJoCo and Meshcat must be optional dependencies -- the system must work without them installed.

---

## 2. Functional Requirements

### FR-1: MuJoCoSim Environment Class

**Description:** A standalone MuJoCo physics simulation environment class that wraps the MuJoCo API (MjModel/MjData) and provides a clean interface for simulation control, joint manipulation, body pose queries, constraint-based grasping, viewer management, and rendering.

**File:** `robot/sim/mujoco_sim.py`

**Requirements:**

**FR-1.1: Initialization and Model Loading**
- The constructor SHALL accept an XML path string and an optional timestep parameter (default 0.002s).
- On construction, the class SHALL load the MuJoCo model via `mujoco.MjModel.from_xml_path()` and create `MjData`.
- The class SHALL store `self.model` (MjModel) and `self.data` (MjData) as public attributes.

**Acceptance Criteria:**
- `sim = MuJoCoSim("path/to/scene.xml")` succeeds and `sim.model` and `sim.data` are valid MuJoCo objects.
- Custom timestep is applied: `sim.model.opt.timestep == custom_value`.

**FR-1.2: Simulation Stepping**
- `step(ctrl=None)` SHALL advance the simulation by one timestep. If `ctrl` is provided, it SHALL set `self.data.ctrl` before stepping.
- `forward()` SHALL call `mujoco.mj_forward(self.model, self.data)`.
- `reset()` SHALL call `mujoco.mj_resetData(self.model, self.data)`.

**Acceptance Criteria:**
- After `step()`, `self.data.time` increases by `self.model.opt.timestep`.
- After `reset()`, `self.data.time` is 0.0 and joint positions return to their initial values.

**FR-1.3: Joint Control**
- `set_joint_q(name_or_id, q)` SHALL set joint positions in `self.data.qpos`. The `name_or_id` parameter SHALL accept either a string joint name or an integer joint ID.
- `get_joint_q(name_or_id)` SHALL return the current joint position as a numpy array.
- `set_ctrl(ctrl)` SHALL set `self.data.ctrl` to the provided array.
- These methods SHALL integrate the existing logic from `utils/mj.py` functions `set_joint_q`, `get_joint_q`, `get_joint_qpos_inds`, `get_joint_qpos_addr`, and `get_joint_dim`.

**Acceptance Criteria:**
- `sim.set_joint_q("joint_1", 0.5)` followed by `sim.get_joint_q("joint_1")` returns `0.5`.
- Both string names and integer IDs work: `sim.set_joint_q(0, 0.5)` and `sim.set_joint_q("joint_1", 0.5)` produce the same result.

**FR-1.4: Body Pose Manipulation**
- `get_body_pose(name)` SHALL return the body pose as a `spatialmath.SE3` object.
- `get_body_pose_xyz(name)` SHALL return the position as a 3-element numpy array.
- `set_body_pose(name, xpos)` SHALL set the body position.
- These methods SHALL integrate the existing logic from `utils/mj.py` functions `get_body_pose`, `get_body_pose_xyz`, `set_body_pose`, and `set_body_position`.

**Acceptance Criteria:**
- `pose = sim.get_body_pose("robot_base")` returns a valid SE3 with position matching MuJoCo's reported position.
- After `sim.set_body_pose("box", np.array([1, 0, 0]))`, `sim.get_body_pose_xyz("box")` returns `[1, 0, 0]`.

**FR-1.5: Grasp/Release via Equality Constraints**
- `attach(equality_name, free_joint_name, T, ...)` SHALL activate a MuJoCo equality constraint to attach an object to the gripper. This SHALL integrate the existing `attach()` function from `utils/mj.py`.
- `detach(equality_name)` SHALL deactivate the equality constraint by setting `data.eq_active[eq_id] = 0`.
- `set_free_joint_pose(joint_name, T)` SHALL set a free joint's position and orientation from an SE3 object.

**Acceptance Criteria:**
- After `sim.attach("grasp_eq", "box_free_joint", gripper_pose)`, `sim.data.eq_active[eq_id]` is 1.
- After `sim.detach("grasp_eq")`, `sim.data.eq_active[eq_id]` is 0.

**FR-1.6: Viewer Management**
- `launch_viewer()` SHALL create a passive MuJoCo viewer via `mujoco.viewer.launch_passive()` and store the handle.
- `close_viewer()` SHALL close the viewer if it exists.
- `sync_viewer()` SHALL call `self.viewer.sync()` if a viewer is open.

**Acceptance Criteria:**
- After `launch_viewer()`, `sim.viewer` is not None.
- After `close_viewer()`, `sim.viewer` is None and no viewer window is displayed.

**FR-1.7: Offscreen Rendering**
- `render_rgb(camera, width, height)` SHALL return an RGB image as a numpy array of shape (height, width, 3).
- `render_depth(camera, width, height)` SHALL return a depth image as a numpy array of shape (height, width).
- `get_point_cloud(camera, width, height)` SHALL return a point cloud as a numpy array of shape (N, 6) where columns 0-2 are XYZ and columns 3-5 are RGB.
- A `Renderer` object SHALL be lazily created on first render call and cached.

**Acceptance Criteria:**
- `rgb = sim.render_rgb(0, 256, 256)` returns shape `(256, 256, 3)` with dtype `uint8`.
- `depth = sim.render_depth(0, 256, 256)` returns shape `(256, 256)` with dtype `float32`.
- `pcd = sim.get_point_cloud(0, 256, 256)` returns shape `(N, 6)` where N <= 256*256.

**FR-1.8: Scene Object Management (Optional/Stretch)**
- `add_box(name, pos, size, rgba)` MAY add a box geom to the scene.
- `add_sphere(name, pos, radius, rgba)` MAY add a sphere geom to the scene.

**Acceptance Criteria:**
- After `add_box("test_box", [0.5, 0, 0.3], [0.05, 0.05, 0.05], [1, 0, 0, 1])`, calling `sim.get_body_pose("test_box")` returns a pose at `[0.5, 0, 0.3]`.

---

### FR-2: DHRobotSim Base Class

**Description:** A simulation-aware base class for DH-parameter-based robots that extends the existing `Robot` class from `robot/robot.py` with optional MuJoCo simulation and Meshcat web visualization. All existing kinematics, dynamics, and motion planning methods (`MoveJ`, `MoveL`, `servoJ`, `fkine`, `ikine`, `get_inertia`, `inv_dynamics`, etc.) are inherited without reimplementation.

**File:** `robot/sim/dh_robot_sim.py`

**Requirements:**

**FR-2.1: Inheritance**
- `DHRobotSim` SHALL inherit from `robot.robot.Robot`.
- All existing methods from `Robot` (including `fkine`, `ikine`, `get_inertia`, `get_coriolis`, `get_gravity`, `inv_dynamics`, `set_tool`, `set_base`, `get_joint`, `set_joint`, `move_cartesian`, `get_identification_matrix`, `get_adaptive_identification_matrix`, `inv_dynamics_adaptive`) SHALL be available without reimplementation.

**Acceptance Criteria:**
- `isinstance(DHRobotSim(), Robot)` returns `True`.
- `DHRobotSim().fkine([0]*6)` works (if a concrete subclass provides DH parameters).

**FR-2.2: Optional MuJoCo Integration**
- The constructor SHALL accept an optional `mujoco_xml_path` parameter.
- If provided, `self.mj_sim` SHALL be a `MuJoCoSim` instance loaded from that path.
- If not provided, `self.mj_sim` SHALL be `None`.
- All MuJoCo operations SHALL gracefully no-op when `self.mj_sim is None`.

**Acceptance Criteria:**
- `DHRobotSim(mujoco_xml_path=None).mj_sim is None`.
- `DHRobotSim(mujoco_xml_path="path/to/scene.xml").mj_sim` is a `MuJoCoSim` instance.
- Calling `sim_robot.mujoco_step()` when `mj_sim is None` does not raise an exception.

**FR-2.3: Optional Meshcat Integration**
- The constructor SHALL accept an optional `meshcat_port` parameter (default `None`).
- If provided (non-None), Meshcat visualization SHALL be initialized via `pinocchio.visualize.MeshcatVisualizer` or a lightweight standalone Meshcat connection.
- `self.meshcat` SHALL store the Meshcat visualizer instance or `None`.
- If Pinocchio is not installed, Meshcat initialization SHALL be skipped gracefully with a warning.

**Acceptance Criteria:**
- `DHRobotSim(meshcat_port=None).meshcat is None`.
- When Meshcat is initialized, `robot.meshcat_display(q)` updates the browser visualization.

**FR-2.4: MuJoCo Convenience Methods**
- `mujoco_step()` SHALL delegate to `self.mj_sim.step()`.
- `mujoco_set_joint(q)` SHALL set all robot joint positions in MuJoCo via `self.mj_sim.set_joint_q()` for each joint name.
- `mujoco_get_joint()` SHALL return the current MuJoCo joint positions as a numpy array.
- `mujoco_get_body_pose(name)` SHALL return the MuJoCo body pose as SE3.

**Acceptance Criteria:**
- After `robot.mujoco_set_joint([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])`, `robot.mujoco_get_joint()` returns `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]`.

**FR-2.5: Meshcat Visualization Methods**
- `meshcat_display(q=None)` SHALL display the robot at joint configuration `q` (or `self.q0` if None) in Meshcat.
- `meshcat_display_trajectory(q_list)` SHALL animate a sequence of joint configurations in Meshcat.
- `meshcat_display_frame(T, name)` SHALL display a coordinate frame at SE3 pose `T` with label `name`.

**Acceptance Criteria:**
- After `meshcat_display([0, 0, 0, 0, 0, 0])`, the Meshcat browser shows the robot at its home position.
- `meshcat_display_trajectory([[0]*6, [0.1]*6, [0.2]*6])` animates through three configurations.

**FR-2.6: servoJ Override**
- `DHRobotSim.servoJ(q_target, dt)` SHALL:
  1. Call `super().servoJ(q_target, dt)` to perform existing position/velocity limit checks and update `self.q0`.
  2. If `self.mj_sim is not None`, sync the joint positions to MuJoCo and step the simulation.
  3. If `self.meshcat is not None`, update the Meshcat visualization.
- The return value SHALL be 0 on success, -1 on failure (matching existing convention).

**Acceptance Criteria:**
- With MuJoCo enabled: after `servoJ([0.1]*6)`, `self.mj_sim.get_joint_q("joint_1")` is approximately `0.1`.
- With Meshcat enabled: the browser visualization updates after each `servoJ` call.
- With both disabled: `servoJ` behaves identically to the parent `Robot.servoJ`.

**FR-2.7: MoveJ/MoveL Inheritance**
- `MoveJ` and `MoveL` SHALL NOT be reimplemented. They are inherited from `Robot` and work through the overridden `servoJ`.

**Acceptance Criteria:**
- `DHRobotSim().MoveJ([0.1]*6, v_max=1.0, a_max=2.0)` executes without error (with a concrete subclass providing DH parameters).
- During `MoveJ`, each intermediate `servoJ` call syncs to MuJoCo and Meshcat.

---

### FR-3: PinocchioRobotSim Base Class

**Description:** A simulation-aware base class for URDF/Pinocchio-based robots that extends the existing `Robot` class from `robot/robot_pinocchio.py` with optional MuJoCo simulation. The existing Meshcat integration from Pinocchio is preserved.

**File:** `robot/sim/pinocchio_robot_sim.py`

**Requirements:**

**FR-3.1: Inheritance**
- `PinocchioRobotSim` SHALL inherit from the `Robot` class defined in `robot/robot_pinocchio.py`.
- All existing methods (`MoveJ`, `MoveL`, `servoJ`, `servol`, `inverse_kinematics`, `cartesian_planning`, `get_cartesian_pose`, `check_and_highlight_collisions`, `display_trajectory`, `visualize_frames`) SHALL be available without reimplementation.

**Acceptance Criteria:**
- `PinocchioRobotSim` is a subclass of the Pinocchio `Robot`.
- All inherited methods work without modification.

**FR-3.2: Optional MuJoCo Integration**
- The constructor SHALL accept an optional `mujoco_xml_path` parameter.
- If provided, `self.mj_sim` SHALL be a `MuJoCoSim` instance.
- If not provided, `self.mj_sim` SHALL be `None`.

**Acceptance Criteria:**
- `PinocchioRobotSim(urdf_path, mesh_dir, mujoco_xml_path=None).mj_sim is None`.
- `PinocchioRobotSim(urdf_path, mesh_dir, mujoco_xml_path="path/to/scene.xml").mj_sim` is a `MuJoCoSim` instance.

**FR-3.3: MuJoCo Convenience Methods**
- Same interface as FR-2.4: `mujoco_step()`, `mujoco_set_joint(q)`, `mujoco_get_joint()`, `mujoco_get_body_pose(name)`.

**Acceptance Criteria:**
- Same as FR-2.4 acceptance criteria.

**FR-3.4: servoJ Override**
- `PinocchioRobotSim.servoJ(q_target, dt)` SHALL:
  1. Call `super().servoJ(q_target, dt)` to perform existing collision checking, position/velocity limit checks, and Meshcat visualization.
  2. If `self.mj_sim is not None`, sync joint positions to MuJoCo and step the simulation.

**Acceptance Criteria:**
- With MuJoCo enabled: after `servoJ`, MuJoCo joint positions match the Pinocchio joint positions.
- Without MuJoCo: behavior is identical to the parent class.

**FR-3.5: MoveJ/MoveL Inheritance**
- `MoveJ` and `MoveL` SHALL NOT be reimplemented.

**Acceptance Criteria:**
- Same as FR-2.7.

---

### FR-4: TaskEnv for Manipulation Tasks

**Description:** A task-level environment that combines a robot simulator with a MuJoCo simulation scene for object manipulation tasks.

**File:** `robot/sim/task_env.py`

**Requirements:**

**FR-4.1: Initialization**
- The constructor SHALL accept a `MuJoCoSim` instance and a robot simulator instance (`DHRobotSim` or `PinocchioRobotSim`).
- `self.sim` SHALL reference the MuJoCoSim instance.
- `self.robot` SHALL reference the robot simulator.

**Acceptance Criteria:**
- `task = TaskEnv(sim, robot)` succeeds when both arguments are valid.
- `task.sim` and `task.robot` reference the correct objects.

**FR-4.2: Object Spawn/Remove**
- `spawn_object(name, pos, shape, size, rgba=None)` SHALL add an object to the MuJoCo scene.
- `remove_object(name)` SHALL remove an object from the scene.
- Supported shapes: `"box"`, `"sphere"`, `"cylinder"`.

**Acceptance Criteria:**
- After `task.spawn_object("box1", [0.5, 0, 0.3], "box", [0.05]*3)`, `task.sim.get_body_pose("box1")` returns a pose near `[0.5, 0, 0.3]`.
- After `task.remove_object("box1")`, subsequent pose queries raise an error or return None.

**FR-4.3: Grasp/Release Operations**
- `grasp(object_name)` SHALL create a MuJoCo equality constraint attaching the object to the robot's gripper body.
- `release(object_name)` SHALL deactivate the equality constraint.
- Grasp SHALL use the `attach`/`detach` methods of `MuJoCoSim`.

**Acceptance Criteria:**
- After `task.grasp("box1")`, moving the robot end-effector causes the box to follow.
- After `task.release("box1")`, the box remains at its current position and the robot moves independently.

**FR-4.4: Observation Generation**
- `get_observation()` SHALL return a dictionary containing:
  - `"joint_pos"`: current joint positions (numpy array)
  - `"joint_vel"`: current joint velocities (numpy array, zeros if unavailable)
  - `"ee_pose"`: end-effector pose as SE3
  - `"rgb"`: RGB image from MuJoCo renderer (optional, only if renderer is configured)
  - `"depth"`: depth image from MuJoCo renderer (optional)
  - `"point_cloud"`: point cloud from depth + RGB (optional)

**Acceptance Criteria:**
- `obs = task.get_observation()` returns a dict with keys `"joint_pos"` and `"ee_pose"` at minimum.
- `obs["joint_pos"]` has shape `(n_dof,)`.
- When rendering is configured, `obs["rgb"]` has shape `(H, W, 3)`.

**FR-4.5: Scene Reset**
- `reset_scene()` SHALL reset the MuJoCo simulation, re-spawn all registered objects to their initial positions, and reset the robot to its initial joint configuration.

**Acceptance Criteria:**
- After manipulation and `reset_scene()`, all objects return to their spawn positions and the robot returns to its initial configuration.

---

### FR-5: DualArmSim

**Description:** A simulation wrapper for dual-arm setups where two robots operate in a single MuJoCo scene.

**File:** `robot/sim/dual_arm_sim.py`

**Requirements:**

**FR-5.1: Initialization**
- The constructor SHALL accept two robot simulator instances and a shared `MuJoCoSim` instance.
- `self.left_arm` and `self.right_arm` SHALL reference the two robot simulators.
- `self.sim` SHALL reference the shared MuJoCo scene.

**Acceptance Criteria:**
- `dual = DualArmSim(left_robot, right_robot, sim)` succeeds.
- `dual.left_arm` and `dual.right_arm` are valid robot instances.

**FR-5.2: Synchronized Stepping**
- `step(q_left, q_right)` SHALL set joint positions for both arms in the MuJoCo scene and advance one simulation step.
- The step SHALL be atomic: both arms update before the simulation steps.

**Acceptance Criteria:**
- After `dual.step(q_left, q_right)`, `dual.left_arm.mujoco_get_joint()` returns `q_left` and `dual.right_arm.mujoco_get_joint()` returns `q_right`.

**FR-5.3: Dual MoveJ**
- `MoveJ_dual_arm(q_left, q_right, v_max, a_max, dt)` SHALL execute `MoveJ` on both arms concurrently (using threads), similar to the existing `DualArmMujocoEnv.MoveJ_dual_arm` in `dual_mujoco.py`.

**Acceptance Criteria:**
- After `MoveJ_dual_arm(q_left_target, q_right_target)`, both arms reach their target configurations.
- Both arms move concurrently, not sequentially.

**FR-5.4: Individual Arm Control**
- `MoveJ_left_arm(q_left, v_max, a_max)` SHALL move only the left arm.
- `MoveJ_right_arm(q_right, v_max, a_max)` SHALL move only the right arm.

**Acceptance Criteria:**
- `MoveJ_left_arm` does not affect the right arm's configuration.
- `MoveJ_right_arm` does not affect the left arm's configuration.

---

## 3. Non-Functional Requirements

### NFR-1: Backward Compatibility

- The existing `Robot` class in `robot/robot.py` SHALL NOT be modified in its public API. Only bug fixes (Section 5) are permitted.
- The existing `Robot` class in `robot/robot_pinocchio.py` SHALL NOT be modified in its public API.
- Existing test scripts in `robot/test/` SHALL continue to work without modification during the transition (they will be refactored in a later phase).
- The existing `robot/__init__.py` exports (`Robot`, `UR5e`, `IIWA14`, `Diana`) SHALL remain unchanged.

**Acceptance Criteria:**
- `from robot import Robot, UR5e, IIWA14, Diana` succeeds after refactoring.
- All existing test scripts run without error.

### NFR-2: Optional Dependencies

- MuJoCo (`mujoco`) SHALL be an optional dependency. If not installed, all MuJoCo-related features SHALL be unavailable but the system SHALL NOT crash.
- Meshcat (`meshcat`, `pinocchio`) SHALL be an optional dependency. If not installed, Meshcat visualization SHALL be unavailable but the system SHALL NOT crash.
- Import guards SHALL use `try/except ImportError` blocks.

**Acceptance Criteria:**
- `import robot` succeeds even when `mujoco` is not installed.
- `DHRobotSim(mujoco_xml_path=None)` works without MuJoCo installed.
- A clear warning message is printed when an optional feature is requested but the dependency is missing.

### NFR-3: Separation of Concerns

- Simulation logic (MuJoCo stepping, viewer management, rendering) SHALL be in the `robot/sim/` package.
- Kinematics and dynamics logic SHALL remain in `robot/robot.py` and `robot/robot_pinocchio.py`.
- Utility functions (`utils/mj.py`) SHALL be preserved for backward compatibility but their logic SHALL be migrated into `MuJoCoSim` methods.

**Acceptance Criteria:**
- `robot/sim/mujoco_sim.py` does not import from `robot/robot.py` or `robot/robot_pinocchio.py`.
- `robot/robot.py` does not import from `robot/sim/`.
- Circular dependencies do not exist.

### NFR-4: Code Quality

- All new files SHALL include module-level docstrings.
- All public methods SHALL have docstrings with parameter types and return types.
- Type hints SHALL be used for all method signatures.
- All new code SHALL pass `python -m py_compile` without errors.

**Acceptance Criteria:**
- `python -m py_compile robot/sim/mujoco_sim.py` succeeds.
- `python -m py_compile robot/sim/dh_robot_sim.py` succeeds.
- `python -m py_compile robot/sim/pinocchio_robot_sim.py` succeeds.

---

## 4. Interface Specifications

### 4.1 MuJoCoSim

```python
class MuJoCoSim:
    """MuJoCo physics simulation environment."""

    def __init__(self, xml_path: str, timestep: float = 0.002):
        """Load MuJoCo model from XML and create simulation data."""
        ...

    # --- Simulation Control ---
    def step(self, ctrl: np.ndarray = None) -> None: ...
    def forward(self) -> None: ...
    def reset(self) -> None: ...

    # --- Joint Control ---
    def set_joint_q(self, name_or_id: Union[str, int], q: Union[np.ndarray, float], unit: str = "rad") -> None: ...
    def get_joint_q(self, name_or_id: Union[str, int]) -> np.ndarray: ...
    def set_ctrl(self, ctrl: np.ndarray) -> None: ...

    # --- Body Pose ---
    def get_body_pose(self, name: Union[str, int]) -> SE3: ...
    def get_body_pose_xyz(self, name: Union[str, int]) -> np.ndarray: ...
    def set_body_pose(self, name: Union[str, int], xpos: np.ndarray) -> None: ...

    # --- Constraints / Grasping ---
    def set_free_joint_pose(self, joint_name: Union[str, int], T: SE3) -> None: ...
    def attach(self, equality_name: str, free_joint_name: str, T: SE3,
               eq_data: np.ndarray = ..., eq_solimp: np.ndarray = ...,
               eq_solref: np.ndarray = ...) -> None: ...
    def detach(self, equality_name: str) -> None: ...

    # --- Viewer ---
    def launch_viewer(self) -> None: ...
    def close_viewer(self) -> None: ...
    def sync_viewer(self) -> None: ...

    # --- Rendering ---
    def render_rgb(self, camera: Union[str, int], width: int, height: int) -> np.ndarray: ...
    def render_depth(self, camera: Union[str, int], width: int, height: int) -> np.ndarray: ...
    def get_point_cloud(self, camera: Union[str, int], width: int, height: int,
                        depth_limit: float = 3.0) -> np.ndarray: ...
```

### 4.2 DHRobotSim

```python
class DHRobotSim(Robot):
    """DH-based robot with optional MuJoCo simulation and Meshcat visualization."""

    def __init__(self, mujoco_xml_path: str = None, meshcat_port: int = None):
        """Initialize with optional MuJoCo and Meshcat backends."""
        ...

    # --- MuJoCo Interface (delegate to self.mj_sim) ---
    def mujoco_step(self) -> None: ...
    def mujoco_set_joint(self, q: np.ndarray) -> None: ...
    def mujoco_get_joint(self) -> np.ndarray: ...
    def mujoco_get_body_pose(self, name: str) -> SE3: ...

    # --- Meshcat Interface ---
    def meshcat_display(self, q: np.ndarray = None) -> None: ...
    def meshcat_display_trajectory(self, q_list: List[np.ndarray]) -> None: ...
    def meshcat_display_frame(self, T: SE3, name: str) -> None: ...

    # --- Override ---
    def servoJ(self, q_target: np.ndarray, dt: float = 0.001) -> int:
        """Servo joint: parent check + MuJoCo sync + Meshcat display."""
        ...
```

### 4.3 PinocchioRobotSim

```python
class PinocchioRobotSim(Robot):
    """Pinocchio-based robot with optional MuJoCo simulation."""

    def __init__(self, urdf_path: str, mesh_dir: str,
                 mujoco_xml_path: str = None, visualizer: bool = False,
                 target_frame: str = "link_7"):
        """Initialize Pinocchio model + optional MuJoCo."""
        ...

    # --- MuJoCo Interface (delegate to self.mj_sim) ---
    def mujoco_step(self) -> None: ...
    def mujoco_set_joint(self, q: np.ndarray) -> None: ...
    def mujoco_get_joint(self) -> np.ndarray: ...
    def mujoco_get_body_pose(self, name: str) -> SE3: ...

    # --- Override ---
    def servoJ(self, q_target: np.ndarray, delta_t: float = 0.001) -> int:
        """Servo joint: parent check + collision + MuJoCo sync."""
        ...
```

### 4.4 TaskEnv

```python
class TaskEnv:
    """Manipulation task environment combining robot and MuJoCo simulation."""

    def __init__(self, sim: MuJoCoSim, robot: Union[DHRobotSim, PinocchioRobotSim]):
        ...

    # --- Scene Management ---
    def spawn_object(self, name: str, pos: np.ndarray, shape: str,
                     size: np.ndarray, rgba: np.ndarray = None) -> None: ...
    def remove_object(self, name: str) -> None: ...
    def reset_scene(self) -> None: ...

    # --- Grasping ---
    def grasp(self, object_name: str) -> None: ...
    def release(self, object_name: str) -> None: ...

    # --- Observation ---
    def get_observation(self) -> dict: ...
```

### 4.5 DualArmSim

```python
class DualArmSim:
    """Dual-arm simulation managing two robots in one MuJoCo scene."""

    def __init__(self, left_arm: Union[DHRobotSim, PinocchioRobotSim],
                 right_arm: Union[DHRobotSim, PinocchioRobotSim],
                 sim: MuJoCoSim):
        ...

    # --- Synchronized Control ---
    def step(self, q_left: np.ndarray, q_right: np.ndarray) -> None: ...

    # --- Motion ---
    def MoveJ_dual_arm(self, q_left: np.ndarray, q_right: np.ndarray,
                       v_max: float = 1.0, a_max: float = 1.0,
                       dt: float = 0.001) -> None: ...
    def MoveJ_left_arm(self, q_left: np.ndarray, v_max: float = 1.0,
                       a_max: float = 1.0) -> None: ...
    def MoveJ_right_arm(self, q_right: np.ndarray, v_max: float = 1.0,
                        a_max: float = 1.0) -> None: ...

    # --- Cleanup ---
    def close(self) -> None: ...
```

---

## 5. Bug Fixes Required

### BUG-1: IIWA14 -- `self.robot` assignment inside for loop

**File:** `robot/iiwa14.py`, line 122
**Problem:** The line `self.robot = rtb.DHRobot(links)` is indented inside the `for i in range(self._dof)` loop body. This means `self.robot` is reassigned on every iteration, and only the partial link list from the last iteration is used. The final robot will have links with incorrect kinematic parameters because the loop builds `links` incrementally but reassigns `self.robot` each time.

**Current code (line 117-123):**
```python
links = []
for i in range(self._dof):
    links.append(
        rtb.DHLink(d=self.d_array[i], alpha=self.alpha_array[i], a=self.a_array[i], offset=self.theta_array[i],
                   mdh=True, m=ms[i], r=rs[i], I=(Rs[i] @ Is[i] @ Rs[i].T)))
    self.robot = rtb.DHRobot(links)  # BUG: inside loop
```

**Fix:** Dedent `self.robot = rtb.DHRobot(links)` by one level so it executes after the loop completes.

**Acceptance Criteria:**
- After the fix, `IIWA14().robot.n == 7` (7 links, not 1).
- Forward kinematics at `q = [0]*7` returns the correct home position.

### BUG-2: Robot.__setstate__ -- hardcoded `range(6)`

**File:** `robot/robot.py`, line 278
**Problem:** The `__setstate__` method uses `for i in range(6)` to rebuild the `rtb.DHRobot` from serialized data. This is hardcoded to 6 joints regardless of the actual robot's DOF. A 7-DOF robot (IIWA14, Diana) would lose its 7th joint when deserialized.

**Current code (line 277-282):**
```python
links = []
for i in range(6):  # BUG: hardcoded 6
    links.append(
        rtb.DHLink(d=self.d_array[i], alpha=self.alpha_array[i], a=self.a_array[i], offset=self.theta_array[i],
                   mdh=True))
self.robot = rtb.DHRobot(links)
```

**Fix:** Replace `range(6)` with `range(self._dof)`.

**Acceptance Criteria:**
- Serialize and deserialize a 7-DOF robot: `pickle.loads(pickle.dumps(IIWA14())).dof == 7`.
- The deserialized robot's `fkine([0]*7)` matches the original.

### BUG-3: Robot.disable_tool -- type inconsistency

**File:** `robot/robot.py`, line 241
**Problem:** `disable_tool` sets `self._tool = np.zeros(3)` (a numpy array), but `set_tool` sets `self._tool = SE3.Trans(tool)` (an SE3 object). The `fkine` and `ikine` methods use `self._tool.inv()`, which will fail with `AttributeError: 'numpy.ndarray' object has no attribute 'inv'` if `disable_tool` was called.

**Current code (line 240-242):**
```python
def disable_tool(self):
    self._tool = np.zeros(3)  # BUG: should be SE3()
    self.robot.tool = self._tool
```

**Fix:** Change `self._tool = np.zeros(3)` to `self._tool = SE3()` and `self.robot.tool = self._tool`.

**Acceptance Criteria:**
- `robot.set_tool(np.array([0, 0, 0.1])); robot.disable_tool(); robot.fkine([0]*6)` does not raise an `AttributeError`.
- After `disable_tool()`, `robot._tool == SE3()`.

### BUG-4: Diana.move_cartesian_with_avoidance -- calls undefined method

**File:** `robot/diana.py`, line 93-96
**Problem:** `move_cartesian_with_avoidance` calls `self.ikine_with_avoidance(T)`, but no such method exists in the `Diana` class. This method only exists in `IIWA14` (line 474 of `iiwa14.py`). Calling it raises `AttributeError`.

**Current code (line 93-96):**
```python
def move_cartesian_with_avoidance(self, T: SE3):
    q = self.ikine_with_avoidance(T)  # BUG: method does not exist in Diana
    if q.size != 0:
        self.q0 = q[:]
```

**Fix options:**
1. **Remove the method** if it is not needed for Diana (recommended, since Diana uses TracIK which handles avoidance differently).
2. **Implement `ikine_with_avoidance`** for Diana if required.

**Acceptance Criteria:**
- `Diana().move_cartesian_with_avoidance(T)` either works correctly or is removed so it does not exist as a broken method.

### BUG-5: Diana -- hardcoded URDF path

**File:** `robot/diana.py`, line 18
**Problem:** The `Diana.__init__` method hardcodes the URDF path as `"/home/sun/Documents/GitHub/imitation_learning_idp3/tracikpy/data/diana_v2.urdf"`. This is an absolute path specific to one developer's machine and will fail on any other machine.

**Current code (line 17-20):**
```python
self.ik_solver = TracIKSolver(
    "/home/sun/Documents/GitHub/imitation_learning_idp3/tracikpy/data/diana_v2.urdf",
    "base",
    "link_7", solve_type="Distance", timeout=0.01, epsilon=1e-5)
```

**Fix:** Accept `urdf_path` as a constructor parameter with a default value relative to the project root, or compute it dynamically like the Pinocchio-based `DianaRobot` does:
```python
urdf_path = Path(__file__).parent.parent / "assets" / "urdf" / "diana7_description" / "urdf" / "diana_v2.urdf"
```

**Acceptance Criteria:**
- `Diana()` succeeds on any machine where the `assets/urdf/diana7_description/urdf/diana_v2.urdf` file exists.
- No hardcoded `/home/sun/...` paths remain in the codebase.

### BUG-6: test_mujoco.py DianaMujocoEnv._get_obs -- typo/undefined variable

**File:** `robot/test/test_mujoco.py`, line 526
**Problem:** The `_get_obs` method contains `agent_pos = self.robot.fkine(self.robot_q).t` but `self.robot` and `self.robot_q` are never defined in `DianaMujocoEnv`. Also line 526 has `'agent_pos': agent_posset_joint_positions` which is a clear copy-paste error (concatenation of `agent_pos` and `set_joint_positions`).

**Current code (line 523-528):**
```python
self.robot_T = self.robot.fkine(self.robot_q)      # BUG: self.robot undefined
agent_pos = self.robot.fkine(self.robot_q).t        # BUG: self.robot undefined
obs = {
    'agent_pos': agent_posset_joint_positions,       # BUG: typo
    'point_cloud': sampled_points
}
```

**Fix:** Replace `self.robot.fkine(self.robot_q)` with `self.fkine(self.q)` (the class itself inherits from Robot). Fix the typo on line 526.

**Acceptance Criteria:**
- `env._get_obs()` returns a valid dict without `AttributeError` or `NameError`.

---

## 6. File Structure

### 6.1 New Files to Create

| File | Description |
|------|-------------|
| `robot/sim/__init__.py` | Package init; exports `MuJoCoSim`, `DHRobotSim`, `PinocchioRobotSim`, `TaskEnv`, `DualArmSim` |
| `robot/sim/mujoco_sim.py` | MuJoCo simulation environment class (FR-1) |
| `robot/sim/dh_robot_sim.py` | DH-based robot simulation base class (FR-2) |
| `robot/sim/pinocchio_robot_sim.py` | Pinocchio-based robot simulation base class (FR-3) |
| `robot/sim/task_env.py` | Manipulation task environment (FR-4) |
| `robot/sim/dual_arm_sim.py` | Dual-arm simulation wrapper (FR-5) |
| `robot/test/test_sim_ur5e.py` | UR5e simulation test using DHRobotSim |
| `robot/test/test_sim_diana.py` | Diana MuJoCo simulation test using PinocchioRobotSim |
| `robot/test/test_task.py` | Manipulation task test using TaskEnv |
| `robot/test/test_dual_arm.py` | Dual-arm simulation test using DualArmSim |

### 6.2 Files to Modify (Bug Fixes Only)

| File | Changes |
|------|---------|
| `robot/iiwa14.py` | BUG-1: Dedent `self.robot = rtb.DHRobot(links)` outside loop |
| `robot/robot.py` | BUG-2: `range(6)` -> `range(self._dof)` in `__setstate__`; BUG-3: `disable_tool` type fix |
| `robot/diana.py` | BUG-4: Remove or fix `move_cartesian_with_avoidance`; BUG-5: Parameterize URDF path |
| `robot/__init__.py` | Add `robot.sim` package re-exports (optional, for convenience) |

### 6.3 Files to Keep Unchanged

| File | Reason |
|------|--------|
| `robot/robot.py` (API surface) | Backward compatibility; only bug fixes applied |
| `robot/robot_pinocchio.py` | No changes needed; serves as base class for PinocchioRobotSim |
| `robot/robot_config.py` | No changes needed |
| `robot/ur5e.py` | Will be adapted to use DHRobotSim in Phase 5, but not in initial refactoring |
| `utils/mj.py` | Preserved for backward compatibility; logic migrated to MuJoCoSim |
| `utils/rtb.py` | No changes needed |
| `utils/__init__.py` | No changes needed |
| `robot/test/test_mujoco.py` | Preserved during transition; BUG-6 fix is optional |
| `robot/test/dual_mujoco.py` | Preserved during transition |
| `robot/test/franka_dual.py` | Preserved during transition |
| `robot/test/traj.py` | Preserved during transition |
| `robot/test/traj_resume.py` | Preserved during transition |
| `robot/test/traj_resume_replaced.py` | Preserved during transition |

### 6.4 Final Directory Structure

```
RoboSimXCtrl/
├── docs/
│   └── REQUIREMENTS.md           # This document
├── robot/
│   ├── __init__.py               # Existing exports (unchanged)
│   ├── robot.py                  # DH Robot base class (bug fixes only)
│   ├── robot_pinocchio.py        # Pinocchio Robot base class (unchanged)
│   ├── robot_config.py           # Robot config (unchanged)
│   ├── ur5e.py                   # UR5e (unchanged in Phase 1)
│   ├── iiwa14.py                 # IIWA14 (bug fix only)
│   ├── diana.py                  # Diana (bug fixes)
│   ├── sim/                      # NEW: Simulation package
│   │   ├── __init__.py
│   │   ├── mujoco_sim.py         # MuJoCo environment
│   │   ├── dh_robot_sim.py       # DH simulation base
│   │   ├── pinocchio_robot_sim.py # Pinocchio simulation base
│   │   ├── task_env.py           # Task environment
│   │   └── dual_arm_sim.py       # Dual-arm wrapper
│   └── test/                     # Existing tests (preserved) + new tests
│       ├── test_mujoco.py        # Existing (preserved)
│       ├── dual_mujoco.py        # Existing (preserved)
│       ├── franka_dual.py        # Existing (preserved)
│       ├── traj.py               # Existing (preserved)
│       ├── traj_resume.py        # Existing (preserved)
│       ├── traj_resume_replaced.py # Existing (preserved)
│       ├── test_sim_ur5e.py      # NEW: UR5e sim test
│       ├── test_sim_diana.py     # NEW: Diana sim test
│       ├── test_task.py          # NEW: Task env test
│       └── test_dual_arm.py      # NEW: Dual-arm test
└── utils/
    ├── __init__.py
    ├── mj.py                     # Existing (preserved)
    └── rtb.py                    # Existing (preserved)
```

---

## 7. Implementation Priority

| Priority | Item | Rationale |
|----------|------|-----------|
| P0 | BUG-1 through BUG-5 bug fixes | Correctness of existing code |
| P1 | `MuJoCoSim` (FR-1) | Foundation for all simulation features; no dependencies |
| P2 | `DHRobotSim` (FR-2) | Depends on MuJoCoSim; enables UR5e/IIWA14/Diana simulation |
| P2 | `PinocchioRobotSim` (FR-3) | Depends on MuJoCoSim; enables Diana Pinocchio simulation |
| P3 | `TaskEnv` (FR-4) | Depends on MuJoCoSim + RobotSim |
| P3 | `DualArmSim` (FR-5) | Depends on MuJoCoSim + RobotSim |
| P4 | Test scripts | Validate all new code |
| P5 | Adapt `UR5e`/`IIWA14`/`Diana` to inherit from `DHRobotSim` | Maximize code reuse (deferred to Phase 5 of plan) |
