# Test Plan: RoboSimXCtrl Simulation Refactoring

**Version:** 1.0
**Date:** 2026-05-31
**Status:** Draft

---

## Test Environment Requirements

- **Python:** 3.8+
- **Required packages:** numpy, spatialmath-python, roboticstoolbox, modern_robotics
- **Optional packages:** mujoco, meshcat, pinocchio, pyroboplan, tracikpy
- **Test assets:** MuJoCo XML scene files, Diana URDF

---

## T1: Unit Tests -- MuJoCoSim

### T1.1: Load Scene XML and Verify Model Loaded

| Field | Value |
|-------|-------|
| **Test ID** | T1.1 |
| **Name** | Load MuJoCo Scene XML |
| **Description** | Verify that MuJoCoSim correctly loads a MuJoCo XML scene file and initializes model/data |
| **Preconditions** | mujoco package installed; a valid MuJoCo XML scene file exists |
| **Priority** | P0 |

**Steps:**
1. Create a minimal MuJoCo XML file (e.g., a single free body in an empty world)
2. Instantiate `MuJoCoSim(xml_path, timestep=0.001)`
3. Check that `sim.model` is not None
4. Check that `sim.data` is not None
5. Check that `sim.model.opt.timestep == 0.001`

**Expected Result:** MuJoCoSim loads successfully with valid model and data objects. Custom timestep is applied.

**Pass/Fail Criteria:**
- PASS: `sim.model` is a `mujoco.MjModel`, `sim.data` is a `mujoco.MjData`, `sim.model.opt.timestep == 0.001`
- FAIL: Any exception during construction, or model/data is None

---

### T1.2: Set/Get Joint Positions

| Field | Value |
|-------|-------|
| **Test ID** | T1.2 |
| **Name** | Joint Position Set/Get |
| **Description** | Verify that joint positions can be set and retrieved by name and by ID |
| **Preconditions** | MuJoCoSim loaded with a scene containing named joints |
| **Priority** | P0 |

**Steps:**
1. Load a MuJoCo scene with at least one named joint (e.g., "joint_1")
2. Call `sim.set_joint_q("joint_1", 0.5)`
3. Call `q = sim.get_joint_q("joint_1")`
4. Verify `q == 0.5`
5. Get the joint ID via `sim._resolve_joint_id("joint_1")`
6. Call `sim.set_joint_q(joint_id, 1.0)` using the integer ID
7. Call `q = sim.get_joint_q(joint_id)`
8. Verify `q == 1.0`

**Expected Result:** Joint positions are correctly set and retrieved by both name and integer ID.

**Pass/Fail Criteria:**
- PASS: Both name-based and ID-based set/get return the correct values
- FAIL: Mismatch in set/get values, or NameError/KeyError for valid joint names

---

### T1.3: Step Simulation and Verify State Changes

| Field | Value |
|-------|-------|
| **Test ID** | T1.3 |
| **Name** | Simulation Step |
| **Description** | Verify that stepping the simulation advances time and changes state |
| **Preconditions** | MuJoCoSim loaded with a dynamic scene (e.g., a falling body) |
| **Priority** | P0 |

**Steps:**
1. Load a scene with gravity and a free body
2. Record initial time: `t0 = sim.data.time`
3. Call `sim.step()`
4. Record new time: `t1 = sim.data.time`
5. Verify `t1 > t0`
6. Call `sim.reset()`
7. Verify `sim.data.time == 0.0`

**Expected Result:** Time advances after step; time resets to 0 after reset.

**Pass/Fail Criteria:**
- PASS: `t1 - t0 == sim.model.opt.timestep` (within float tolerance); `sim.data.time == 0.0` after reset
- FAIL: Time does not advance, or reset does not restore initial state

---

### T1.4: Body Pose Get/Set

| Field | Value |
|-------|-------|
| **Test ID** | T1.4 |
| **Name** | Body Pose Manipulation |
| **Description** | Verify that body poses can be queried and set |
| **Preconditions** | MuJoCoSim loaded with a scene containing a named body |
| **Priority** | P0 |

**Steps:**
1. Load a scene with a body named "box"
2. Call `pose = sim.get_body_pose("box")`
3. Verify `pose` is a valid `spatialmath.SE3` object
4. Call `pos = sim.get_body_pose_xyz("box")`
5. Verify `pos` is a 3-element numpy array matching `pose.t`
6. Call `sim.set_body_pose("box", np.array([1.0, 2.0, 3.0]))`
7. Call `pos_new = sim.get_body_pose_xyz("box")`
8. Verify `pos_new == [1.0, 2.0, 3.0]`

**Expected Result:** Body pose is correctly queried as SE3 and position can be set.

**Pass/Fail Criteria:**
- PASS: SE3 pose is valid; position matches after set
- FAIL: Wrong type returned, position mismatch after set

---

### T1.5: Attach/Detach Constraints

| Field | Value |
|-------|-------|
| **Test ID** | T1.5 |
| **Name** | Equality Constraint Attach/Detach |
| **Description** | Verify that equality constraints can be activated and deactivated for grasping |
| **Preconditions** | MuJoCoSim loaded with a scene containing an equality constraint and a free joint |
| **Priority** | P1 |

**Steps:**
1. Load a scene with an equality constraint named "grasp_eq" and a free joint "box_free"
2. Call `sim.attach("grasp_eq", "box_free", SE3())`
3. Verify `sim.data.eq_active[eq_id] == 1`
4. Call `sim.detach("grasp_eq")`
5. Verify `sim.data.eq_active[eq_id] == 0`
6. Verify that calling `sim.attach("nonexistent", "box_free", SE3())` raises `ValueError`

**Expected Result:** Constraints are correctly activated/deactivated; invalid names raise errors.

**Pass/Fail Criteria:**
- PASS: eq_active toggles correctly; ValueError for invalid names
- FAIL: eq_active does not change, or no error for invalid names

---

### T1.6: Viewer Management

| Field | Value |
|-------|-------|
| **Test ID** | T1.6 |
| **Name** | Viewer Launch/Close/Sync |
| **Description** | Verify viewer lifecycle management |
| **Preconditions** | MuJoCoSim loaded; display available (not headless) |
| **Priority** | P2 |

**Steps:**
1. Verify `sim.viewer is None` initially
2. Call `sim.launch_viewer()`
3. Verify `sim.viewer is not None`
4. Call `sim.step()` (should not raise)
5. Call `sim.close_viewer()`
6. Verify `sim.viewer is None`
7. Call `sim.close_viewer()` again (should not raise)

**Expected Result:** Viewer can be launched, used, and closed without errors.

**Pass/Fail Criteria:**
- PASS: Viewer lifecycle works; no errors on double-close
- FAIL: Viewer launch fails, or double-close raises an exception

---

### T1.7: Offscreen Rendering

| Field | Value |
|-------|-------|
| **Test ID** | T1.7 |
| **Name** | RGB/Depth Rendering and Point Cloud |
| **Description** | Verify offscreen rendering produces correct image shapes |
| **Preconditions** | MuJoCoSim loaded with a scene containing a camera |
| **Priority** | P2 |

**Steps:**
1. Call `rgb = sim.render_rgb(0, 256, 256)`
2. Verify `rgb.shape == (256, 256, 3)` and `rgb.dtype == np.uint8`
3. Call `depth = sim.render_depth(0, 256, 256)`
4. Verify `depth.shape == (256, 256)` and `depth.dtype == np.float32` (or float64)
5. Call `pcd = sim.get_point_cloud(0, 64, 64)`
6. Verify `pcd.shape[1] == 6` (x, y, z, r, g, b)
7. Verify `pcd.shape[0] <= 64 * 64`

**Expected Result:** Images have correct shapes and types.

**Pass/Fail Criteria:**
- PASS: All shapes and dtypes match expected values
- FAIL: Shape mismatch, wrong dtype, or rendering exception

---

### T1.8: Optional Dependency Graceful Failure

| Field | Value |
|-------|-------|
| **Test ID** | T1.8 |
| **Name** | MuJoCo Not Installed Graceful Failure |
| **Description** | Verify that MuJoCoSim raises a clear ImportError when mujoco is not installed |
| **Preconditions** | mujoco package NOT installed (or mocked as unavailable) |
| **Priority** | P1 |

**Steps:**
1. Mock `mujoco` as unavailable (monkeypatch `_HAS_MUJOCO = False`)
2. Attempt `MuJoCoSim("dummy.xml")`
3. Verify `ImportError` is raised with a clear message

**Expected Result:** Clear ImportError message directing user to install mujoco.

**Pass/Fail Criteria:**
- PASS: ImportError raised with message containing "pip install mujoco"
- FAIL: Generic ImportError, or crash without clear message

---

## T2: Integration Tests -- DHRobotSim

### T2.1: Create UR5e with MuJoCo, Verify Both Models Initialized

| Field | Value |
|-------|-------|
| **Test ID** | T2.1 |
| **Name** | DHRobotSim Initialization with MuJoCo |
| **Description** | Verify that DHRobotSim initializes both the DH kinematic model and MuJoCo simulation |
| **Preconditions** | mujoco installed; UR5e MuJoCo XML scene exists |
| **Priority** | P0 |

**Steps:**
1. Create a concrete DHRobotSim subclass (or use UR5e if adapted) with `mujoco_xml_path="path/to/ur5e_scene.xml"`
2. Verify `robot.mj_sim is not None`
3. Verify `robot.mj_sim.model` is a valid MjModel
4. Verify `robot.robot` (the rtb.DHRobot) is not None
5. Verify `robot.fkine([0]*6)` returns a valid SE3

**Expected Result:** Both DH kinematics and MuJoCo simulation are functional.

**Pass/Fail Criteria:**
- PASS: mj_sim and rtb.DHRobot both initialized; fkine works
- FAIL: Any AttributeError or initialization failure

---

### T2.2: servoJ Updates q0, MuJoCo State, and Meshcat

| Field | Value |
|-------|-------|
| **Test ID** | T2.2 |
| **Name** | servoJ State Synchronization |
| **Description** | Verify that servoJ updates internal state, MuJoCo joints, and optionally Meshcat |
| **Preconditions** | DHRobotSim with MuJoCo enabled |
| **Priority** | P0 |
| **Note** | BLOCKED by C-1 (super().servoJ does not exist). Must fix C-1 first. |

**Steps:**
1. Create DHRobotSim with MuJoCo XML
2. Set initial state: `robot.q0 = [0]*6`
3. Call `result = robot.servoJ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])`
4. Verify `result == 0`
5. Verify `robot.q0 == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]`
6. Verify MuJoCo joints match: `robot.mujoco_get_joint()` approximately equals target

**Expected Result:** All three state sinks (q0, MuJoCo, Meshcat) are updated.

**Pass/Fail Criteria:**
- PASS: q0 updated, MuJoCo joints match, no exceptions
- FAIL: q0 not updated, MuJoCo mismatch, or exception (especially AttributeError from C-1)

---

### T2.3: MoveJ Executes Trapezoidal Trajectory Through servoJ

| Field | Value |
|-------|-------|
| **Test ID** | T2.3 |
| **Name** | MoveJ Trajectory Execution |
| **Description** | Verify that MoveJ generates and executes a trapezoidal velocity profile |
| **Preconditions** | DHRobotSim with MoveJ available (blocked by C-2) |
| **Priority** | P0 |
| **Note** | BLOCKED by C-2 (MoveJ not available in DH Robot base). Must fix C-2 first. |

**Steps:**
1. Create DHRobotSim subclass with DH parameters and MuJoCo XML
2. Call `robot.MoveJ([0.5]*6, v_max=1.0, a_max=2.0, dt=0.001)`
3. Verify final `robot.q0` is approximately `[0.5]*6`
4. Verify MuJoCo joints match

**Expected Result:** Robot reaches target via smooth trajectory.

**Pass/Fail Criteria:**
- PASS: Final position within tolerance of target (e.g., 1e-3 rad)
- FAIL: Position error exceeds tolerance, or AttributeError

---

### T2.4: fkine/ikine Round-Trip

| Field | Value |
|-------|-------|
| **Test ID** | T2.4 |
| **Name** | Forward/Inverse Kinematics Round-Trip |
| **Description** | Verify that fkine followed by ikine returns to the original joint configuration |
| **Preconditions** | DHRobotSim subclass with ikine implemented |
| **Priority** | P1 |

**Steps:**
1. Set `q_orig = [0.3, -0.5, 0.8, 0.1, -0.2, 0.4]`
2. Compute `T = robot.fkine(q_orig)`
3. Compute `q_solved = robot.ikine(T)`
4. Verify `q_solved` is not empty
5. Compute `T_check = robot.fkine(q_solved)`
6. Verify `SE3(T).inv() * SE3(T_check)` is approximately identity (position error < 1mm)

**Expected Result:** fkine(ikine(T)) returns approximately T.

**Pass/Fail Criteria:**
- PASS: Position error < 0.001m, rotation error < 0.01 rad
- FAIL: Error exceeds tolerance, or ikine returns empty

---

### T2.5: Dynamics Methods Still Work

| Field | Value |
|-------|-------|
| **Test ID** | T2.5 |
| **Name** | Dynamics Method Inheritance |
| **Description** | Verify that inherited dynamics methods (get_inertia, inv_dynamics, etc.) work through DHRobotSim |
| **Preconditions** | DHRobotSim subclass with full DH parameters |
| **Priority** | P1 |

**Steps:**
1. Set `q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]`
2. Call `M = robot.get_inertia(q)` -- verify returns 6x6 matrix
3. Call `C = robot.get_coriolis(q, dq)` -- verify returns 6x6 matrix
4. Call `g = robot.get_gravity(q)` -- verify returns 6-element vector
5. Call `tau = robot.inv_dynamics(q, dq, ddq)` -- verify returns 6-element vector

**Expected Result:** All dynamics methods return correct-shaped results.

**Pass/Fail Criteria:**
- PASS: Correct shapes; values are physically reasonable (positive inertia diagonal, etc.)
- FAIL: AttributeError (not inherited), wrong shapes, or NaN values

---

### T2.6: MuJoCo Operations Without MuJoCo Enabled

| Field | Value |
|-------|-------|
| **Test ID** | T2.6 |
| **Name** | Graceful No-Op Without MuJoCo |
| **Description** | Verify that all MuJoCo methods are no-ops when mj_sim is None |
| **Preconditions** | DHRobotSim without mujoco_xml_path |
| **Priority** | P1 |

**Steps:**
1. Create DHRobotSim without MuJoCo: `robot = DHRobotSim()`
2. Verify `robot.mj_sim is None`
3. Call `robot.mujoco_step()` -- should not raise
4. Call `robot.mujoco_set_joint([0]*6)` -- should not raise
5. Call `q = robot.mujoco_get_joint()` -- should return empty array
6. Call `pose = robot.mujoco_get_body_pose("anything")` -- should return SE3()

**Expected Result:** All methods gracefully handle None mj_sim.

**Pass/Fail Criteria:**
- PASS: No exceptions; mujoco_get_joint returns `np.array([])`; mujoco_get_body_pose returns `SE3()`
- FAIL: Any exception

---

## T3: Integration Tests -- PinocchioRobotSim

### T3.1: Create Diana with Pinocchio + MuJoCo

| Field | Value |
|-------|-------|
| **Test ID** | T3.1 |
| **Name** | PinocchioRobotSim Initialization |
| **Description** | Verify that PinocchioRobotSim initializes both Pinocchio and MuJoCo models |
| **Preconditions** | pinocchio, pyroboplan installed; Diana URDF exists; MuJoCo XML exists |
| **Priority** | P0 |

**Steps:**
1. Create PinocchioRobotSim: `robot = PinocchioRobotSim(urdf_path, mesh_dir, mujoco_xml_path="path/to/diana_scene.xml")`
2. Verify `robot.mj_sim is not None`
3. Verify `robot.model` (Pinocchio model) is not None
4. Verify `robot.q` is a 7-element array

**Expected Result:** Both Pinocchio and MuJoCo models initialized.

**Pass/Fail Criteria:**
- PASS: Both models valid; 7 DOF confirmed
- FAIL: Initialization exception

---

### T3.2: servoJ Syncs Pinocchio and MuJoCo

| Field | Value |
|-------|-------|
| **Test ID** | T3.2 |
| **Name** | servoJ Dual Sync |
| **Description** | Verify that servoJ updates both Pinocchio and MuJoCo joint states |
| **Preconditions** | PinocchioRobotSim with MuJoCo |
| **Priority** | P0 |

**Steps:**
1. Create PinocchioRobotSim with MuJoCo
2. Set target: `q_target = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])`
3. Call `result = robot.servoJ(q_target)`
4. Verify `result == 0`
5. Verify `robot.q` approximately equals `q_target` (Pinocchio state)
6. Verify `robot.mujoco_get_joint()` approximately equals `q_target` (MuJoCo state)

**Expected Result:** Both Pinocchio and MuJoCo states match the target.

**Pass/Fail Criteria:**
- PASS: Both states match within tolerance (1e-6 rad)
- FAIL: State mismatch

---

### T3.3: MoveJ/MoveL Work End-to-End

| Field | Value |
|-------|-------|
| **Test ID** | T3.3 |
| **Name** | MoveJ/MoveL End-to-End |
| **Description** | Verify that MoveJ and MoveL execute correctly with MuJoCo sync |
| **Preconditions** | PinocchioRobotSim with MuJoCo; Meshcat available |
| **Priority** | P0 |

**Steps:**
1. Create PinocchioRobotSim with MuJoCo
2. Set initial: `robot.q = np.zeros(7)`
3. Call `robot.MoveJ(np.array([0.3, 0.2, 0.1, 0.5, 0.3, 0.2, 0.1]))`
4. Verify final `robot.q` is approximately the target
5. Verify MuJoCo joints match
6. Reset and test MoveL with a Cartesian path

**Expected Result:** Both MoveJ and MoveL reach their targets.

**Pass/Fail Criteria:**
- PASS: Joint position error < 1e-3 rad after MoveJ
- FAIL: Large position error, or exception during execution

---

### T3.4: Collision Detection Still Works

| Field | Value |
|-------|-------|
| **Test ID** | T3.4 |
| **Name** | Collision Detection Inheritance |
| **Description** | Verify that collision detection from Pinocchio Robot still works |
| **Preconditions** | PinocchioRobotSim with collision model |
| **Priority** | P1 |

**Steps:**
1. Create PinocchioRobotSim
2. Check `robot.collision_model` is not None
3. Check `robot.collision_data` is not None
4. If collision checking method exists, call it with a known collision-free configuration
5. Verify no collision detected

**Expected Result:** Collision model is available and functional.

**Pass/Fail Criteria:**
- PASS: Collision model initialized; collision check returns expected result
- FAIL: AttributeError or wrong collision result

---

## T4: Task Environment Tests

### T4.1: Spawn and Remove Objects

| Field | Value |
|-------|-------|
| **Test ID** | T4.1 |
| **Name** | Object Spawn/Remove |
| **Description** | Verify that objects can be spawned and removed from the task environment |
| **Preconditions** | TaskEnv with MuJoCoSim; scene with pre-defined bodies |
| **Priority** | P1 |

**Steps:**
1. Create TaskEnv: `env = TaskEnv(robot, mujoco_sim)`
2. Call `env.spawn_object("box1", np.array([0.5, 0, 0.3]))`
3. Verify "box1" is in `env._spawned_objects`
4. Call `env.remove_object("box1")`
5. Verify "box1" is NOT in `env._spawned_objects`
6. Verify the body is moved far away in MuJoCo (position ~ [100, 100, 100])

**Expected Result:** Objects are tracked and removed correctly.

**Pass/Fail Criteria:**
- PASS: Object tracking works; MuJoCo body position updated
- FAIL: Object not tracked, or MuJoCo body not moved

---

### T4.2: Grasp and Release

| Field | Value |
|-------|-------|
| **Test ID** | T4.2 |
| **Name** | Grasp/Release Operations |
| **Description** | Verify that grasp activates constraint and release deactivates it |
| **Preconditions** | TaskEnv with MuJoCoSim; scene with equality constraint and free joint |
| **Priority** | P1 |

**Steps:**
1. Create TaskEnv
2. Call `env.grasp("box1")`
3. Verify "box1" is in `env._grasp_constraints`
4. Call `env.release("box1")`
5. Verify "box1" is NOT in `env._grasp_constraints`
6. Call `env.grasp("nonexistent")` -- should warn, not crash

**Expected Result:** Grasp/release track constraints; invalid objects handled gracefully.

**Pass/Fail Criteria:**
- PASS: Constraint tracking works; warnings for invalid objects
- FAIL: Crash on invalid object, or constraint not tracked

---

### T4.3: Get Observation Returns Expected Keys

| Field | Value |
|-------|-------|
| **Test ID** | T4.3 |
| **Name** | Observation Generation |
| **Description** | Verify that get_observation returns a dict with expected keys |
| **Preconditions** | TaskEnv with robot that has q0 and fkine |
| **Priority** | P1 |

**Steps:**
1. Create TaskEnv with a robot that has `q0` and `fkine`
2. Call `obs = env.get_observation()`
3. Verify `obs` is a dict
4. Verify `"joint_pos" in obs`
5. Verify `"ee_pose" in obs`
6. Verify `"joint_vel" in obs`
7. Verify `obs["joint_pos"]` is a numpy array
8. Verify `obs["ee_pose"]` is an SE3 object

**Expected Result:** Observation dict contains expected keys with correct types.

**Pass/Fail Criteria:**
- PASS: All expected keys present with correct types
- FAIL: Missing keys, wrong types, or exception

---

### T4.4: Scene Reset

| Field | Value |
|-------|-------|
| **Test ID** | T4.4 |
| **Name** | Scene Reset |
| **Description** | Verify that reset_scene restores all objects and robot to initial state |
| **Preconditions** | TaskEnv with spawned objects and moved robot |
| **Priority** | P1 |

**Steps:**
1. Create TaskEnv and spawn objects
2. Move robot to a non-zero configuration
3. Call `env.reset_scene()`
4. Verify all spawned objects are at their initial positions
5. Verify robot `q0` is reset to zeros
6. Verify MuJoCo simulation is reset

**Expected Result:** Full scene reset to initial state.

**Pass/Fail Criteria:**
- PASS: Objects and robot at initial positions; MuJoCo time is 0
- FAIL: Objects or robot not reset

---

## T5: Backward Compatibility

### T5.1: Import robot.Robot Still Works

| Field | Value |
|-------|-------|
| **Test ID** | T5.1 |
| **Name** | Robot Import Compatibility |
| **Description** | Verify that existing import paths still work |
| **Preconditions** | All dependencies installed |
| **Priority** | P0 |

**Steps:**
1. Run `from robot import Robot`
2. Verify `Robot` is the DH-based Robot class
3. Run `from robot import UR5e, IIWA14, Diana`
4. Verify all imports succeed

**Expected Result:** All existing imports work without modification.

**Pass/Fail Criteria:**
- PASS: All imports succeed
- FAIL: ImportError or wrong class returned

---

### T5.2: Import robot.UR5e Still Works

| Field | Value |
|-------|-------|
| **Test ID** | T5.2 |
| **Name** | UR5e Import and Instantiation |
| **Description** | Verify that UR5e can be imported and instantiated as before |
| **Preconditions** | roboticstoolbox installed |
| **Priority** | P0 |

**Steps:**
1. Run `from robot import UR5e`
2. Run `ur = UR5e()`
3. Verify `ur.dof == 6`
4. Verify `ur.fkine([0]*6)` returns a valid SE3

**Expected Result:** UR5e works identically to before refactoring.

**Pass/Fail Criteria:**
- PASS: UR5e instantiates and fkine works
- FAIL: Import error, instantiation error, or wrong DOF

---

### T5.3: IIWA14 Bug Fix (BUG-1) Verified

| Field | Value |
|-------|-------|
| **Test ID** | T5.3 |
| **Name** | IIWA14 Robot Initialization After BUG-1 Fix |
| **Description** | Verify that IIWA14 creates a 7-link robot (not 1-link) |
| **Preconditions** | roboticstoolbox installed |
| **Priority** | P0 |

**Steps:**
1. Run `from robot import IIWA14`
2. Run `robot = IIWA14()`
3. Verify `robot.robot.n == 7` (7 links)
4. Verify `robot.dof == 7`
5. Verify `robot.fkine([0]*7)` returns a valid SE3

**Expected Result:** IIWA14 has 7 links, not 1 (the BUG-1 fix).

**Pass/Fail Criteria:**
- PASS: `robot.robot.n == 7`
- FAIL: `robot.robot.n != 7` (BUG-1 not fixed)

---

### T5.4: Diana Bug Fix (BUG-5) Verified

| Field | Value |
|-------|-------|
| **Test ID** | T5.4 |
| **Name** | Diana URDF Path Resolution |
| **Description** | Verify that Diana uses a relative path, not hardcoded absolute path |
| **Preconditions** | tracikpy installed; URDF at `assets/urdf/diana7_description/urdf/diana_v2.urdf` |
| **Priority** | P0 |

**Steps:**
1. Run `from robot import Diana`
2. Run `robot = Diana()`
3. Verify no exception (URDF found via relative path)
4. Verify `robot.ik_solver` is initialized
5. Verify no hardcoded `/home/sun/...` path in `robot/diana.py` source

**Expected Result:** Diana initializes without hardcoded path.

**Pass/Fail Criteria:**
- PASS: Diana() succeeds; no hardcoded paths in source
- FAIL: FileNotFoundError or hardcoded path found

---

### T5.5: Robot Serialization (BUG-2)

| Field | Value |
|-------|-------|
| **Test ID** | T5.5 |
| **Name** | Robot Pickle Round-Trip |
| **Description** | Verify that 7-DOF robots survive pickle serialization |
| **Preconditions** | BUG-2 fixed (range(6) changed to range(self._dof)) |
| **Priority** | P1 |

**Steps:**
1. Create `robot = IIWA14()`
2. Serialize: `data = pickle.dumps(robot)`
3. Deserialize: `robot2 = pickle.loads(data)`
4. Verify `robot2.dof == 7`
5. Verify `robot2.fkine([0]*7)` matches `robot.fkine([0]*7)`

**Expected Result:** 7-DOF robot survives pickle round-trip.

**Pass/Fail Criteria:**
- PASS: dof == 7 after deserialization; fkine matches
- FAIL: dof != 7 (BUG-2 not fixed), or deserialization error

---

### T5.6: Robot disable_tool (BUG-3)

| Field | Value |
|-------|-------|
| **Test ID** | T5.6 |
| **Name** | disable_tool Type Consistency |
| **Description** | Verify that disable_tool sets _tool to SE3, not numpy array |
| **Preconditions** | BUG-3 fixed |
| **Priority** | P1 |

**Steps:**
1. Create a robot (e.g., UR5e)
2. Call `robot.set_tool(np.array([0, 0, 0.1]))`
3. Verify `robot._tool` is SE3
4. Call `robot.disable_tool()`
5. Verify `robot._tool` is SE3 (not numpy array)
6. Verify `robot.fkine([0]*6)` works without AttributeError

**Expected Result:** disable_tool sets _tool to SE3(), preserving type consistency.

**Pass/Fail Criteria:**
- PASS: `_tool` is SE3 after disable; fkine works
- FAIL: `_tool` is numpy array; fkine raises AttributeError

---

## Test Execution Summary

| Test ID | Name | Priority | Status | Blocked By |
|---------|------|----------|--------|------------|
| T1.1 | Load Scene XML | P0 | Not Run | -- |
| T1.2 | Set/Get Joints | P0 | Not Run | -- |
| T1.3 | Step Simulation | P0 | Not Run | -- |
| T1.4 | Body Pose | P0 | Not Run | -- |
| T1.5 | Attach/Detach | P1 | Not Run | -- |
| T1.6 | Viewer Management | P2 | Not Run | -- |
| T1.7 | Offscreen Rendering | P2 | Not Run | -- |
| T1.8 | Graceful Failure | P1 | Not Run | -- |
| T2.1 | DH+MuJoCo Init | P0 | Not Run | -- |
| T2.2 | servoJ Sync | P0 | BLOCKED | C-1 |
| T2.3 | MoveJ Trajectory | P0 | BLOCKED | C-2 |
| T2.4 | fkine/ikine Round-Trip | P1 | Not Run | -- |
| T2.5 | Dynamics Methods | P1 | Not Run | -- |
| T2.6 | No-Op Without MuJoCo | P1 | Not Run | -- |
| T3.1 | Pinocchio+MuJoCo Init | P0 | Not Run | -- |
| T3.2 | Pinocchio servoJ Sync | P0 | Not Run | -- |
| T3.3 | MoveJ/MoveL E2E | P0 | Not Run | -- |
| T3.4 | Collision Detection | P1 | Not Run | -- |
| T4.1 | Spawn/Remove Objects | P1 | Not Run | -- |
| T4.2 | Grasp/Release | P1 | Not Run | -- |
| T4.3 | Get Observation | P1 | Not Run | -- |
| T4.4 | Scene Reset | P1 | Not Run | -- |
| T5.1 | Robot Import | P0 | Not Run | -- |
| T5.2 | UR5e Import | P0 | Not Run | -- |
| T5.3 | IIWA14 BUG-1 Fix | P0 | Not Run | -- |
| T5.4 | Diana BUG-5 Fix | P0 | Not Run | -- |
| T5.5 | Pickle Round-Trip | P1 | Not Run | -- |
| T5.6 | disable_tool | P1 | Not Run | -- |

**Total Tests:** 28
**P0 (Must Pass):** 14
**P1 (Should Pass):** 12
**P2 (Nice to Have):** 2
**Blocked:** 2 (by C-1, C-2)
