# Development Notes: RoboSimXCtrl Simulation Refactoring

**Date:** 2026-05-31
**Author:** Development Team

---

## Project Overview

RoboSimXCtrl is a robot simulation and control framework supporting DH-parameter robots (UR5e, IIWA14) and URDF-based robots (Diana) with MuJoCo physics simulation. The simulation refactoring introduces:

- A unified `MuJoCoSim` wrapper class
- `DHRobotSim` base class for DH-parameter robots with optional MuJoCo backend
- `PinocchioRobotSim` base class for Pinocchio-based robots with MuJoCo backend
- `TaskEnv` for manipulation task environments
- `DualArmSim` for dual-arm coordination
- Bug fixes for IIWA14, Robot serialization, disable_tool, and Diana URDF paths

---

## Architecture Decisions

### 1. Inheritance Hierarchy

```
Robot (robot/robot.py)           -- DH-based abstract Robot
  +-- DHRobotSim (robot/sim/)    -- Adds MuJoCo + Meshcat to DH Robot
  +-- UR5e, IIWA14               -- Concrete DH robots

robot_pinocchio.Robot            -- Pinocchio-based abstract Robot
  +-- PinocchioRobotSim          -- Adds MuJoCo to Pinocchio Robot
  +-- Diana                      -- Concrete Pinocchio robot
```

### 2. MuJoCo as Optional Dependency

MuJoCo is guarded behind `try/except ImportError` with a `_HAS_MUJOCO` flag. The `MuJoCoSim` class is always defined but raises `ImportError` on instantiation if mujoco is not installed. This allows all modules to be imported without mujoco.

### 3. Delegation Pattern

Both `DHRobotSim` and `PinocchioRobotSim` delegate MuJoCo operations to a `MuJoCoSim` instance (`self.mj_sim`). Methods like `mujoco_step()`, `mujoco_set_joint()`, `mujoco_get_joint()` check `if self.mj_sim is not None` and are no-ops when MuJoCo is not enabled.

### 4. servoJ as the Single State Update Point

Both `DHRobotSim.servoJ` and `PinocchioRobotSim.servoJ` follow the pattern:
1. Position/velocity limit checks
2. Update internal state (`self.q0` or `self.q`)
3. Sync MuJoCo joints
4. Sync Meshcat visualization

`MoveJ` and `MoveL` are NOT reimplemented -- they are inherited from the parent Robot class where available.

### 5. Bilingual Documentation

All new modules include bilingual (Chinese/English) docstrings for both module-level and method-level documentation.

---

## Implementation Summary

### Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `robot/sim/__init__.py` | Created | Package init with all sim module exports |
| `robot/sim/mujoco_sim.py` | Created | MuJoCo wrapper class |
| `robot/sim/dh_robot_sim.py` | Created | DH Robot + MuJoCo/Meshcat base |
| `robot/sim/pinocchio_robot_sim.py` | Created | Pinocchio Robot + MuJoCo base |
| `robot/sim/task_env.py` | Created | Manipulation task environment |
| `robot/sim/dual_arm_sim.py` | Created | Dual-arm simulation manager |
| `robot/iiwa14.py` | Fixed | BUG-1: Link construction indentation |
| `robot/robot.py` | Fixed | BUG-2: pickle range(self._dof); BUG-3: disable_tool SE3 |
| `robot/diana.py` | Fixed | BUG-4: removed bad method; BUG-5: relative URDF path |

### Key API Methods

**MuJoCoSim:**
- `__init__(xml_path, timestep)` -- Load MuJoCo scene
- `step(ctrl)` / `reset()` / `forward()` -- Simulation control
- `set_joint_q(name, q)` / `get_joint_q(name)` -- Joint manipulation
- `get_body_pose(name)` / `set_body_pose(name, pos)` -- Body pose
- `attach(eq_name, joint_name, T)` / `detach(eq_name)` -- Constraints
- `launch_viewer()` / `close_viewer()` / `sync_viewer()` -- Viewer
- `render_rgb(camera, w, h)` / `render_depth(camera, w, h)` -- Rendering
- `get_point_cloud(camera, w, h)` -- Point cloud generation

**DHRobotSim:**
- `mujoco_step()` / `mujoco_set_joint(q)` / `mujoco_get_joint()` -- MuJoCo interface
- `servoJ(q_target, delta_t)` -- Joint servo with limit checks + sync
- `meshcat_display(q)` / `meshcat_display_trajectory(q_list)` -- Visualization

**PinocchioRobotSim:**
- Same MuJoCo interface as DHRobotSim
- `servoJ(q_target, delta_t)` -- Calls parent servoJ then syncs MuJoCo

**TaskEnv:**
- `spawn_object(name, pos)` / `remove_object(name)` -- Object management
- `grasp(name)` / `release(name)` -- Constraint-based grasping
- `get_observation()` -- Observation dict with joint_pos, joint_vel, ee_pose
- `reset_scene()` -- Full scene reset

---

## Test Results Summary

- **120 total tests** across 4 test files
- **105 passed**, **15 skipped**, **0 failed**
- Skipped tests require `tracikpy` (not available) for Diana robot tests
- All P0 test cases that can be automated have been verified

See `docs/TEST-REPORT.md` for detailed results.

---

## Known Limitations and Future Work

### Critical Issues (from Code Review)

1. **C-1: DHRobotSim.servoJ does NOT call super().servoJ()** -- The implementation is standalone and does not call the parent Robot's servoJ (which doesn't exist). This is actually correct behavior since the DH Robot base class has no servoJ method. The velocity check uses a hardcoded 3.0 rad/s limit instead of per-joint limits.

2. **C-2: DHRobotSim lacks MoveJ/MoveL** -- The DH Robot base class does not define MoveJ/MoveL. Only PinocchioRobotSim inherits these from its parent. DHRobotSim users must implement their own motion planners or use servoJ directly.

### Major Issues

3. **M-1**: Hardcoded velocity limit in servoJ (3.0 rad/s)
4. **M-2**: Meshcat import guard checks pinocchio but uses standalone meshcat
5. **M-3**: `robot/sim/__init__.py` imports unconditionally -- fails if dependencies missing
6. **M-4**: Velocity check uses last commanded position, not actual MuJoCo state
7. **M-5**: DualArmSim.MoveJ_dual_arm has thread safety concerns (shared MuJoCo state)

### Minor Issues

8. **m-1**: servoJ uses `print()` instead of `warnings.warn()`
9. **m-3**: TaskEnv.get_observation missing "point_cloud" key
10. **m-4**: MuJoCoSim.set_body_pose only sets position, not orientation
11. **m-5**: get_point_cloud uses slow Python loop (should be vectorized)
12. **m-6**: spawn_object doesn't actually create objects in MuJoCo
13. **m-7**: meshcat_display only shows end-effector frame, not full robot

### Additional Issue Found During Testing

14. **Import Bug**: `robot/ur5e.py` uses bare imports (`from robot import Robot`) instead of relative imports (`from .robot import Robot`), causing circular import through `robot/__init__.py`. This prevents `from robot import UR5e` from working without path manipulation.

15. **NumPy Deprecation**: `dh_robot_sim.py` line 128 uses array-to-scalar conversion that is deprecated in NumPy 1.25+.

### Future Work

- Implement MoveJ/MoveL for DHRobotSim (trapezoidal velocity profile)
- Vectorize get_point_cloud() with numpy
- Add thread-safe MuJoCo operations for DualArmSim
- Add TaskEnv integration tests with full MuJoCo scenes
- Fix ur5e.py imports to use relative imports
- Add CI/CD pipeline with headless test execution
