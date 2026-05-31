# Development Log - Simulation Framework Implementation

**Date:** 2026-05-31
**Phase:** Phase 1 - Core Simulation Framework

---

## Summary

Implemented the unified simulation framework for RoboSimXCtrl, extracting MuJoCo interaction logic from scattered test scripts into a clean, reusable class hierarchy. Also fixed 5 known bugs in existing code.

---

## Files Created

### New Package: `robot/sim/`

| File | Description |
|------|-------------|
| `robot/sim/__init__.py` | Package init; exports MuJoCoSim, DHRobotSim, PinocchioRobotSim, TaskEnv, DualArmSim |
| `robot/sim/mujoco_sim.py` | MuJoCo physics simulation environment class (FR-1). Wraps MjModel/MjData with methods for simulation control, joint manipulation, body pose queries, constraint-based grasping, viewer management, and offscreen rendering. Integrates logic from `utils/mj.py`. |
| `robot/sim/dh_robot_sim.py` | DH-parameter robot simulation base class (FR-2). Inherits `robot.robot.Robot`, adds optional MuJoCo simulation and Meshcat web visualization. Implements `servoJ` with MuJoCo/Meshcat sync. |
| `robot/sim/pinocchio_robot_sim.py` | Pinocchio robot simulation base class (FR-3). Inherits `robot.robot_pinocchio.Robot`, adds optional MuJoCo simulation. Overrides `servoJ` to sync MuJoCo after parent's collision check. |
| `robot/sim/task_env.py` | Manipulation task environment (FR-4). Combines robot simulator with MuJoCo scene for object management, grasp/release operations, and observation generation. |
| `robot/sim/dual_arm_sim.py` | Dual-arm simulation wrapper (FR-5). Manages two robots in a single MuJoCo scene with synchronized stepping and concurrent MoveJ. |

---

## Files Modified

### Bug Fixes

| File | Bug | Fix |
|------|-----|-----|
| `robot/iiwa14.py` | BUG-1: `self.robot = rtb.DHRobot(links)` was inside the for loop | Dedented to execute after loop completes. Now `IIWA14().robot.n == 7`. |
| `robot/robot.py` | BUG-2: `__setstate__` used hardcoded `range(6)` | Changed to `range(self._dof)` so 7-DOF robots serialize/deserialize correctly. |
| `robot/robot.py` | BUG-3: `disable_tool` set `self._tool = np.zeros(3)` | Changed to `self._tool = SE3()` to match `set_tool`'s type and prevent `AttributeError` on `.inv()`. |
| `robot/diana.py` | BUG-4: `move_cartesian_with_avoidance` called non-existent `ikine_with_avoidance` | Removed the method. Diana uses TracIK which handles singularity avoidance internally. |
| `robot/diana.py` | BUG-5: Hardcoded `/home/sun/...` URDF path | Made `urdf_path` a constructor parameter with default relative to project root: `assets/urdf/diana7_description/urdf/diana_v2.urdf`. |

---

## Key Design Decisions

1. **Inheritance over composition for RobotSim classes**: `DHRobotSim` inherits from `robot.robot.Robot` and `PinocchioRobotSim` inherits from `robot.robot_pinocchio.Robot`. This ensures all existing kinematics, dynamics, and motion methods are available without reimplementation.

2. **Only `servoJ` is overridden**: The key integration point is `servoJ`, which is the lowest-level joint control method called by `MoveJ` and `MoveL`. By overriding only `servoJ`, the entire motion pipeline automatically gets MuJoCo/Meshcat sync.

3. **MuJoCo and Meshcat are optional**: All new code uses `try/except ImportError` for optional dependencies. The system works without mujoco or pinocchio installed, with clear warning messages.

4. **MuJoCoSim is standalone**: The `MuJoCoSim` class does not depend on any Robot class. It can be used independently for scene manipulation or combined with any robot simulator.

5. **TaskEnv uses composition**: `TaskEnv` holds references to both a robot simulator and a `MuJoCoSim` instance, combining them for manipulation tasks.

6. **servoJ sync pattern**: For both DHRobotSim and PinocchioRobotSim, the sync pattern is:
   - DHRobotSim: check limits -> update self.q0 -> sync MuJoCo -> step -> sync Meshcat
   - PinocchioRobotSim: call parent servoJ (collision check + limits + Meshcat) -> if success, sync MuJoCo -> step

---

## Known Limitations

1. **Runtime object spawning**: MuJoCo does not support adding bodies/geoms at runtime via the standard API. `TaskEnv.spawn_object` works with pre-defined objects in the XML scene. For dynamic scenes, objects must be pre-placed in the XML and repositioned.

2. **DHRobotSim Meshcat**: The Meshcat visualization for DH-based robots is basic (displays end-effector frame). Full robot model rendering in Meshcat requires either a URDF/MJCF model or a custom geometry builder, which is not yet implemented.

3. **mujoco_set_joint assumes sequential joints**: The `mujoco_set_joint` method in DHRobotSim and PinocchioRobotSim uses joint index offset (skipping world joint). This works for standard single-robot scenes but may need refinement for complex multi-body scenes.

4. **No collision detection in DHRobotSim**: Unlike PinocchioRobotSim (which inherits collision checking from the parent), DHRobotSim does not have collision detection. This is inherent to the DH/rtb approach which lacks collision models.

5. **Environment compatibility**: The existing `roboticstoolbox` package has a numpy 2.x compatibility issue (pre-existing, not caused by this refactoring). This affects imports of `robot.robot`, `robot.ur5e`, `robot.iiwa14`, and `robot.diana`.

---

## Architecture

```
robot/sim/
├── __init__.py              # Package exports
├── mujoco_sim.py            # Standalone MuJoCo wrapper (no Robot dependency)
├── dh_robot_sim.py          # DHRobotSim(Robot) + MuJoCo + Meshcat
├── pinocchio_robot_sim.py   # PinocchioRobotSim(PinocchioRobot) + MuJoCo
├── task_env.py              # TaskEnv(robot, sim) for manipulation
└── dual_arm_sim.py          # DualArmSim(left, right, sim) for dual-arm
```

Dependency graph:
```
MuJoCoSim (standalone)
    ^
    |
DHRobotSim -----> robot.robot.Robot
    |
PinocchioRobotSim -----> robot.robot_pinocchio.Robot
    |
TaskEnv -----> DHRobotSim | PinocchioRobotSim + MuJoCoSim
    |
DualArmSim -----> DHRobotSim | PinocchioRobotSim + MuJoCoSim
```

No circular dependencies exist. `robot/sim/` imports from `robot/robot.py` and `robot/robot_pinocchio.py`, but neither of those imports from `robot/sim/`.
