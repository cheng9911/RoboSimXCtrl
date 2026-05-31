"""
Backward compatibility tests.

Covers: T5.1 (import Robot), T5.2 (import UR5e), T5.3 (IIWA14 BUG-1),
T5.4 (Diana BUG-5), T5.5 (pickle round-trip), T5.6 (disable_tool).

Note: Imports go through conftest.py which patches the circular import
caused by ur5e.py using bare imports.
"""

import os
import sys
import pickle
import numpy as np
import pytest
from spatialmath import SE3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ========================================================================
# T5.1: Import robot.Robot Still Works
# ========================================================================

class TestT5_1_RobotImport:
    """T5.1: Verify existing import paths still work."""

    def test_import_robot(self):
        from robot import Robot
        assert Robot is not None

    def test_robot_is_class(self):
        from robot import Robot
        assert isinstance(Robot, type)

    def test_robot_has_fkine(self):
        from robot import Robot
        assert hasattr(Robot, 'fkine')

    def test_robot_has_ikine(self):
        from robot import Robot
        assert hasattr(Robot, 'ikine')

    def test_robot_has_dynamics(self):
        from robot import Robot
        assert hasattr(Robot, 'get_inertia')
        assert hasattr(Robot, 'get_coriolis')
        assert hasattr(Robot, 'get_gravity')
        assert hasattr(Robot, 'inv_dynamics')


# ========================================================================
# T5.2: Import UR5e Still Works
# ========================================================================

class TestT5_2_UR5eImport:
    """T5.2: Verify UR5e can be imported and instantiated."""

    def test_import_ur5e(self):
        from robot import UR5e
        assert UR5e is not None

    def test_ur5e_instantiation(self):
        from robot import UR5e
        ur = UR5e()
        assert ur is not None

    def test_ur5e_dof(self):
        from robot import UR5e
        ur = UR5e()
        assert ur.dof == 6

    def test_ur5e_fkine(self):
        from robot import UR5e
        ur = UR5e()
        T = ur.fkine([0] * 6)
        assert isinstance(T, SE3)

    def test_ur5e_ikine(self):
        from robot import UR5e
        ur = UR5e()
        # Use a small perturbation from home to ensure config matches
        q_orig = [0.1, -0.2, 0.3, 0.1, -0.1, 0.1]
        ur.set_robot_config(q_orig)
        T = ur.fkine(q_orig)
        q = ur.ikine(T)
        assert len(q) > 0, "UR5e ikine returned empty array"

    def test_ur5e_is_robot(self):
        from robot import Robot, UR5e
        ur = UR5e()
        assert isinstance(ur, Robot)


# ========================================================================
# T5.3: IIWA14 BUG-1 Fix Verified
# ========================================================================

class TestT5_3_IIWA14Bug1:
    """T5.3: Verify IIWA14 creates a 7-link robot (BUG-1 fix)."""

    def test_import_iiwa14(self):
        from robot import IIWA14
        assert IIWA14 is not None

    def test_iiwa14_instantiation(self):
        from robot import IIWA14
        r = IIWA14()
        assert r is not None

    def test_iiwa14_dof_is_7(self):
        from robot import IIWA14
        r = IIWA14()
        assert r.dof == 7

    def test_iiwa14_robot_n_is_7(self):
        """BUG-1: robot.n should be 7, not 1."""
        from robot import IIWA14
        r = IIWA14()
        assert r.robot.n == 7

    def test_iiwa14_fkine(self):
        from robot import IIWA14
        r = IIWA14()
        T = r.fkine([0] * 7)
        assert isinstance(T, SE3)

    def test_iiwa14_ikine(self):
        from robot import IIWA14
        r = IIWA14()
        T = r.fkine([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        q = r.ikine(T)
        assert len(q) == 7


# ========================================================================
# T5.4: Diana BUG-5 Fix (requires tracikpy)
# ========================================================================

_has_tracikpy = False
try:
    import tracikpy
    _has_tracikpy = True
except ImportError:
    pass

_has_diana_urdf = os.path.exists(
    os.path.join(PROJECT_ROOT, "assets", "urdf", "diana7_description", "urdf", "diana_v2.urdf"))


@pytest.mark.skipif(not _has_tracikpy, reason="tracikpy not installed")
@pytest.mark.skipif(not _has_diana_urdf, reason="Diana URDF not found")
class TestT5_4_DianaBug5:
    """T5.4: Verify Diana uses relative path, not hardcoded absolute path."""

    def test_import_diana(self):
        from robot import Diana
        assert Diana is not None

    def test_diana_instantiation(self):
        from robot import Diana
        r = Diana()
        assert r is not None

    def test_diana_ik_solver_initialized(self):
        from robot import Diana
        r = Diana()
        assert r.ik_solver is not None

    def test_diana_no_hardcoded_path_in_source(self):
        """Verify no hardcoded /home/sun/... path in diana.py."""
        diana_py = os.path.join(PROJECT_ROOT, "robot", "diana.py")
        with open(diana_py, "r") as f:
            content = f.read()
        assert "/home/sun" not in content, "Hardcoded path found in diana.py"

    def test_diana_dof(self):
        from robot import Diana
        r = Diana()
        assert r.dof == 7


# ========================================================================
# T5.5: Robot Pickle Round-Trip (BUG-2)
# ========================================================================

class TestT5_5_PickleRoundTrip:
    """T5.5: Verify 7-DOF robots survive pickle serialization."""

    def test_ur5e_pickle_roundtrip(self):
        from robot import UR5e
        ur = UR5e()
        data = pickle.dumps(ur)
        ur2 = pickle.loads(data)
        assert ur2.dof == 6
        T1 = ur.fkine([0] * 6)
        T2 = ur2.fkine([0] * 6)
        np.testing.assert_allclose(T1.A, T2.A, atol=1e-10)

    def test_iiwa14_pickle_roundtrip(self):
        """BUG-2: 7-DOF robot survives pickle (range(6) -> range(self._dof))."""
        from robot import IIWA14
        r = IIWA14()
        data = pickle.dumps(r)
        r2 = pickle.loads(data)
        assert r2.dof == 7, "BUG-2 not fixed: dof changed after pickle"
        T1 = r.fkine([0] * 7)
        T2 = r2.fkine([0] * 7)
        np.testing.assert_allclose(T1.A, T2.A, atol=1e-10)

    def test_iiwa14_pickle_preserves_q0(self):
        from robot import IIWA14
        r = IIWA14()
        r.q0 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        data = pickle.dumps(r)
        r2 = pickle.loads(data)
        np.testing.assert_allclose(r2.q0, r.q0, atol=1e-10)


# ========================================================================
# T5.6: Robot disable_tool (BUG-3)
# ========================================================================

class TestT5_6_DisableTool:
    """T5.6: Verify disable_tool sets _tool to SE3, not numpy array."""

    def test_set_tool_returns_se3(self):
        from robot import UR5e
        ur = UR5e()
        ur.set_tool(np.array([0, 0, 0.1]))
        assert isinstance(ur._tool, SE3)

    def test_disable_tool_returns_se3(self):
        """BUG-3: _tool must be SE3 after disable_tool, not numpy."""
        from robot import UR5e
        ur = UR5e()
        ur.set_tool(np.array([0, 0, 0.1]))
        ur.disable_tool()
        assert isinstance(ur._tool, SE3), "BUG-3: _tool is not SE3 after disable_tool"

    def test_fkine_after_disable_tool(self):
        """fkine should work without AttributeError after disable_tool."""
        from robot import UR5e
        ur = UR5e()
        ur.set_tool(np.array([0, 0, 0.1]))
        ur.disable_tool()
        T = ur.fkine([0] * 6)
        assert isinstance(T, SE3)

    def test_set_base_returns_se3(self):
        from robot import UR5e
        ur = UR5e()
        ur.set_base(np.array([0, 0, 0.5]))
        assert isinstance(ur._base, SE3)

    def test_disable_base_returns_se3(self):
        from robot import UR5e
        ur = UR5e()
        ur.set_base(np.array([0, 0, 0.5]))
        ur.disable_base()
        assert isinstance(ur._base, SE3)


# ========================================================================
# Robot Base API Completeness
# ========================================================================

class TestRobotBaseAPI:
    """Verify Robot has all expected methods from the API."""

    def test_robot_has_expected_methods(self):
        from robot import Robot
        expected_methods = [
            'fkine', 'ikine',
            'get_inertia', 'get_coriolis', 'get_gravity', 'inv_dynamics',
            'set_tool', 'disable_tool', 'set_base', 'disable_base',
            'set_joint', 'get_joint', 'move_joint', 'move_cartesian',
            'get_cartesian',
        ]
        for method in expected_methods:
            assert hasattr(Robot, method), f"Robot missing method: {method}"

    def test_ur5e_has_expected_attributes(self):
        from robot import UR5e
        ur = UR5e()
        assert hasattr(ur, 'robot')
        assert hasattr(ur, 'q0')
        assert hasattr(ur, '_dof')
        assert hasattr(ur, 'alpha_array')
        assert hasattr(ur, 'a_array')
        assert hasattr(ur, 'd_array')
        assert hasattr(ur, 'theta_array')

    def test_ur5e_dh_parameters_set(self):
        """Verify DH parameters are set for UR5e."""
        from robot import UR5e
        ur = UR5e()
        assert len(ur.alpha_array) == 6
        assert len(ur.a_array) == 6
        assert len(ur.d_array) == 6
        assert len(ur.theta_array) == 6


# ========================================================================
# DHRobotSim Import Tests
# ========================================================================

class TestDHRobotSimImport:
    """Test importing DHRobotSim from robot.sim."""

    def test_import_dh_robot_sim(self):
        from robot.sim.dh_robot_sim import DHRobotSim
        assert DHRobotSim is not None

    def test_dh_robot_sim_inherits_robot(self):
        from robot.sim.dh_robot_sim import DHRobotSim
        from robot import Robot
        assert issubclass(DHRobotSim, Robot)

    def test_dh_robot_sim_has_mujoco_methods(self):
        from robot.sim.dh_robot_sim import DHRobotSim
        assert hasattr(DHRobotSim, 'mujoco_step')
        assert hasattr(DHRobotSim, 'mujoco_set_joint')
        assert hasattr(DHRobotSim, 'mujoco_get_joint')
        assert hasattr(DHRobotSim, 'mujoco_get_body_pose')
        assert hasattr(DHRobotSim, 'servoJ')


# ========================================================================
# MuJoCoSim Import Tests
# ========================================================================

class TestMuJoCoSimImport:
    """Test importing MuJoCoSim from robot.sim."""

    def test_import_mujoco_sim(self):
        from robot.sim.mujoco_sim import MuJoCoSim
        assert MuJoCoSim is not None

    def test_import_has_mujoco_flag(self):
        from robot.sim.mujoco_sim import _HAS_MUJOCO
        assert isinstance(_HAS_MUJOCO, bool)
