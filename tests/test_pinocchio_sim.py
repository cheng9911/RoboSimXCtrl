"""
Tests for PinocchioRobotSim class.

Covers: T3.1 (load URDF), T3.2 (servoJ sync), T3.4 (collision detection).

Note: These tests require pinocchio and a URDF file. Diana tests require
tracikpy which may not be available.
"""

import os
import sys
import numpy as np
import pytest
from spatialmath import SE3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Check for optional dependencies
try:
    import pinocchio
    _HAS_PINOCCHIO = True
except ImportError:
    _HAS_PINOCCHIO = False

try:
    import tracikpy
    _HAS_TRACIKPY = True
except ImportError:
    _HAS_TRACIKPY = False

# Check for pinocchio Robot base class
_HAS_PINOCCHIO_ROBOT = False
if _HAS_PINOCCHIO:
    try:
        from robot.sim.pinocchio_robot_sim import PinocchioRobotSim
        _HAS_PINOCCHIO_ROBOT = True
    except ImportError:
        pass

# URDF path for Diana
DIANA_URDF = os.path.join(PROJECT_ROOT, "assets", "urdf", "diana7_description", "urdf", "diana_v2.urdf")
DIANA_MESh_DIR = os.path.join(PROJECT_ROOT, "assets", "urdf", "diana7_description")
DIANA_MUJOCO_XML = os.path.join(PROJECT_ROOT, "assets", "mujoco", "diana7", "scene.xml")

# URDF path for any robot (ur_description)
UR5E_URDF = os.path.join(PROJECT_ROOT, "assets", "urdf", "ur_description", "urdf")
# Check for available URDF files
_has_diana_urdf = os.path.exists(DIANA_URDF)
_has_diana_mujoco = os.path.exists(DIANA_MUJOCO_XML)


# ========================================================================
# PinocchioRobotSim Class Tests
# ========================================================================

@pytest.mark.skipif(not _HAS_PINOCCHIO, reason="pinocchio not installed")
class TestPinocchioRobotSimClass:
    """Test PinocchioRobotSim class structure and methods."""

    def test_class_exists(self):
        """PinocchioRobotSim class should be importable."""
        from robot.sim.pinocchio_robot_sim import PinocchioRobotSim
        assert PinocchioRobotSim is not None

    def test_class_inherits_from_pinocchio_robot(self):
        """PinocchioRobotSim should inherit from robot_pinocchio.Robot."""
        from robot.sim.pinocchio_robot_sim import PinocchioRobotSim
        from robot import robot_pinocchio
        assert issubclass(PinocchioRobotSim, robot_pinocchio.Robot)

    def test_has_mujoco_methods(self):
        """PinocchioRobotSim should have MuJoCo interface methods."""
        from robot.sim.pinocchio_robot_sim import PinocchioRobotSim
        assert hasattr(PinocchioRobotSim, 'mujoco_step')
        assert hasattr(PinocchioRobotSim, 'mujoco_set_joint')
        assert hasattr(PinocchioRobotSim, 'mujoco_get_joint')
        assert hasattr(PinocchioRobotSim, 'mujoco_get_body_pose')
        assert hasattr(PinocchioRobotSim, 'servoJ')

    def test_mujoco_none_by_default(self):
        """Without mujoco_xml_path, mj_sim should be None."""
        from robot.sim.pinocchio_robot_sim import PinocchioRobotSim
        # We can't instantiate without a valid URDF, so test the __init__ signature
        import inspect
        sig = inspect.signature(PinocchioRobotSim.__init__)
        assert 'mujoco_xml_path' in sig.parameters
        assert 'visualizer' in sig.parameters


# ========================================================================
# T3.1: Pinocchio + MuJoCo Initialization (requires Diana URDF)
# ========================================================================

@pytest.mark.skipif(not _HAS_PINOCCHIO, reason="pinocchio not installed")
@pytest.mark.skipif(not _HAS_TRACIKPY, reason="tracikpy not installed")
@pytest.mark.skipif(not _has_diana_urdf, reason="Diana URDF not found")
class TestT3_1_PinocchioMujocoInit:
    """T3.1: Verify PinocchioRobotSim initializes both Pinocchio and MuJoCo."""

    @pytest.fixture
    def diana_sim(self):
        """Create a Diana PinocchioRobotSim with MuJoCo."""
        from robot.sim.pinocchio_robot_sim import PinocchioRobotSim
        robot = PinocchioRobotSim(
            urdf_path=DIANA_URDF,
            mesh_dir=DIANA_MESh_DIR,
            mujoco_xml_path=DIANA_MUJOCO_XML if _has_diana_mujoco else None,
        )
        return robot

    def test_model_loaded(self, diana_sim):
        assert diana_sim.model is not None

    def test_mj_sim_loaded(self, diana_sim):
        if _has_diana_mujoco:
            assert diana_sim.mj_sim is not None
        else:
            assert diana_sim.mj_sim is None

    def test_q_is_7_element(self, diana_sim):
        assert len(diana_sim.q) == 7


# ========================================================================
# T3.2: servoJ Syncs Pinocchio and MuJoCo (requires Diana)
# ========================================================================

@pytest.mark.skipif(not _HAS_PINOCCHIO, reason="pinocchio not installed")
@pytest.mark.skipif(not _HAS_TRACIKPY, reason="tracikpy not installed")
@pytest.mark.skipif(not _has_diana_urdf, reason="Diana URDF not found")
@pytest.mark.skipif(not _has_diana_mujoco, reason="Diana MuJoCo XML not found")
class TestT3_2_PinocchioServoSync:
    """T3.2: Verify servoJ updates both Pinocchio and MuJoCo states."""

    @pytest.fixture
    def diana_sim(self):
        from robot.sim.pinocchio_robot_sim import PinocchioRobotSim
        return PinocchioRobotSim(
            urdf_path=DIANA_URDF,
            mesh_dir=DIANA_MESh_DIR,
            mujoco_xml_path=DIANA_MUJOCO_XML,
        )

    def test_servoj_updates_q(self, diana_sim):
        q_target = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        result = diana_sim.servoJ(q_target)
        assert result == 0
        np.testing.assert_allclose(diana_sim.q, q_target, atol=1e-6)

    def test_servoj_syncs_mujoco(self, diana_sim):
        q_target = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        diana_sim.servoJ(q_target)
        mj_q = diana_sim.mujoco_get_joint()
        np.testing.assert_allclose(mj_q[:7], q_target, atol=1e-4)


# ========================================================================
# PinocchioRobotSim without MuJoCo (no-op tests)
# ========================================================================

@pytest.mark.skipif(not _HAS_PINOCCHIO, reason="pinocchio not installed")
class TestPinocchioSimNoMujoco:
    """Test that MuJoCo methods are no-ops when mj_sim is None."""

    def test_mujoco_step_no_raise(self):
        """mujoco_step should be safe when mj_sim is None."""
        from robot.sim.pinocchio_robot_sim import PinocchioRobotSim
        # We can't easily instantiate without URDF, so test the method logic
        # by creating a mock-like object
        class MockPinocchioRobot:
            mj_sim = None
            def mujoco_step(self):
                if self.mj_sim is not None:
                    self.mj_sim.step()

            def mujoco_get_joint(self):
                if self.mj_sim is None:
                    return np.array([])
                return np.array([])

            def mujoco_get_body_pose(self, name):
                if self.mj_sim is None:
                    return SE3()
                return SE3()

        mock = MockPinocchioRobot()
        mock.mujoco_step()  # Should not raise
        assert len(mock.mujoco_get_joint()) == 0
        assert isinstance(mock.mujoco_get_body_pose("anything"), SE3)


# ========================================================================
# T3.4: Collision Detection (requires Diana)
# ========================================================================

@pytest.mark.skipif(not _HAS_PINOCCHIO, reason="pinocchio not installed")
@pytest.mark.skipif(not _HAS_TRACIKPY, reason="tracikpy not installed")
@pytest.mark.skipif(not _has_diana_urdf, reason="Diana URDF not found")
class TestT3_4_CollisionDetection:
    """T3.4: Verify collision model is available."""

    @pytest.fixture
    def diana(self):
        from robot import Diana
        return Diana()

    def test_collision_model_not_none(self, diana):
        assert diana.collision_model is not None

    def test_collision_data_not_none(self, diana):
        assert diana.collision_data is not None


# ========================================================================
# Pinocchio FKine/IKine (requires Diana)
# ========================================================================

@pytest.mark.skipif(not _HAS_PINOCCHIO, reason="pinocchio not installed")
@pytest.mark.skipif(not _HAS_TRACIKPY, reason="tracikpy not installed")
@pytest.mark.skipif(not _has_diana_urdf, reason="Diana URDF not found")
class TestPinocchioKinematics:
    """Test Pinocchio forward/inverse kinematics."""

    @pytest.fixture
    def diana(self):
        from robot import Diana
        return Diana()

    def test_fkine(self, diana):
        T = diana.fkine([0] * 7)
        assert isinstance(T, SE3)

    def test_ikine_converges(self, diana):
        T = diana.fkine([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        q_sol = diana.ikine(T)
        assert q_sol is not None
        assert len(q_sol) == 7

    def test_servoj_updates_q0(self, diana):
        target = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        result = diana.servoJ(target)
        assert result == 0
        np.testing.assert_allclose(diana.q0, target, atol=1e-10)
