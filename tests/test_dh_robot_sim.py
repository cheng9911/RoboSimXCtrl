"""
Tests for DHRobotSim class.

Covers: T2.1 (DH+MuJoCo init), T2.2 (servoJ sync), T2.4 (fkine/ikine round-trip),
T2.5 (dynamics methods), T2.6 (no-op without MuJoCo), T5.3 (IIWA14 BUG-1).
"""

import os
import sys
import numpy as np
import pytest
from spatialmath import SE3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import roboticstoolbox as rtb
from robot.sim.mujoco_sim import MuJoCoSim
from robot.sim.dh_robot_sim import DHRobotSim
from robot import Robot

SCENE_XML = os.path.join(os.path.dirname(__file__), "minimal_scene.xml")


# ========================================================================
# Helper: Concrete DHRobotSim subclass for testing
# ========================================================================

class TestRobot6DOF(DHRobotSim):
    """A 6-DOF DHRobotSim subclass for testing (UR5e-like DH parameters)."""

    def __init__(self, mujoco_xml_path=None, meshcat_port=None):
        super().__init__(mujoco_xml_path=mujoco_xml_path, meshcat_port=meshcat_port)

        self._dof = 6
        self.q0 = [0.0] * 6

        d1, d4, d5, d6 = 0.163, 0.134, 0.1, 0.1
        a3, a4 = 0.425, 0.392

        self.alpha_array = [0.0, -np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2]
        self.a_array = [0.0, 0.0, a3, a4, 0.0, 0.0]
        self.d_array = [d1, 0.0, 0.0, d4, d5, d6]
        self.theta_array = [0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0, 0.0]
        self.sigma_array = [0, 0, 0, 0, 0, 0]

        # Mass properties (simplified)
        masses = [3.7, 8.4, 2.3, 1.2, 1.2, 0.2]
        inertias = [np.diag([0.01, 0.01, 0.007]),
                     np.diag([0.015, 0.134, 0.134]),
                     np.diag([0.004, 0.031, 0.031]),
                     np.diag([0.003, 0.002, 0.003]),
                     np.diag([0.003, 0.003, 0.002]),
                     np.diag([1e-4, 1e-4, 1.3e-4])]
        coms = [np.zeros(3), np.array([0.21, 0, 0.14]), np.array([0.2, 0, 0.007]),
                np.zeros(3), np.zeros(3), np.array([0, 0, -0.023])]

        links = []
        for i in range(self._dof):
            links.append(rtb.DHLink(
                d=self.d_array[i],
                alpha=self.alpha_array[i],
                a=self.a_array[i],
                offset=self.theta_array[i],
                mdh=True,
                m=masses[i], r=coms[i], I=inertias[i]))
        self.robot = rtb.DHRobot(links)

        # Position limits for servoJ checks
        self._q_lim_low = np.array([-2.0] * 6)
        self._q_lim_up = np.array([2.0] * 6)

    def ikine(self, Tep: SE3) -> np.ndarray:
        """Simple IK using rtb built-in."""
        sol = self.robot.ikine_LM(Tep)
        if sol.success:
            return sol.q
        return np.array([])


class TestRobot7DOF(DHRobotSim):
    """A 7-DOF DHRobotSim subclass for testing (IIWA14-like)."""

    def __init__(self, mujoco_xml_path=None, meshcat_port=None):
        super().__init__(mujoco_xml_path=mujoco_xml_path, meshcat_port=meshcat_port)

        self._dof = 7
        self.q0 = [0.0] * 7

        self.alpha_array = [0.0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi/2]
        self.a_array = [0.0] * 7
        self.d_array = [0.36, 0.0, 0.42, 0.0, 0.4, 0.0, 0.126]
        self.theta_array = [0.0] * 7
        self.sigma_array = [0] * 7

        masses = [5.76, 6.35, 3.5, 3.5, 3.5, 1.8, 1.2]
        inertias = [np.diag([0.033, 0.033, 0.012]),
                     np.diag([0.030, 0.030, 0.011]),
                     np.diag([0.025, 0.024, 0.008]),
                     np.diag([0.017, 0.016, 0.006]),
                     np.diag([0.010, 0.009, 0.004]),
                     np.diag([0.005, 0.005, 0.004]),
                     np.diag([0.001, 0.001, 0.001])]

        links = []
        for i in range(self._dof):
            links.append(rtb.DHLink(
                d=self.d_array[i],
                alpha=self.alpha_array[i],
                a=self.a_array[i],
                offset=self.theta_array[i],
                mdh=True,
                m=masses[i], r=np.zeros(3), I=inertias[i]))
        self.robot = rtb.DHRobot(links)

        self._q_lim_low = np.array([-3.0] * 7)
        self._q_lim_up = np.array([3.0] * 7)

    def ikine(self, Tep: SE3) -> np.ndarray:
        sol = self.robot.ikine_LM(Tep)
        if sol.success:
            return sol.q
        return np.array([])


# ========================================================================
# T2.1: DH + MuJoCo Initialization
# ========================================================================

class TestT2_1_DHMujocoInit:
    """T2.1: Verify DHRobotSim initializes both DH model and MuJoCo."""

    def test_mujoco_initialized_with_xml(self):
        robot = TestRobot6DOF(mujoco_xml_path=SCENE_XML)
        assert robot.mj_sim is not None
        assert robot.mj_sim.model is not None

    def test_dh_robot_not_none(self):
        robot = TestRobot6DOF()
        assert robot.robot is not None
        assert isinstance(robot.robot, rtb.DHRobot)

    def test_fkine_returns_se3(self):
        robot = TestRobot6DOF()
        T = robot.fkine([0] * 6)
        assert isinstance(T, SE3)

    def test_dof_is_correct(self):
        robot = TestRobot6DOF()
        assert robot.dof == 6

    def test_dh_params_set(self):
        robot = TestRobot6DOF()
        assert len(robot.alpha_array) == 6
        assert len(robot.a_array) == 6
        assert len(robot.d_array) == 6
        assert len(robot.theta_array) == 6


# ========================================================================
# T2.2: servoJ State Synchronization
# ========================================================================

class TestT2_2_ServoJSync:
    """T2.2: Verify servoJ updates q0 and MuJoCo state."""

    def test_servo_updates_q0(self):
        """servoJ with a large enough delta_t passes velocity check."""
        robot = TestRobot6DOF()
        target = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        # Use delta_t=1.0 so velocity = 0.6/1.0 = 0.6 rad/s < 3.0
        result = robot.servoJ(target, delta_t=1.0)
        assert result == 0
        np.testing.assert_allclose(robot.q0, target, atol=1e-10)

    def test_servo_returns_zero_on_success(self):
        robot = TestRobot6DOF()
        result = robot.servoJ([0.0] * 6, delta_t=1.0)
        assert result == 0

    def test_servo_with_mujoco_sync(self):
        robot = TestRobot6DOF(mujoco_xml_path=SCENE_XML)
        target = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
        result = robot.servoJ(target, delta_t=1.0)
        assert result == 0
        np.testing.assert_allclose(robot.q0, target, atol=1e-10)

    def test_servo_velocity_limit_exceeded(self):
        """Large position jump in small dt should fail velocity check."""
        robot = TestRobot6DOF()
        robot.q0 = [0.0] * 6
        # Jump of 1.0 rad in 0.001s = 1000 rad/s >> 3.0 rad/s
        result = robot.servoJ([1.0] * 6, delta_t=0.001)
        assert result == -1

    def test_servo_position_limit_exceeded(self):
        """Target beyond position limits should fail."""
        robot = TestRobot6DOF()
        result = robot.servoJ([5.0] * 6, delta_t=1.0)
        assert result == -1

    def test_servo_small_move_succeeds(self):
        """Small moves within limits should succeed."""
        robot = TestRobot6DOF()
        robot.q0 = [0.0] * 6
        # Move 0.01 rad in 1.0s = 0.01 rad/s < 3.0
        target = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        result = robot.servoJ(target, delta_t=1.0)
        assert result == 0


# ========================================================================
# T2.4: fkine/ikine Round-Trip
# ========================================================================

class TestT2_4_FKineIKineRoundTrip:
    """T2.4: Verify fkine(ikine(T)) returns approximately T."""

    def test_fkine_ikine_roundtrip(self):
        robot = TestRobot6DOF()
        q_orig = [0.3, -0.5, 0.8, 0.1, -0.2, 0.4]
        T = robot.fkine(q_orig)
        q_solved = robot.ikine(T)
        assert len(q_solved) > 0, "ikine returned empty"
        T_check = robot.fkine(q_solved)
        err = np.linalg.norm(T_check.t - T.t)
        assert err < 0.01, f"Position error {err} > 10mm"

    def test_fkine_home_position(self):
        robot = TestRobot6DOF()
        T = robot.fkine([0] * 6)
        assert isinstance(T, SE3)

    def test_fkine_varies_with_config(self):
        robot = TestRobot6DOF()
        T0 = robot.fkine([0] * 6)
        T1 = robot.fkine([0.5, 0, 0, 0, 0, 0])
        assert np.linalg.norm(T1.t - T0.t) > 0.01


# ========================================================================
# T2.5: Dynamics Methods
# ========================================================================

class TestT2_5_DynamicsMethods:
    """T2.5: Verify dynamics methods return correct shapes."""

    def test_get_inertia_shape(self):
        robot = TestRobot6DOF()
        q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        M = robot.get_inertia(q)
        assert M.shape == (6, 6)

    def test_get_inertia_positive_definite(self):
        robot = TestRobot6DOF()
        q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        M = robot.get_inertia(q)
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues > 0), "Inertia matrix not positive definite"

    def test_get_inertia_symmetric(self):
        robot = TestRobot6DOF()
        M = robot.get_inertia([0.1] * 6)
        np.testing.assert_allclose(M, M.T, atol=1e-10)

    def test_get_coriolis_shape(self):
        robot = TestRobot6DOF()
        C = robot.get_coriolis([0.1]*6, [0.01]*6)
        assert C.shape == (6, 6)

    def test_get_gravity_shape(self):
        robot = TestRobot6DOF()
        g = robot.get_gravity([0.1] * 6)
        assert g.shape == (6,)

    def test_get_gravity_nonzero(self):
        """Gravity torques should be nonzero for non-horizontal configs."""
        robot = TestRobot6DOF()
        g = robot.get_gravity([0.0] * 6)
        assert np.linalg.norm(g) > 0

    def test_inv_dynamics_shape(self):
        robot = TestRobot6DOF()
        tau = robot.inv_dynamics([0.1]*6, [0.01]*6, [0.001]*6)
        assert tau.shape == (6,)


# ========================================================================
# T2.6: MuJoCo Operations Without MuJoCo Enabled
# ========================================================================

class TestT2_6_NoOpWithoutMujoco:
    """T2.6: Verify all MuJoCo methods are no-ops when mj_sim is None."""

    def test_mj_sim_is_none_without_xml(self):
        robot = TestRobot6DOF()
        assert robot.mj_sim is None

    def test_mujoco_step_no_raise(self):
        robot = TestRobot6DOF()
        robot.mujoco_step()

    def test_mujoco_set_joint_no_raise(self):
        robot = TestRobot6DOF()
        robot.mujoco_set_joint(np.zeros(6))

    def test_mujoco_get_joint_returns_empty(self):
        robot = TestRobot6DOF()
        q = robot.mujoco_get_joint()
        assert len(q) == 0

    def test_mujoco_get_body_pose_returns_identity(self):
        robot = TestRobot6DOF()
        pose = robot.mujoco_get_body_pose("anything")
        assert isinstance(pose, SE3)
        np.testing.assert_allclose(pose.A, np.eye(4), atol=1e-10)


# ========================================================================
# DHRobotSim with MuJoCo Integration
# ========================================================================

class TestDHRobotSimMujocoIntegration:
    """Integration tests for DHRobotSim + MuJoCo."""

    def test_mujoco_step_advances_time(self):
        robot = TestRobot6DOF(mujoco_xml_path=SCENE_XML)
        t0 = robot.mj_sim.data.time
        robot.mujoco_step()
        t1 = robot.mj_sim.data.time
        assert t1 > t0

    def test_mujoco_get_body_pose(self):
        robot = TestRobot6DOF(mujoco_xml_path=SCENE_XML)
        pose = robot.mujoco_get_body_pose("link1")
        assert isinstance(pose, SE3)

    def test_mujoco_set_and_get_joint(self):
        """Set MuJoCo hinge joints and read them back."""
        robot = TestRobot6DOF(mujoco_xml_path=SCENE_XML)
        # Scene has joints: box_free(0), joint_1(1), joint_2(2)
        # mujoco_set_joint skips index 0 (box_free), sets hinge joints
        # With 6 DOF target, only first 2 hinge joints are set
        target = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
        robot.mujoco_set_joint(target)
        q = robot.mujoco_get_joint()
        # Only joint_1 and joint_2 exist as hinge joints (indices 1,2 out of 3 joints)
        # mujoco_get_joint returns njnt-1 = 2 elements
        assert len(q) >= 2
        np.testing.assert_allclose(q[0], 0.1, atol=1e-4)
        np.testing.assert_allclose(q[1], 0.2, atol=1e-4)

    def test_mujoco_none_returns_defaults(self):
        robot = TestRobot6DOF()
        assert len(robot.mujoco_get_joint()) == 0
        assert isinstance(robot.mujoco_get_body_pose("x"), SE3)


# ========================================================================
# T5.3: 7-DOF DHRobotSim (IIWA14 BUG-1 verification)
# ========================================================================

class TestT5_3_IIWA14Bug1:
    """T5.3: Verify 7-DOF DH robot has correct number of links."""

    def test_7dof_dof(self):
        robot = TestRobot7DOF()
        assert robot.dof == 7

    def test_7dof_robot_n(self):
        robot = TestRobot7DOF()
        assert robot.robot.n == 7

    def test_7dof_fkine(self):
        robot = TestRobot7DOF()
        T = robot.fkine([0] * 7)
        assert isinstance(T, SE3)

    def test_7dof_dynamics(self):
        robot = TestRobot7DOF()
        M = robot.get_inertia([0.1] * 7)
        assert M.shape == (7, 7)
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues > 0), "7-DOF inertia not positive definite"

    def test_7dof_servoj(self):
        robot = TestRobot7DOF()
        target = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        result = robot.servoJ(target, delta_t=1.0)
        assert result == 0
        np.testing.assert_allclose(robot.q0, target, atol=1e-10)
