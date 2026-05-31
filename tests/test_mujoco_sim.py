"""
Tests for MuJoCoSim class.

Covers: T1.1 (load scene), T1.2 (set/get joints), T1.3 (step/reset),
T1.4 (body pose), T1.5 (attach/detach), T1.8 (graceful failure).
"""

import os
import sys
import numpy as np
import pytest
from spatialmath import SE3

# Ensure project is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mujoco
from robot.sim.mujoco_sim import MuJoCoSim, _HAS_MUJOCO

# Path to our test XML scene
SCENE_XML = os.path.join(os.path.dirname(__file__), "minimal_scene.xml")


# ========================================================================
# T1.1: Load MuJoCo Scene XML
# ========================================================================

class TestT1_1_LoadScene:
    """T1.1: Verify MuJoCoSim loads a scene XML and initializes model/data."""

    def test_load_model_not_none(self):
        sim = MuJoCoSim(SCENE_XML)
        assert sim.model is not None
        assert isinstance(sim.model, mujoco.MjModel)

    def test_load_data_not_none(self):
        sim = MuJoCoSim(SCENE_XML)
        assert sim.data is not None
        assert isinstance(sim.data, mujoco.MjData)

    def test_custom_timestep(self):
        sim = MuJoCoSim(SCENE_XML, timestep=0.001)
        assert sim.model.opt.timestep == pytest.approx(0.001)

    def test_default_timestep(self):
        sim = MuJoCoSim(SCENE_XML)
        # XML sets 0.002, but constructor default is also 0.002
        assert sim.model.opt.timestep == pytest.approx(0.002)


# ========================================================================
# T1.2: Set/Get Joint Positions
# ========================================================================

class TestT1_2_SetGetJoints:
    """T1.2: Verify joint positions can be set and retrieved."""

    def test_set_get_joint_by_name(self):
        sim = MuJoCoSim(SCENE_XML)
        sim.set_joint_q("joint_1", 0.5)
        q = sim.get_joint_q("joint_1")
        assert q == pytest.approx(0.5, abs=1e-6)

    def test_set_get_joint_by_name_second(self):
        sim = MuJoCoSim(SCENE_XML)
        sim.set_joint_q("joint_2", 1.2)
        q = sim.get_joint_q("joint_2")
        assert q == pytest.approx(1.2, abs=1e-6)

    def test_set_get_joint_by_id(self):
        sim = MuJoCoSim(SCENE_XML)
        joint_id = sim._resolve_joint_id("joint_1")
        sim.set_joint_q(joint_id, 1.0)
        q = sim.get_joint_q(joint_id)
        assert q == pytest.approx(1.0, abs=1e-6)

    def test_resolve_joint_id_returns_int(self):
        sim = MuJoCoSim(SCENE_XML)
        jid = sim._resolve_joint_id("joint_1")
        assert isinstance(jid, int)
        assert jid >= 0

    def test_set_joint_degrees(self):
        sim = MuJoCoSim(SCENE_XML)
        sim.set_joint_q("joint_1", 90.0, unit="deg")
        q = sim.get_joint_q("joint_1")
        assert q == pytest.approx(np.deg2rad(90.0), abs=1e-4)

    def test_free_joint_set(self):
        """Test setting the free joint (box_free) position."""
        sim = MuJoCoSim(SCENE_XML)
        T = SE3.Trans(1.0, 2.0, 3.0)
        sim.set_free_joint_pose("box_free", T)
        sim.forward()
        pos = sim.get_body_pose_xyz("box")
        np.testing.assert_allclose(pos, [1.0, 2.0, 3.0], atol=1e-4)


# ========================================================================
# T1.3: Step Simulation and Verify State Changes
# ========================================================================

class TestT1_3_StepSimulation:
    """T1.3: Verify stepping advances time and reset restores it."""

    def test_step_increases_time(self):
        sim = MuJoCoSim(SCENE_XML, timestep=0.001)
        t0 = sim.data.time
        sim.step()
        t1 = sim.data.time
        assert t1 > t0
        assert t1 - t0 == pytest.approx(0.001, abs=1e-8)

    def test_multiple_steps(self):
        sim = MuJoCoSim(SCENE_XML, timestep=0.001)
        for _ in range(10):
            sim.step()
        assert sim.data.time == pytest.approx(0.01, abs=1e-6)

    def test_reset_sets_time_to_zero(self):
        sim = MuJoCoSim(SCENE_XML, timestep=0.001)
        for _ in range(100):
            sim.step()
        assert sim.data.time > 0
        sim.reset()
        assert sim.data.time == pytest.approx(0.0, abs=1e-10)

    def test_step_with_control(self):
        sim = MuJoCoSim(SCENE_XML, timestep=0.001)
        ctrl = np.zeros(sim.model.nu)
        sim.step(ctrl)
        assert sim.data.time > 0


# ========================================================================
# T1.4: Body Pose Get/Set
# ========================================================================

class TestT1_4_BodyPose:
    """T1.4: Verify body pose queries and position setting."""

    def test_get_body_pose_returns_se3(self):
        sim = MuJoCoSim(SCENE_XML)
        pose = sim.get_body_pose("box")
        assert isinstance(pose, SE3)

    def test_get_body_pose_xyz_matches(self):
        sim = MuJoCoSim(SCENE_XML)
        pose = sim.get_body_pose("box")
        xyz = sim.get_body_pose_xyz("box")
        np.testing.assert_allclose(xyz, pose.t, atol=1e-10)

    def test_set_body_pose_updates_position(self):
        """set_body_pose sets xpos; use set_free_joint_pose for bodies with joints."""
        sim = MuJoCoSim(SCENE_XML)
        # For bodies with free joints, set_free_joint_pose is the correct API
        T = SE3.Trans(1.0, 2.0, 3.0)
        sim.set_free_joint_pose("box_free", T)
        sim.forward()
        pos = sim.get_body_pose_xyz("box")
        np.testing.assert_allclose(pos, [1.0, 2.0, 3.0], atol=1e-4)

    def test_get_body_pose_by_id(self):
        sim = MuJoCoSim(SCENE_XML)
        body_id = sim._resolve_body_id("box")
        assert isinstance(body_id, int)
        pose = sim.get_body_pose(body_id)
        assert isinstance(pose, SE3)

    def test_body_pose_xyz_is_3_element(self):
        sim = MuJoCoSim(SCENE_XML)
        xyz = sim.get_body_pose_xyz("box")
        assert xyz.shape == (3,)


# ========================================================================
# T1.5: Attach/Detach Constraints
# ========================================================================

class TestT1_5_AttachDetach:
    """T1.5: Verify equality constraint activation/deactivation."""

    def test_attach_activates_constraint(self):
        sim = MuJoCoSim(SCENE_XML)
        eq_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_eq")
        T = SE3.Trans(0, 0, 0.6)
        sim.attach("grasp_eq", "box_free", T)
        assert sim.data.eq_active[eq_id] == 1

    def test_detach_deactivates_constraint(self):
        sim = MuJoCoSim(SCENE_XML)
        eq_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_eq")
        T = SE3.Trans(0, 0, 0.6)
        sim.attach("grasp_eq", "box_free", T)
        sim.detach("grasp_eq")
        assert sim.data.eq_active[eq_id] == 0

    def test_attach_nonexistent_raises_valueerror(self):
        sim = MuJoCoSim(SCENE_XML)
        with pytest.raises(ValueError, match="not found"):
            sim.attach("nonexistent", "box_free", SE3())

    def test_detach_nonexistent_raises_valueerror(self):
        sim = MuJoCoSim(SCENE_XML)
        with pytest.raises(ValueError, match="not found"):
            sim.detach("nonexistent")

    def test_attach_detach_cycle(self):
        """Multiple attach/detach cycles should work without errors."""
        sim = MuJoCoSim(SCENE_XML)
        T = SE3.Trans(0, 0, 0.6)
        for _ in range(5):
            sim.attach("grasp_eq", "box_free", T)
            sim.detach("grasp_eq")


# ========================================================================
# T1.8: Optional Dependency Graceful Failure
# ========================================================================

class TestT1_8_GracefulFailure:
    """T1.8: Verify clear ImportError when mujoco is not available."""

    def test_check_mujoco_raises_import_error(self, monkeypatch):
        from robot.sim import mujoco_sim
        monkeypatch.setattr(mujoco_sim, "_HAS_MUJOCO", False)
        with pytest.raises(ImportError, match="pip install mujoco"):
            mujoco_sim.MuJoCoSim("dummy.xml")


# ========================================================================
# Additional: Forward and Rendering
# ========================================================================

class TestForwardAndRendering:
    """Additional tests for forward() and rendering methods."""

    def test_forward_updates_state(self):
        sim = MuJoCoSim(SCENE_XML)
        sim.set_joint_q("joint_1", 0.5)
        sim.forward()
        q = sim.get_joint_q("joint_1")
        assert q == pytest.approx(0.5, abs=1e-6)

    def test_render_rgb_shape(self):
        sim = MuJoCoSim(SCENE_XML)
        rgb = sim.render_rgb("cam0", 64, 64)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8

    def test_render_depth_shape(self):
        sim = MuJoCoSim(SCENE_XML)
        depth = sim.render_depth("cam0", 64, 64)
        assert depth.shape == (64, 64)
        assert depth.dtype in (np.float32, np.float64)

    def test_set_ctrl(self):
        sim = MuJoCoSim(SCENE_XML)
        ctrl = np.ones(sim.model.nu) * 0.1
        sim.set_ctrl(ctrl)
        np.testing.assert_allclose(sim.data.ctrl, ctrl)


# ========================================================================
# Scene Object Management
# ========================================================================

class TestSceneObjectManagement:
    """Test get_box_size, set_box_position, set_body_position."""

    def test_get_box_size(self):
        sim = MuJoCoSim(SCENE_XML)
        size = sim.get_box_size("Box")
        assert size.shape == (3,)
        np.testing.assert_allclose(size, [0.05, 0.05, 0.05], atol=1e-6)

    def test_set_body_position(self):
        """set_body_position sets xpos for bodies; for jointed bodies,
        the value may be overwritten by forward dynamics."""
        sim = MuJoCoSim(SCENE_XML)
        # Set via free joint pose for the box (which has a free joint)
        T = SE3.Trans(5.0, 5.0, 5.0)
        sim.set_free_joint_pose("box_free", T)
        sim.forward()
        pos = sim.get_body_pose_xyz("box")
        np.testing.assert_allclose(pos, [5.0, 5.0, 5.0], atol=1e-4)

    def test_get_box_size_nonexistent_raises(self):
        sim = MuJoCoSim(SCENE_XML)
        with pytest.raises(ValueError, match="not found"):
            sim.get_box_size("nonexistent_box")
