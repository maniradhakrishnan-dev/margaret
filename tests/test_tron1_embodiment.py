"""
Test Suite for LimX Dynamics TRON 1 Biped-Wheeled Robot Embodiment in Margaret
Validates model loading, joint and actuator configurations, standing keyframe,
differential wheel drive kinematics, sensor pipelines, and simulation stability.
"""

import os
import unittest
import numpy as np
import mujoco

from robots import get_robot, Tron1Robot
from agent.world_model import WorldModel


class TestTron1Embodiment(unittest.TestCase):
    def setUp(self):
        self.robot = get_robot("limx_tron1")
        self.assertTrue(os.path.exists(self.robot.model_path), f"Model path not found: {self.robot.model_path}")
        self.model = mujoco.MjModel.from_xml_path(self.robot.model_path)
        self.data = mujoco.MjData(self.model)
        self.robot.setup_data(self.model, self.data)

    def test_factory_aliases(self):
        """Verify all aliases instantiate Tron1Robot."""
        for alias in ["limx_tron1", "tron1", "wheeled_biped"]:
            bot = get_robot(alias)
            self.assertIsInstance(bot, Tron1Robot)
            self.assertEqual(bot.name, "limx_tron1")

    def test_model_structure_and_degrees_of_freedom(self):
        """Verify joints, actuators, and generalized coordinates."""
        # 1 free joint (root, 7 qpos, 6 qvel) + 6 leg revolute joints + 2 wheel continuous joints = 15 qpos, 14 qvel
        self.assertEqual(self.model.nq, 15, "Expected 15 generalized positions (7 base + 8 joints)")
        self.assertEqual(self.model.nv, 14, "Expected 14 generalized velocities (6 base + 8 joints)")
        self.assertEqual(self.model.nu, 8, "Expected 8 actuators (6 leg position + 2 wheel velocity)")

        # Verify joint names
        for j_name in self.robot.leg_joint_names:
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
            self.assertGreaterEqual(j_id, 0, f"Joint {j_name} not found in model")

        for w_name in self.robot.wheel_joint_names:
            w_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, w_name)
            self.assertGreaterEqual(w_id, 0, f"Wheel joint {w_name} not found in model")

    def test_sensors_and_camera(self):
        """Verify IMU sensors and head inspection camera exist."""
        # Camera
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
        self.assertGreaterEqual(cam_id, 0, "head_cam must exist for visual perception")

        # Sensors
        for s_name in ["imu_quat", "imu_gyro", "imu_acc"]:
            s_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, s_name)
            self.assertGreaterEqual(s_id, 0, f"Sensor {s_name} not found")

    def test_stand_keyframe_initialization(self):
        """Verify that stand keyframe positions robot upright above floor."""
        self.assertEqual(self.robot.keyframe_name, "stand")
        kf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.assertGreaterEqual(kf_id, 0, "Stand keyframe must exist")

        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
        base_z = self.data.xpos[base_id][2]
        self.assertGreater(base_z, 0.45, f"Robot base should be elevated at standing height, got {base_z:.3f}m")

    def test_differential_drive_kinematics(self):
        """Test wheel angular velocity computations for planar motion."""
        # Pure forward 1.0 m/s
        omega_l, omega_r = self.robot.compute_differential_drive(linear_vel=1.0, angular_vel=0.0)
        expected_omega = 1.0 / self.robot.wheel_radius
        self.assertAlmostEqual(omega_l, expected_omega, places=4)
        self.assertAlmostEqual(omega_r, expected_omega, places=4)

        # Pure counter-clockwise spin (angular_vel = 1.0 rad/s)
        omega_l_spin, omega_r_spin = self.robot.compute_differential_drive(linear_vel=0.0, angular_vel=1.0)
        self.assertAlmostEqual(omega_l_spin, -omega_r_spin, places=4)
        self.assertLess(omega_l_spin, 0.0)
        self.assertGreater(omega_r_spin, 0.0)

    def test_apply_wheel_velocities_and_step(self):
        """Apply drive velocities and step physics without numerical instability."""
        self.robot.apply_wheel_velocities(self.model, self.data, linear_vel=0.5, angular_vel=0.0)

        # Step simulation forward for 100 timesteps (0.2 seconds at dt=0.002)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        # Verify no NaN or Inf states
        self.assertFalse(np.isnan(self.data.qpos).any(), "qpos contains NaN values")
        self.assertFalse(np.isnan(self.data.qvel).any(), "qvel contains NaN values")

        # Verify wheels spun forward
        act_l = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_L_Joint")
        act_r = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_R_Joint")
        self.assertAlmostEqual(self.data.ctrl[act_l], 0.5 / self.robot.wheel_radius, places=3)
        self.assertAlmostEqual(self.data.ctrl[act_r], 0.5 / self.robot.wheel_radius, places=3)

    def test_prompt_embodiment_description(self):
        """Check prompt description includes key kinematic details."""
        desc = self.robot.get_prompt_embodiment_description()
        self.assertIn("LimX Dynamics TRON 1", desc)
        self.assertIn("differential drive wheels", desc)


if __name__ == "__main__":
    unittest.main()
