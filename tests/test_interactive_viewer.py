"""
Test Suite for Interactive 3D Viewer & Disturbance Sandbox.
Verifies programmatic initialization, keybinding actions, and disturbance dispatches.
"""

import os
import unittest
import numpy as np
import mujoco

from arena.environment_generator import ProceduralEnvironment
from arena.disturbance_engine import EnvironmentDisturber
from skills.vision import SpatialVision
from skills.reach_controller import ReachController
from skills.grasp import GraspController
from agent.tools import ToolRegistry
from agent.brain import AgentBrain
from robots import get_robot


class TestInteractiveViewerHooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = ProceduralEnvironment(robot_platform="unitree_g1", seed=123)
        cls.xml_path, cls.props = generator.generate_random_arena(
            num_objects=3,
            furniture_type="shelf_table",
            obstacle_type="barrier",
            output_filename="test_viewer_arena.xml",
        )
        cls.model = mujoco.MjModel.from_xml_path(cls.xml_path)
        cls.data = mujoco.MjData(cls.model)
        cls.robot = get_robot("unitree_g1")
        cls.robot.setup_data(cls.model, cls.data)
        mujoco.mj_forward(cls.model, cls.data)

        cls.disturber = EnvironmentDisturber(cls.model, cls.data)
        cls.vision = SpatialVision(cls.model, cls.data, camera_name="head_cam")
        cls.reach_ctrl = ReachController(cls.model, cls.data, robot=cls.robot)
        cls.grasp_ctrl = GraspController(cls.model, cls.data, robot=cls.robot)
        cls.registry = ToolRegistry()
        cls.brain = AgentBrain(cls.registry)

    def setUp(self):
        mujoco.mj_resetData(self.model, self.data)
        self.robot.setup_data(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

    def test_hotkey_displace_and_event_emission(self):
        """Verifies Hotkey [D] displacement and event posting."""
        target = self.props[0]["name"]
        old_pos = self.vision.track_object_position_3d(target)
        self.assertIsNotNone(old_pos)

        # Trigger displacement
        ok = self.disturber.displace_object(target, delta_x=0.0, delta_y=0.12, brain=self.brain)
        self.assertTrue(ok)

        new_pos = self.vision.track_object_position_3d(target)
        self.assertAlmostEqual(new_pos[1] - old_pos[1], 0.12, delta=0.01)

        # Confirm harness event
        events = self.brain.graph.harness.get_pending_events()
        self.assertTrue(any(e.event_type == "ENVIRONMENT_CHANGE" for e in events))

    def test_hotkey_table_tremor_shake(self):
        """Verifies Hotkey [T] table tremor shake perturbation."""
        count = self.disturber.shake_all_props(jitter_std=0.03, brain=self.brain)
        self.assertGreaterEqual(count, 3)

    def test_visual_servoing_reach_to_prop(self):
        """Verifies Hotkey [R] visual servoing reaches active prop."""
        target = self.props[1]["name"]
        t_pos = self.vision.track_object_position_3d(target)
        res = self.reach_ctrl.execute_visual_servo_reach(
            target_x=float(t_pos[0]),
            target_y=float(t_pos[1]),
            target_z=float(t_pos[2]),
            target_object=target,
            vision=self.vision,
            hand="auto",
            max_duration_s=1.5,
            control_freq_hz=40.0,
        )
        self.assertIn("reached_position", res)
        self.assertLess(res["distance_error_cm"], 15.0)


if __name__ == "__main__":
    unittest.main()
