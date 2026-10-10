"""
Test Suite for Continuous Closed-Loop Visual Servoing (50Hz Mid-Motion Tracking).
Verifies that Margaret dynamically curves her arm trajectory when an object
is physically perturbed/displaced while the arm is actively reaching for it.
"""

import os
import unittest
import numpy as np
import mujoco

from arena.environment_generator import ProceduralEnvironment
from arena.disturbance_engine import EnvironmentDisturber
from skills.reach_controller import ReachController
from skills.vision import SpatialVision
from robots import get_robot


class TestVisualServoing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = ProceduralEnvironment(robot_platform="unitree_g1", seed=42)
        cls.xml_path, cls.props = generator.generate_random_arena(
            num_objects=3,
            output_filename="test_servoing_arena.xml"
        )
        cls.model = mujoco.MjModel.from_xml_path(cls.xml_path)
        cls.robot = get_robot("unitree_g1")

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        self.robot.setup_data(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def test_visual_servoing_tracks_displaced_object_mid_flight(self):
        """
        Verifies that when an object is displaced mid-reach, visual servoing
        detects the movement and dynamically curves the arm trajectory to the new location.
        """
        vision = SpatialVision(self.model, self.data, camera_name="head_cam")
        reach_ctrl = ReachController(self.model, self.data, robot=self.robot)
        disturber = EnvironmentDisturber(self.model, self.data)

        target_prop = self.props[0]["name"]
        initial_pos = vision.track_object_position_3d(target_prop)
        self.assertIsNotNone(initial_pos)

        # External disturbance injected at step 10 while arm is in mid-flight
        displaced = False
        def trigger_mid_motion_disturbance(step, current_hand):
            nonlocal displaced
            if step == 10 and not displaced:
                displaced = True
                print("\n💥 [Mid-Motion Perturbation Triggered at step 10]!")
                disturber.displace_object(target_prop, delta_x=0.0, delta_y=0.12)

        # Start visual servoing reach toward initial target
        result = reach_ctrl.execute_visual_servo_reach(
            target_x=float(initial_pos[0]),
            target_y=float(initial_pos[1]),
            target_z=float(initial_pos[2]),
            target_object=target_prop,
            vision=vision,
            hand="auto",
            max_duration_s=2.0,
            control_freq_hz=40.0,
            pos_tolerance_m=0.06,
            step_hook=trigger_mid_motion_disturbance,
        )

        final_obj_pos = vision.track_object_position_3d(target_prop)
        hand_pos = np.array(result["reached_position"])
        distance_to_new_obj_pos = np.linalg.norm(hand_pos - final_obj_pos)

        print("\n🎯 [Visual Servoing Test Results]:")
        print(f"   • Initial Prop Pos: {initial_pos.round(3)}")
        print(f"   • Displaced Prop Pos: {final_obj_pos.round(3)}")
        print(f"   • Hand Final Pos: {hand_pos.round(3)}")
        print(f"   • Final Distance Error: {distance_to_new_obj_pos*100:.1f} cm")
        print(f"   • Dynamic Course Corrections: {result['dynamic_corrections']}")

        # Verify that dynamic adjustments occurred and final distance error is within reach tolerance
        self.assertGreater(result["dynamic_corrections"], 0, "Visual servoing should have detected target displacement")
        self.assertLess(distance_to_new_obj_pos, 0.15, "Arm should reach within 15cm of the displaced position")

    def test_open_loop_fails_when_object_moves(self):
        """
        Contrasts visual servoing with traditional open-loop reaching:
        Open-loop reaches the old coordinate, leaving a large error when object is displaced.
        """
        reach_ctrl = ReachController(self.model, self.data, robot=self.robot)
        disturber = EnvironmentDisturber(self.model, self.data)
        vision = SpatialVision(self.model, self.data, camera_name="head_cam")

        target_prop = self.props[1]["name"]
        initial_pos = vision.track_object_position_3d(target_prop)
        
        # Displace object BEFORE open-loop reach
        disturber.displace_object(target_prop, delta_x=0.0, delta_y=0.18)
        new_obj_pos = vision.track_object_position_3d(target_prop)

        # Open loop reach to INITIAL coordinate without visual tracking
        open_loop_res = reach_ctrl.execute_reach(
            target_x=float(initial_pos[0]),
            target_y=float(initial_pos[1]),
            target_z=float(initial_pos[2]),
            hand="auto",
            duration_s=1.0,
            steps=30,
        )

        hand_pos = np.array(open_loop_res["reached_position"])
        error_to_new_obj = np.linalg.norm(hand_pos - new_obj_pos)
        print(f"\n⚠️ Open-loop error to displaced object: {error_to_new_obj*100:.1f} cm (Failed to adapt)")
        self.assertGreater(error_to_new_obj, 0.12, "Open-loop should miss the displaced object by >12cm")


if __name__ == "__main__":
    unittest.main()
