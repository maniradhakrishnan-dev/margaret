"""
Test Suite for ALOHA Desktop Dual-Arm Embodiment Awareness in Margaret
Validates kinematic arm partitioning, tabletop height grounding, zero floor-recovery false positives,
and bimanual reach execution on the ALOHA ViperX platform.
"""

import os
import unittest
import numpy as np
import mujoco

from robots import get_robot
from agent.world_model import WorldModel
from agent.tools import ToolRegistry
from agent.memory import SpatialMemory, EpisodicHistory
from agent.graph import LiveEmbodiedGraph
from skills.reach_controller import ReachController
from skills.grasp import GraspController
from skills.vision import SpatialVision

class TestAlohaEmbodiment(unittest.TestCase):
    def setUp(self):
        self.robot = get_robot("aloha")
        self.model = mujoco.MjModel.from_xml_path(self.robot.model_path)
        self.data = mujoco.MjData(self.model)
        self.robot.setup_data(self.model, self.data)

        self.reach_ctrl = ReachController(self.model, self.data, robot=self.robot)
        self.grasp_ctrl = GraspController(self.model, self.data, robot=self.robot)
        self.wm = WorldModel(self.model, robot=self.robot)

    def test_aloha_robot_properties(self):
        self.assertEqual(self.robot.name, "aloha")
        self.assertEqual(self.robot.table_surface_z, 0.00)
        self.assertEqual(self.robot.floor_threshold_z, -0.10)
        self.assertEqual(self.robot.get_optimal_arm([0.16, 0.0, 0.034]), "right")
        self.assertEqual(self.robot.get_optimal_arm([-0.16, 0.0, 0.034]), "left")

    def test_aloha_hand_resolution_in_controllers(self):
        # Right arm workspace (X > 0)
        self.assertEqual(self.reach_ctrl._resolve_hand([0.16, 0.0, 0.034]), "right")
        self.assertEqual(self.grasp_ctrl._resolve_hand([0.16, 0.0, 0.034]), "right")

        # Left arm workspace (X < 0)
        self.assertEqual(self.reach_ctrl._resolve_hand([-0.16, 0.0, 0.034]), "left")
        self.assertEqual(self.grasp_ctrl._resolve_hand([-0.16, 0.0, 0.034]), "left")

    def test_tabletop_props_classified_as_on_table_not_on_floor(self):
        registry = ToolRegistry()
        memory = SpatialMemory()
        history = EpisodicHistory()

        # Seed memory with ALOHA props resting on table surface (Z ~ 0.034m - 0.045m)
        memory.update_object("red_mug", [0.16, 0.0, 0.034])
        memory.update_object("blue_block", [-0.16, 0.0, 0.035])
        memory.update_object("green_ball", [0.0, 0.08, 0.035])

        graph = LiveEmbodiedGraph(registry, memory, history, world_model=self.wm)

        # Verify embodiment helpers on graph
        self.assertEqual(graph.table_z, 0.00)
        self.assertEqual(graph.floor_thresh_z, -0.10)
        self.assertEqual(graph.get_optimal_arm([0.16, 0.0, 0.034]), "right")
        self.assertEqual(graph.get_optimal_arm([-0.16, 0.0, 0.034]), "left")

        # Check heuristic reasoning when picking red_mug:
        # Crucial test: MUST NOT return recover_floor_object!
        graph.current_goal = "pick up the red mug"
        decision = graph.node_embodied_reasoning(step_idx=1)

        # Decision action must be reach_hand or inspect_scene, NEVER recover_floor_object!
        self.assertNotEqual(decision.action, "recover_floor_object",
                            "Regression: Margaret falsely classified tabletop prop as fallen on floor!")
        self.assertIn(decision.action, ["reach_hand", "inspect_scene"])
        if decision.action == "reach_hand":
            self.assertEqual(decision.arm, "right")

    def test_physical_reach_and_grasp_on_aloha_rig(self):
        # Test reaching and grasping the red_mug with the right arm
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "red_mug")
        mug_pos = self.data.xpos[body_id].copy()

        reach_res = self.reach_ctrl.execute_reach(
            target_x=float(mug_pos[0]),
            target_y=float(mug_pos[1]),
            target_z=float(mug_pos[2]),
            hand="right",
            duration_s=0.8,
            steps=30
        )
        self.assertTrue(reach_res.get("success"), f"Right arm failed to reach red_mug on ALOHA: {reach_res}")
        self.assertLess(reach_res.get("distance_error_cm", 999), 10.0)

        grasp_res = self.grasp_ctrl.execute_grasp("red_mug", hand="right")
        self.assertTrue(grasp_res.get("success"), f"Right arm failed to grasp red_mug: {grasp_res}")

        # Test lifting the red_mug
        lift_res = self.grasp_ctrl.execute_lift(height=0.10, hand="right")
        self.assertTrue(lift_res.get("success"), f"Lift failed: {lift_res}")
        self.assertGreater(lift_res["final_hand_z"], 0.05)


if __name__ == "__main__":
    unittest.main()
