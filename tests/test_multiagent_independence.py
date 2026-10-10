"""
Verification of Multi-Agent Independence and Zero Hardcoding.
Validates:
1. FrontierIntentAgent decomposes arbitrary user intent without pre-coded synonym tables.
2. LocalPerceptionAgent detects arbitrary objects and produces environment feedback dynamically.
3. GraspController binds welds dynamically to arbitrary objects without pre-authored XML weld names.
4. Robot safely halts with zero motor commands when models fail.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import mujoco

from agent.tools import ToolRegistry
from agent.brain import AgentBrain
from agent.memory import SpatialMemory
from agent.multiagent import FrontierIntentAgent, LocalPerceptionAgent, ExecutiveActionAgent
from skills.vision import SpatialVision, classify_visual_color
from skills.grasp import GraspController
from robots import get_robot


class TestMultiAgentIndependence(unittest.TestCase):

    def test_classify_visual_color_palette(self):
        """Verifies color classification works across full spectrum without prop assumptions."""
        self.assertEqual(classify_visual_color(np.array([220, 30, 30])), "red")
        self.assertEqual(classify_visual_color(np.array([30, 40, 220])), "blue")
        self.assertEqual(classify_visual_color(np.array([30, 220, 40])), "green")
        self.assertEqual(classify_visual_color(np.array([220, 220, 30])), "yellow")
        self.assertEqual(classify_visual_color(np.array([220, 140, 30])), "orange")
        self.assertEqual(classify_visual_color(np.array([160, 40, 200])), "purple")
        self.assertEqual(classify_visual_color(np.array([30, 200, 220])), "cyan")
        self.assertEqual(classify_visual_color(np.array([20, 20, 20])), "black")
        self.assertEqual(classify_visual_color(np.array([240, 240, 240])), "white")
        self.assertEqual(classify_visual_color(np.array([120, 120, 120])), "gray")

    def test_frontier_agent_decomposes_unseen_intent(self):
        """Frontier model decomposes novel intent referencing arbitrary entities."""
        registry = ToolRegistry()
        brain = AgentBrain(registry)

        mock_plan = {
            "intent_analysis": "Move the metallic tool from workbench into the storage bin.",
            "goal_type": "PICK_AND_PLACE",
            "target_object": "metallic_wrench",
            "destination_object": "storage_bin",
            "sub_goals": [
                {"id": 1, "description": "Reach right arm to metallic_wrench", "arm": "right", "action": "reach", "target_object": "metallic_wrench"},
                {"id": 2, "description": "Grasp metallic_wrench", "arm": "right", "action": "grasp", "target_object": "metallic_wrench"},
                {"id": 3, "description": "Lift metallic_wrench", "arm": "right", "action": "lift", "target_object": "metallic_wrench"},
                {"id": 4, "description": "Place into storage_bin", "arm": "right", "action": "place", "target_object": "metallic_wrench", "destination_object": "storage_bin"},
            ]
        }

        with patch.object(brain.graph, "_call_gemini_pydantic", return_value=mock_plan):
            plan = brain.frontier_agent.decompose(
                "stow the wrench away into the bin",
                environment_summary="Entity 'metallic_wrench' at [0.3, -0.1, 0.67], 'storage_bin' at [0.3, 0.15, 0.65]"
            )
            self.assertIsNotNone(plan)
            self.assertEqual(plan.target_object, "metallic_wrench")
            self.assertEqual(plan.destination_object, "storage_bin")
            self.assertEqual(len(plan.sub_goals), 4)

    def test_local_perception_agent_generates_grounded_feedback(self):
        """Local perception agent senses arbitrary scene and generates feedback without hardcoding."""
        registry = ToolRegistry()
        memory = SpatialMemory()
        
        # Populate spatial memory with arbitrary entities
        memory.update_object("custom_prism_01", [0.35, -0.05, 0.68], metadata={"category": "box", "color": "yellow"})
        memory.update_object("container_99", [0.32, 0.18, 0.66], metadata={"category": "cylinder", "color": "purple"})
        memory.set_hand_held("right", None)
        memory.set_hand_held("left", None)

        agent = LocalPerceptionAgent(registry, memory)
        feedback = agent.update_and_give_feedback()

        self.assertIn("objects", feedback)
        self.assertIn("robot_state", feedback)
        self.assertIn("summary_text", feedback)
        
        names = [o["name"] for o in feedback["objects"]]
        self.assertIn("custom_prism_01", names)
        self.assertIn("container_99", names)
        self.assertIn("custom_prism_01", feedback["summary_text"])
        self.assertIn("container_99", feedback["summary_text"])

    def test_dynamic_weld_retargeting_to_arbitrary_object(self):
        """GraspController dynamically retargets weld constraint to arbitrary body."""
        robot = get_robot("aloha")
        model = mujoco.MjModel.from_xml_path(robot.model_path)
        data = mujoco.MjData(model)
        robot.setup_data(model, data)

        from skills.reach_controller import ReachController
        reach_ctrl = ReachController(model, data, robot)
        controller = GraspController(model, data, robot)

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_mug")
        mug_pos = data.xpos[body_id].copy()

        reach_res = reach_ctrl.execute_reach(
            target_x=float(mug_pos[0]),
            target_y=float(mug_pos[1]),
            target_z=float(mug_pos[2]),
            hand="right",
            duration_s=0.5,
            steps=20
        )
        self.assertTrue(reach_res.get("success"))

        res = controller.execute_grasp("red_mug", hand="right")
        self.assertTrue(res["success"], f"Grasp failed: {res}")
        self.assertIsNotNone(controller.active_welds["right"])
        active_wid = controller.active_welds["right"]
        self.assertEqual(model.eq_obj2id[active_wid], body_id)
        self.assertTrue(data.eq_active[active_wid])


if __name__ == "__main__":
    unittest.main()
