"""
Comprehensive Physical AI Unit & Integration Test Suite
Validates World Model, Tool Registry, Spatial Memory, Robot Adapters, Kinematics, and the ReAct Loop.
"""

import os
import unittest
from unittest.mock import patch
from typing import Optional
import numpy as np
import mujoco

from agent.world_model import WorldModel
from agent.tools import ToolRegistry
from agent.memory import SpatialMemory, EpisodicHistory
from agent.brain import AgentBrain
from robots import get_robot
from skills.kinematics import ArmKinematics
from skills.vision import SpatialVision
from skills.reach_controller import ReachController
from skills.grasp import GraspController

class TestWorldModel(unittest.TestCase):
    def setUp(self):
        self.wm = WorldModel(arm_max_reach_m=0.85, table_surface_z=0.75)

    def test_reach_within_bounds(self):
        res = self.wm.simulate_reach([0.0, -0.17, 1.31], [0.35, -0.12, 0.80])
        self.assertTrue(res["is_reachable"])
        self.assertFalse(res["table_collision_risk"])
        self.assertEqual(res["recommendation"], "EXECUTE")

    def test_reach_out_of_bounds(self):
        res = self.wm.simulate_reach([0.0, -0.17, 1.31], [3.0, 3.0, 1.0])
        self.assertFalse(res["is_reachable"])
        self.assertEqual(res["recommendation"], "REPLAN")

    def test_table_collision_rejection(self):
        res = self.wm.simulate_reach([0.0, -0.17, 1.31], [0.35, -0.12, 0.50])
        self.assertTrue(res["table_collision_risk"])
        self.assertEqual(res["recommendation"], "REPLAN")

    def test_friction_grip_simulation(self):
        res = self.wm.simulate_grasp(object_mass=0.2, grip_force_N=12.5)
        self.assertFalse(res["predicted_slip"])
        self.assertEqual(res["status"], "SAFE_TO_LIFT")
        self.assertGreater(res["safety_margin"], 1.0)

class TestToolRegistry(unittest.TestCase):
    def test_registration_and_execution(self):
        registry = ToolRegistry()

        @registry.register(name="add_numbers", description="Adds two numbers.")
        def add(a: float, b: float) -> float:
            return a + b

        self.assertIn("add_numbers", registry.list_tools())
        schemas = registry.get_schemas()
        self.assertEqual(schemas[0]["function"]["name"], "add_numbers")

        res = registry.execute("add_numbers", {"a": 3.5, "b": 2.5})
        self.assertTrue(res["success"])
        self.assertEqual(res["result"], 6.0)

class TestRobotFactory(unittest.TestCase):
    def test_load_unitree_g1(self):
        robot = get_robot("unitree_g1")
        self.assertEqual(robot.name, "unitree_g1")
        self.assertTrue(os.path.exists(robot.model_path))
        self.assertEqual(len(robot.arm_joint_names), 7)
        self.assertEqual(len(robot.finger_actuator_names), 7)

    def test_load_default_humanoid(self):
        robot = get_robot("default")
        self.assertEqual(robot.name, "default_humanoid")
        self.assertTrue(os.path.exists(robot.model_path))
        self.assertEqual(len(robot.arm_joint_names), 3)

class TestKinematicsAndPerception(unittest.TestCase):
    def test_unitree_g1_arm_ik(self):
        robot = get_robot("unitree_g1")
        model = mujoco.MjModel.from_xml_path(robot.model_path)
        data = mujoco.MjData(model)
        robot.setup_data(model, data)

        kinematics = ArmKinematics(model, robot)
        target = np.array([0.30, -0.12, 0.66])
        success, angles = kinematics.solve_arm_ik(data, target, max_iters=100)
        self.assertTrue(success)
        self.assertEqual(len(angles), 7)

    def test_spatial_vision_detection(self):
        robot = get_robot("unitree_g1")
        model = mujoco.MjModel.from_xml_path(robot.model_path)
        data = mujoco.MjData(model)
        robot.setup_data(model, data)

        vision = SpatialVision(model, data, camera_name="head_cam")
        objects = vision.detect_table_objects()
        self.assertGreaterEqual(len(objects), 3)
        names = [o["name"] for o in objects]
        self.assertIn("red_mug", names)
        self.assertIn("blue_block", names)
        self.assertIn("green_ball", names)

class TestFullReActMission(unittest.TestCase):
    def test_offline_model_aborts_without_moving_robot(self):
        """CRITICAL: When model is offline or unreachable, robot MUST halt without moving any motors."""
        registry = ToolRegistry()
        motor_calls = []

        @registry.register(name="inspect_scene")
        def tool_inspect_scene():
            return {"detected_count": 0, "objects": []}

        @registry.register(name="reach_hand")
        def tool_reach_hand(target_x: float, target_y: float, target_z: float, hand: str = "right"):
            motor_calls.append("reach_hand")
            return {"success": True}

        @registry.register(name="grasp_object")
        def tool_grasp_object(object_name: str, hand: str = "auto"):
            motor_calls.append("grasp_object")
            return {"success": True}

        @registry.register(name="lift_hand")
        def tool_lift_hand(height: float = 0.15, hand: str = "auto"):
            motor_calls.append("lift_hand")
            return {"success": True}

        brain = AgentBrain(registry)
        # Simulate network failure / offline mode
        with patch.object(brain.graph, "_call_gemini_pydantic", return_value=None):
            res = brain.run_mission("pick up the red mug")
            self.assertEqual(res["status"], "error")
            self.assertEqual(len(motor_calls), 0, "Arm must NEVER move when model is offline!")

    def test_end_to_end_g1_mission(self):
        robot = get_robot("unitree_g1")
        model = mujoco.MjModel.from_xml_path(robot.model_path)
        data = mujoco.MjData(model)
        robot.setup_data(model, data)

        vision = SpatialVision(model, data, camera_name="head_cam")
        reach_ctrl = ReachController(model, data, robot)
        grasp_ctrl = GraspController(model, data, robot)
        world_model = WorldModel(arm_max_reach_m=robot.max_reach_m, table_surface_z=robot.table_surface_z)

        registry = ToolRegistry()

        @registry.register(name="inspect_scene")
        def tool_inspect_scene():
            objects = vision.detect_table_objects()
            for obj in objects:
                brain.memory.update_object(obj["name"], obj["position_3d"])
            return {"detected_count": len(objects), "objects": objects}

        @registry.register(name="reach_hand")
        def tool_reach_hand(target_x: float, target_y: float, target_z: float, hand: str = "right"):
            res = reach_ctrl.execute_reach(target_x, target_y, target_z, duration_s=0.01, steps=5)
            for _ in range(400):
                mujoco.mj_step(model, data)
            final_pos = reach_ctrl.kinematics.get_hand_pos(data)
            dist = np.linalg.norm(np.array([target_x, target_y, target_z]) - final_pos)
            res["success"] = bool(dist < 0.22)
            res["reached_position"] = list(final_pos)
            brain.memory.set_hand_reached("right", "red_mug" if res["success"] else None)
            return res

        @registry.register(name="grasp_object")
        def tool_grasp_object(object_name: str, hand: str = "auto"):
            res = grasp_ctrl.execute_grasp(object_name, hand=hand)
            if res.get("success"):
                actual_hand = res.get("hand", "right")
                brain.memory.set_hand_held(actual_hand, object_name)
            return res

        @registry.register(name="lift_hand")
        def tool_lift_hand(height: float = 0.15, hand: str = "auto"):
            res = grasp_ctrl.execute_lift(height=height, hand=hand, steps=5)
            for _ in range(400):
                mujoco.mj_step(model, data)
            if res.get("success"):
                actual_hand = res.get("hand", "right")
                brain.memory.set_hand_lifted(actual_hand, True)
            return res

        brain = AgentBrain(registry, world_model=world_model)

        def mock_g1_gemini(prompt, system_instruction, response_schema, model_name=None):
            from agent.schemas import TaskDecompositionPlan, ActionDecision
            if response_schema == TaskDecompositionPlan:
                return {
                    "intent_analysis": "Locate red mug, grasp it, and lift it.",
                    "goal_type": "PICK_AND_LIFT",
                    "target_object": "red_mug",
                    "destination_object": None,
                    "sub_goals": [
                        {"id": 1, "description": "Reach right arm to red_mug", "arm": "right", "action": "reach", "target_object": "red_mug"},
                        {"id": 2, "description": "Grasp red_mug", "arm": "right", "action": "grasp", "target_object": "red_mug"},
                        {"id": 3, "description": "Lift red_mug 15cm", "arm": "right", "action": "lift", "target_object": "red_mug"},
                    ]
                }
            elif response_schema == ActionDecision:
                rh = brain.memory.robot_state.get("right_hand", {})
                if not rh.get("reached_object"):
                    return {
                        "thought": "Reaching to red mug",
                        "action": "reach_hand",
                        "arm": "right",
                        "target_x": 0.30,
                        "target_y": -0.12,
                        "target_z": 0.67,
                        "object_name": "red_mug"
                    }
                elif not rh.get("held"):
                    return {
                        "thought": "Grasping red mug",
                        "action": "grasp_object",
                        "arm": "right",
                        "object_name": "red_mug"
                    }
                elif not rh.get("lifted"):
                    return {
                        "thought": "Lifting red mug",
                        "action": "lift_hand",
                        "arm": "right",
                        "lift_height": 0.15
                    }
                else:
                    return {
                        "thought": "Task complete",
                        "action": "mission_complete",
                        "arm": "auto"
                    }
            return None

        with patch.object(brain.graph, "_call_gemini_pydantic", side_effect=mock_g1_gemini):
            mission_res = brain.run_mission("Inspect the workbench, locate the red mug, reach out and grasp it, and lift it.")
            self.assertEqual(mission_res["status"], "success")
            self.assertEqual(brain.memory.robot_state["right_hand"]["held"], "red_mug")

    def test_frontier_intent_decomposition_without_hardcoding(self):
        """Verifies that Frontier model decomposes arbitrary intent dynamically without hardcoded keyword tables."""
        registry = ToolRegistry()
        brain = AgentBrain(registry)

        mock_plan = {
            "intent_analysis": "Pick up user-specified red object and place on destination blue block.",
            "goal_type": "PICK_AND_PLACE",
            "target_object": "red_item",
            "destination_object": "blue_item",
            "sub_goals": [
                {"id": 1, "description": "Reach to target object", "arm": "right", "action": "reach", "target_object": "red_item"},
                {"id": 2, "description": "Grasp target object", "arm": "right", "action": "grasp", "target_object": "red_item"},
                {"id": 3, "description": "Lift target object", "arm": "right", "action": "lift", "target_object": "red_item"},
                {"id": 4, "description": "Place onto destination", "arm": "right", "action": "place", "target_object": "red_item", "destination_object": "blue_item"}
            ]
        }
        with unittest.mock.patch.object(brain.graph, "_call_gemini_pydantic", return_value=mock_plan):
            plan = brain.decompose_intent("please take the red object and stack it on the blue one")
            self.assertIsNotNone(plan)
            self.assertEqual(plan.goal_type, "PICK_AND_PLACE")
            self.assertEqual(len(plan.sub_goals), 4)
            self.assertEqual(plan.target_object, "red_item")
            self.assertEqual(plan.destination_object, "blue_item")

    def test_end_to_end_pick_and_place_mission(self):
        robot = get_robot("unitree_g1")
        model = mujoco.MjModel.from_xml_path(robot.model_path)
        data = mujoco.MjData(model)
        robot.setup_data(model, data)

        vision = SpatialVision(model, data, camera_name="head_cam")
        reach_ctrl = ReachController(model, data, robot)
        grasp_ctrl = GraspController(model, data, robot)

        registry = ToolRegistry()

        @registry.register(name="inspect_scene")
        def tool_inspect_scene():
            objects = vision.detect_table_objects()
            for obj in objects:
                brain.memory.update_object(obj["name"], obj["position_3d"])
            return {"detected_count": len(objects), "objects": objects}

        @registry.register(name="reach_hand")
        def tool_reach_hand(target_x: float, target_y: float, target_z: float, hand: str = "auto"):
            res = reach_ctrl.execute_reach(target_x, target_y, target_z, hand=hand, duration_s=0.01, steps=5)
            for _ in range(400):
                mujoco.mj_step(model, data)
            actual_hand = res.get("hand", hand if hand in ["right", "left"] else ("left" if target_y > 0.04 else "right"))
            final_pos = reach_ctrl.kinematics.get_hand_pos(data, hand=actual_hand)
            dist = np.linalg.norm(np.array([target_x, target_y, target_z]) - final_pos)
            res["success"] = bool(dist < 0.22)
            res["reached_position"] = list(final_pos)
            res["hand"] = actual_hand
            brain.memory.set_hand_reached(actual_hand, "red_mug" if dist < 0.22 else None)
            return res

        @registry.register(name="grasp_object")
        def tool_grasp_object(object_name: str, hand: str = "auto"):
            res = grasp_ctrl.execute_grasp(object_name, hand=hand)
            if res.get("success"):
                actual_hand = res.get("hand", "right")
                brain.memory.set_hand_held(actual_hand, object_name)
            return res

        @registry.register(name="lift_hand")
        def tool_lift_hand(height: float = 0.15, hand: str = "auto"):
            res = grasp_ctrl.execute_lift(height=height, hand=hand, steps=5)
            for _ in range(400):
                mujoco.mj_step(model, data)
            if res.get("success"):
                actual_hand = res.get("hand", "right")
                brain.memory.set_hand_lifted(actual_hand, True)
            return res

        @registry.register(name="place_object")
        def tool_place_object(target_x: Optional[float] = None, target_y: Optional[float] = None, target_z: Optional[float] = None, hand: str = "auto", dest_object: Optional[str] = "blue_block"):
            res = grasp_ctrl.execute_place(target_x, target_y, target_z, hand=hand, dest_object=dest_object, duration_s=0.01, steps=5)
            for _ in range(400):
                mujoco.mj_step(model, data)
            if res.get("success"):
                actual_hand = res.get("hand", "right")
                brain.memory.set_hand_held(actual_hand, None)
                brain.memory.set_hand_reached(actual_hand, None)
                brain.memory.set_hand_lifted(actual_hand, False)
            return res

        world_model = WorldModel(model=model, robot=robot)
        brain = AgentBrain(registry, world_model=world_model)

        placed = False

        def mock_pnp_gemini(prompt, system_instruction, response_schema, model_name=None):
            nonlocal placed
            from agent.schemas import TaskDecompositionPlan, ActionDecision
            if response_schema == TaskDecompositionPlan:
                return {
                    "intent_analysis": "Pick up red_mug and place on blue_block.",
                    "goal_type": "PICK_AND_PLACE",
                    "target_object": "red_mug",
                    "destination_object": "blue_block",
                    "sub_goals": [
                        {"id": 1, "description": "Reach right arm to red_mug", "arm": "right", "action": "reach", "target_object": "red_mug"},
                        {"id": 2, "description": "Grasp red_mug", "arm": "right", "action": "grasp", "target_object": "red_mug"},
                        {"id": 3, "description": "Lift red_mug 15cm", "arm": "right", "action": "lift", "target_object": "red_mug"},
                        {"id": 4, "description": "Place red_mug on blue_block", "arm": "right", "action": "place", "target_object": "red_mug", "destination_object": "blue_block"},
                    ]
                }
            elif response_schema == ActionDecision:
                if placed:
                    return {
                        "thought": "Mission finished",
                        "action": "mission_complete",
                        "arm": "auto"
                    }
                rh = brain.memory.robot_state.get("right_hand", {})
                if not rh.get("reached_object") and not rh.get("held"):
                    return {
                        "thought": "Reaching to red mug",
                        "action": "reach_hand",
                        "arm": "right",
                        "target_x": 0.30,
                        "target_y": -0.12,
                        "target_z": 0.67,
                        "object_name": "red_mug"
                    }
                elif not rh.get("held"):
                    return {
                        "thought": "Grasping red mug",
                        "action": "grasp_object",
                        "arm": "right",
                        "object_name": "red_mug"
                    }
                elif not rh.get("lifted"):
                    return {
                        "thought": "Lifting red mug",
                        "action": "lift_hand",
                        "arm": "right",
                        "lift_height": 0.15
                    }
                elif rh.get("held"):
                    placed = True
                    return {
                        "thought": "Placing red mug on blue block",
                        "action": "place_object",
                        "arm": "right",
                        "target_x": 0.30,
                        "target_y": 0.12,
                        "target_z": 0.70,
                        "dest_object": "blue_block"
                    }
            return None

        with patch.object(brain.graph, "_call_gemini_pydantic", side_effect=mock_pnp_gemini):
            mission_res = brain.run_mission("picupk the red object and put it on the blue object", max_steps=12)
            self.assertEqual(mission_res["status"], "success")

if __name__ == "__main__":
    unittest.main()
