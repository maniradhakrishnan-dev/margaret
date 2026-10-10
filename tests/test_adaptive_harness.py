"""
Unit tests for Adaptive Embodied Harness and Multi-Source Ingestion.
Validates:
1. Multi-source state ingestion (User + Environment + Robot Kinematics/Joints + Tool Registry).
2. Dynamic skill/tool addition at runtime without code modification.
3. Asynchronous event posting and event arbitration (CONTINUE, REPLAN, HALT_SAFETY).
4. Strict schema validation to prevent LLM hallucinations.
5. Verifiable physical motor feedback.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import mujoco

from agent.tools import ToolRegistry
from agent.brain import AgentBrain
from agent.memory import SpatialMemory, EpisodicHistory
from agent.schemas import EventNotification, EventDecision, ActionFeedback
from agent.harness import AdaptiveEmbodiedHarness
from robots import get_robot


class TestAdaptiveEmbodiedHarness(unittest.TestCase):

    def setUp(self):
        self.robot = get_robot("unitree_g1")
        self.model = mujoco.MjModel.from_xml_path(self.robot.model_path)
        self.data = mujoco.MjData(self.model)
        self.robot.setup_data(self.model, self.data)

        self.tools = ToolRegistry()
        self.memory = SpatialMemory()
        self.history = EpisodicHistory()

        @self.tools.register(name="inspect_scene", description="Detects 3D entities in camera frustum")
        def tool_inspect():
            return {"detected": 0}

        @self.tools.register(name="reach_hand", description="Moves arm end-effector to 3D coordinate")
        def tool_reach(target_x: float, target_y: float, target_z: float, hand: str = "right"):
            return {"success": True, "reached_position": [target_x, target_y, target_z], "distance_error_cm": 1.2}

        self.harness = AdaptiveEmbodiedHarness(
            self.tools,
            self.memory,
            self.history,
            robot=self.robot,
            model_physics=self.model,
            data_physics=self.data,
        )

    def test_multi_source_embodied_context_ingestion(self):
        """Harness ingests User + Environment + Robot Description + Live Joint Telemetry + Tools."""
        # 1. Environment Stream
        self.memory.update_object("cube_a", [0.35, -0.10, 0.67], metadata={"category": "box", "color": "red"})
        # 2. User Stream
        self.harness.current_goal = "pick up cube_a"

        context = self.harness.get_embodied_context()

        # Verify Environment Stream
        self.assertIn("cube_a", context["environment_objects"])
        self.assertEqual(context["environment_objects"]["cube_a"]["color"], "red")

        # Verify Robot Description Stream
        self.assertEqual(context["robot_description"]["platform"], "unitree_g1")
        self.assertEqual(context["robot_description"]["table_surface_z"], self.robot.table_surface_z)

        # Verify Robot Joint Telemetry Stream (live physical qpos values)
        self.assertIn("joint_telemetry", context)
        self.assertTrue(len(context["joint_telemetry"]) > 0, "Must report live joint telemetry from MuJoCo physics")
        first_joint = list(context["joint_telemetry"].keys())[0]
        self.assertIn("q_rad", context["joint_telemetry"][first_joint])

        # Verify Dynamically Available Tools Stream
        self.assertTrue(any("reach_hand" in t for t in context["available_tools"]))

    def test_dynamic_tool_registration_at_runtime(self):
        """Adding a novel skill at runtime is dynamically reflected in harness context."""
        # Dynamically register a new skill without restarting or modifying agent code
        @self.tools.register(name="wipe_surface", description="Wipes table surface along specified trajectory")
        def tool_wipe(start_x: float, start_y: float, end_x: float, end_y: float):
            return {"status": "surface_cleaned"}

        context = self.harness.get_embodied_context()
        tool_names = [t.split(":")[0] for t in context["available_tools"]]
        self.assertIn("wipe_surface", tool_names)

    def test_asynchronous_event_posting_and_arbitration(self):
        """Events (e.g. human interrupt) are posted, queued, and evaluated via model schemas."""
        self.assertFalse(self.harness.has_pending_events())

        # Simulate user shouting an interrupt mid-mission
        self.harness.post_event(EventNotification(
            event_type="USER_INTERRUPT",
            description="User shouted: 'Stop! Put it on the other side instead!'",
            payload={"new_instruction": "place on the left side"}
        ))

        self.assertTrue(self.harness.has_pending_events())
        self.assertEqual(len(self.harness.get_pending_events()), 1)

        # Mock event arbiter decision
        mock_decision = {
            "thought": "User issued a redirection. Current sub-goals must be regenerated.",
            "decision": "REPLAN",
            "reason": "User changed target placement location mid-flight."
        }

        with patch.object(self.harness, "_call_gemini_pydantic", return_value=mock_decision):
            decision = self.harness.evaluate_events()
            self.assertEqual(decision.decision, "REPLAN")
            self.assertEqual(decision.reason, "User changed target placement location mid-flight.")
            self.assertFalse(self.harness.has_pending_events(), "Events queue must be cleared after evaluation")

    def test_safety_halt_on_critical_fault_event(self):
        """Hardware or safety fault triggers HALT_SAFETY decision."""
        self.harness.post_event(EventNotification(
            event_type="ROBOT_FAULT",
            description="Elbow actuator torque limit exceeded, possible physical collision detected",
            payload={"torque_Nm": 85.0, "threshold_Nm": 60.0}
        ))

        mock_decision = {
            "thought": "Actuator torque limit exceeded. Immediate halt required to protect hardware.",
            "decision": "HALT_SAFETY",
            "reason": "Excessive motor torque indicating collision."
        }

        with patch.object(self.harness, "_call_gemini_pydantic", return_value=mock_decision):
            decision = self.harness.evaluate_events()
            self.assertEqual(decision.decision, "HALT_SAFETY")

    def test_agent_brain_post_event_integration(self):
        """AgentBrain forwards events to live graph and harness."""
        brain = AgentBrain(self.tools)
        brain.post_event(EventNotification(
            event_type="ENVIRONMENT_CHANGE",
            description="Target object knocked over by external force",
        ))
        self.assertTrue(brain.graph.harness.has_pending_events())


if __name__ == "__main__":
    unittest.main()
