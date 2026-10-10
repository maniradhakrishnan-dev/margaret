"""
Test Suite for MargaretAgent (Agent-First Architecture).
Verifies:
1. Agent autonomous loading of environment and robot model
2. Plain workbench setup with zero obstacle barriers or top sheets
3. Perception grounding and spatial memory population
4. Dynamic physical disturbances (displacement, knock over, table tremor)
5. Event emission into Agent Brain for adaptive response
"""

import unittest
import numpy as np
from agent.embodied_agent import MargaretAgent


class TestMargaretAgentFirst(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = MargaretAgent(
            robot="unitree_g1",
            num_objects=3,
            furniture="workbench",
            obstacle="none",
            seed=42,
        )
        cls.agent.load_environment()

    def test_plain_workbench_environment_loaded(self):
        """Verifies agent loads clean plain workbench with props and no upper sheets."""
        self.assertIsNotNone(self.agent.model)
        self.assertIsNotNone(self.agent.data)
        self.assertEqual(len(self.agent.spawned_props), 3)

        # Confirm workbench body exists
        import mujoco
        workbench_id = mujoco.mj_name2id(self.agent.model, mujoco.mjtObj.mjOBJ_BODY, "workbench")
        self.assertNotEqual(workbench_id, -1)

        # Confirm NO upper shelf or barrier exists
        shelf_id = mujoco.mj_name2id(self.agent.model, mujoco.mjtObj.mjOBJ_BODY, "shelf_top")
        self.assertEqual(shelf_id, -1)
        barrier_id = mujoco.mj_name2id(self.agent.model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_barrier")
        self.assertEqual(barrier_id, -1)

    def test_perception_grounding(self):
        """Verifies agent grounds table props in spatial memory."""
        detected = self.agent.inspect_scene()
        self.assertGreaterEqual(len(detected), 3)
        self.assertEqual(len(self.agent.brain.memory.objects), 3)

    def test_agent_disturbance_displacement(self):
        """Verifies agent can disturb an object and detect state shift."""
        target = self.agent.get_current_target_name()
        old_pos = self.agent.vision.track_object_position_3d(target)
        self.assertIsNotNone(old_pos)

        # Nudge object +12cm
        self.agent.disturb("displace", target, val=0.12)
        new_pos = self.agent.vision.track_object_position_3d(target)
        self.assertAlmostEqual(new_pos[1] - old_pos[1], 0.12, delta=0.015)

        # Verify event notification reached brain harness
        events = self.agent.brain.graph.harness.get_pending_events()
        self.assertTrue(any(e.event_type == "ENVIRONMENT_CHANGE" for e in events))

    def test_agent_disturbance_knock_and_shake(self):
        """Verifies knock over and table tremor disturb props."""
        target = self.agent.get_current_target_name()
        self.agent.disturb("knock", target)
        self.agent.disturb("shake")

        events = self.agent.brain.graph.harness.get_pending_events()
        self.assertGreater(len(events), 0)

    def test_nudge_active_prop_and_auto_release(self):
        """Verifies nudge_active_prop glides prop and auto-releases grip if held."""
        target = self.agent.get_current_target_name()
        old_pos = self.agent.vision.track_object_position_3d(target)
        self.assertIsNotNone(old_pos)

        # Simulate holding the object with right hand
        self.agent.grasp_ctrl.held_objects["right"] = target
        self.agent.brain.memory.set_hand_held("right", target)

        # Nudge active prop along X and Y
        self.agent.nudge_active_prop(delta_x=0.04, delta_y=0.04)

        # Confirm grip was automatically released before disturbance
        self.assertIsNone(self.agent.grasp_ctrl.held_objects.get("right"))
        self.assertIsNone(self.agent.brain.memory.robot_state["right_hand"]["held"])
        self.assertIsNone(self.agent.brain.memory.robot_state["held_object"])

        # Confirm position shifted
        new_pos = self.agent.vision.track_object_position_3d(target)
        self.assertAlmostEqual(new_pos[0] - old_pos[0], 0.04, delta=0.015)
        self.assertAlmostEqual(new_pos[1] - old_pos[1], 0.04, delta=0.015)


if __name__ == "__main__":
    unittest.main()
