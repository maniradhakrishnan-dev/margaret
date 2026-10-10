"""
Tests for Independent Procedural Environments and Dynamic Disturbances.
Validates:
1. ProceduralEnvironment generates independent scenes with arbitrary objects and furniture.
2. Margaret drops into the unseen procedural arena and perceives novel entities via camera vision.
3. EnvironmentDisturber displaces objects and knocks them over during physical simulation.
4. Events are emitted to Margaret's Adaptive Embodied Harness and processed cleanly.
"""

import os
import unittest
import numpy as np
import mujoco

from arena.environment_generator import ProceduralEnvironment
from arena.disturbance_engine import EnvironmentDisturber
from skills.vision import SpatialVision
from agent.tools import ToolRegistry
from agent.brain import AgentBrain
from robots import get_robot


class TestProceduralEnvironmentAndDisturbances(unittest.TestCase):

    def setUp(self):
        self.generator = ProceduralEnvironment(robot_platform="unitree_g1", seed=42)
        self.xml_path, self.spawned_props = self.generator.generate_random_arena(
            num_objects=4,
            add_obstacle_barrier=True,
            output_filename="test_procedural_arena.xml"
        )

    def tearDown(self):
        if os.path.exists(self.xml_path):
            os.remove(self.xml_path)

    def test_procedural_arena_generation_and_loading(self):
        """Verifies independent arena generation with random objects and barriers."""
        self.assertTrue(os.path.exists(self.xml_path))
        self.assertEqual(len(self.spawned_props), 4)

        # Load into MuJoCo physics
        model = mujoco.MjModel.from_xml_path(self.xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        # Verify robot and props exist in the compiled model
        self.assertGreater(model.nbody, 40)
        barrier_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_barrier")
        self.assertNotEqual(barrier_id, -1, "Obstacle barrier must be present in arena")

        for prop in self.spawned_props:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prop["name"])
            self.assertNotEqual(bid, -1, f"Object {prop['name']} must exist in compiled physics model")

    def test_margaret_perceives_unseen_procedural_environment(self):
        """Margaret's camera vision discovers novel procedural objects without any pre-coding."""
        model = mujoco.MjModel.from_xml_path(self.xml_path)
        data = mujoco.MjData(model)
        robot = get_robot("unitree_g1")
        robot.setup_data(model, data)

        vision = SpatialVision(model, data, camera_name="head_cam")
        detected = vision.detect_table_objects()

        self.assertGreaterEqual(len(detected), 3, "Vision should detect at least 3 procedural objects in view")
        detected_names = [d["name"] for d in detected]
        for prop in self.spawned_props[:3]:
            self.assertIn(prop["name"], detected_names)

    def test_dynamic_environment_disturbance(self):
        """Environment disturber perturbs objects and notifies Margaret's harness."""
        model = mujoco.MjModel.from_xml_path(self.xml_path)
        data = mujoco.MjData(model)
        robot = get_robot("unitree_g1")
        robot.setup_data(model, data)

        registry = ToolRegistry()
        brain = AgentBrain(registry)
        disturber = EnvironmentDisturber(model, data)

        target_obj = self.spawned_props[0]["name"]
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_obj)
        initial_pos = data.xpos[bid].copy()

        # Disturbance 1: Displace object 8cm along Y axis
        success = disturber.displace_object(target_obj, delta_x=0.0, delta_y=0.08, brain=brain)
        self.assertTrue(success)

        new_pos = data.xpos[bid].copy()
        self.assertAlmostEqual(new_pos[1] - initial_pos[1], 0.08, places=2)

        # Verify Margaret received the disturbance event
        self.assertTrue(brain.graph.harness.has_pending_events())
        events = brain.graph.harness.get_pending_events()
        self.assertEqual(events[0].event_type, "ENVIRONMENT_CHANGE")
        self.assertIn(target_obj, events[0].description)

        # Disturbance 2: Drop object off the table
        success_drop = disturber.drop_object_off_table(target_obj, brain=brain)
        self.assertTrue(success_drop)
        self.assertLess(data.xpos[bid][2], 0.15, "Object must fall to floor level")


if __name__ == "__main__":
    unittest.main()
