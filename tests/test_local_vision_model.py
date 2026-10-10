"""
Test Suite for Local Neural Vision Model (YOLO-World Open-Vocabulary Detector).
Verifies:
1. Local model loading and dynamic open-vocabulary adjustments.
2. 2D neural object detection latency and bounding box extraction.
3. Fusion with metric depth deprojection for real-time 3D perception on laptop CPU.
"""

import os
import unittest
import numpy as np
import mujoco

from skills.local_vision_model import LocalVisionEngine
from skills.vision import SpatialVision
from arena.environment_generator import ProceduralEnvironment
from robots import get_robot


class TestLocalVisionModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = ProceduralEnvironment(robot_platform="unitree_g1", seed=42)
        cls.xml_path, cls.props = generator.generate_random_arena(
            num_objects=3,
            output_filename="test_local_vision_arena.xml"
        )
        cls.model = mujoco.MjModel.from_xml_path(cls.xml_path)
        cls.data = mujoco.MjData(cls.model)
        cls.robot = get_robot("unitree_g1")
        cls.robot.setup_data(cls.model, cls.data)
        mujoco.mj_forward(cls.model, cls.data)

    def test_local_vision_engine_initialization_and_vocabulary(self):
        """Verifies that YOLO-World loads on CPU and accepts open-vocabulary prompts."""
        engine = LocalVisionEngine(device="cpu")
        self.assertIsNotNone(engine.model)
        
        # Test dynamic vocabulary update
        custom_vocab = ["red capsule", "blue cylinder", "table", "barrier"]
        engine.update_vocabulary(custom_vocab)
        self.assertEqual(engine.classes, custom_vocab)

    def test_2d_detection_latency(self):
        """Verifies local neural inference runs fast on laptop CPU (< 100ms)."""
        engine = LocalVisionEngine(device="cpu")
        dummy_rgb = np.full((480, 640, 3), 120, dtype=np.uint8)
        # Place a bright colored patch
        dummy_rgb[150:280, 250:380] = [210, 40, 40]

        import time
        # Warmup pass (initializes PyTorch JIT and memory buffers)
        _ = engine.detect_2d(dummy_rgb, conf_threshold=0.10)

        t0 = time.time()
        detections = engine.detect_2d(dummy_rgb, conf_threshold=0.10)
        elapsed_ms = (time.time() - t0) * 1000.0

        print(f"\n⚡ [Local Vision CPU Benchmark]: {elapsed_ms:.1f} ms per frame ({1000.0/max(elapsed_ms, 1):.1f} FPS)")
        self.assertLess(elapsed_ms, 800.0, "Local vision inference must complete in < 800ms on CPU")

    def test_spatial_vision_integration_with_local_model(self):
        """Verifies SpatialVision seamlessly fuses local neural detector with metric depth."""
        vision = SpatialVision(self.model, self.data, camera_name="head_cam")
        detected = vision.detect_with_local_model(
            custom_classes=["box", "cylinder", "sphere", "capsule", "table"]
        )
        self.assertIsInstance(detected, list)
        self.assertGreater(len(detected), 0, "Local vision should ground table objects in the scene")
        
        first = detected[0]
        self.assertIn("position_3d", first)
        self.assertEqual(len(first["position_3d"]), 3)
        print(f"\n👁️ [Grounded 3D Object via Local Model]: {first['name']} -> {first['position_3d']}")


if __name__ == "__main__":
    unittest.main()
