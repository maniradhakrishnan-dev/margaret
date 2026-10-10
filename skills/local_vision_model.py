"""
Local Vision Model Engine (YOLO-World Open-Vocabulary Detector).
Runs locally on laptop CPU to continuously perceive the 3D environment in real-time.
Pairs open-vocabulary 2D neural detection with synchronized metric depth deprojection (K^-1).
"""

import os
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw


class LocalVisionEngine:
    """
    Onboard Local Neural Perception Engine.
    Employs YOLO-World for open-vocabulary detection without cloud dependencies.
    """

    def __init__(
        self,
        model_path: str = "yolov8s-world.pt",
        device: str = "cpu",
        default_classes: Optional[List[str]] = None,
    ):
        self.device = device
        self.model_path = model_path
        self.classes = default_classes or [
            "box",
            "cylinder",
            "sphere",
            "capsule",
            "barrier",
            "cup",
            "mug",
            "bottle",
            "tool",
            "prop",
        ]

        print(f"🤖 [Local Vision]: Loading YOLO-World on device '{self.device}'...")
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        self.model.set_classes(self.classes)
        # Pre-warm GPU/CPU execution buffers
        _warmup = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(_warmup, verbose=False, device=self.device)
        print(f"✅ [Local Vision]: Model ready with {len(self.classes)} vocabulary classes: {self.classes}")

    def update_vocabulary(self, new_classes: List[str]) -> None:
        """
        Dynamically adjusts the open-vocabulary detection classes based on user intent
        or Frontier Agent sub-goals (e.g. ['red capsule', 'barrier']).
        """
        if not new_classes:
            return
        self.classes = list(dict.fromkeys(new_classes))  # Deduplicate while preserving order
        self.model.set_classes(self.classes)
        print(f"🔄 [Local Vision]: Vocabulary updated to: {self.classes}")

    def detect_2d(
        self,
        rgb_image: np.ndarray,
        conf_threshold: float = 0.20,
    ) -> List[Dict[str, Any]]:
        """
        Runs local neural forward pass on RGB frame.
        Returns list of 2D bounding boxes and confidence scores.
        """
        t0 = time.time()
        results = self.model(rgb_image, verbose=False, device=self.device, conf=conf_threshold)
        inference_time_ms = (time.time() - t0) * 1000.0

        detections = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        names = self.model.names
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            u_min, v_min, u_max, v_max = xyxy
            u_center = float((u_min + u_max) / 2.0)
            v_center = float((v_min + v_max) / 2.0)
            label = names.get(cls_id, f"class_{cls_id}")

            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [round(x, 1) for x in xyxy],
                "center_pixel": [round(u_center, 1), round(v_center, 1)],
                "inference_time_ms": round(inference_time_ms, 1),
            })

        return detections

    def detect_3d(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        cam_pos: np.ndarray,
        cam_rot: np.ndarray,
        fovy: float,
        conf_threshold: float = 0.20,
    ) -> List[Dict[str, Any]]:
        """
        Fused 2D Neural Detection + Metric Depth Deprojection (K^-1).
        Converts 2D bounding boxes into true 3D world coordinates.
        """
        detections_2d = self.detect_2d(rgb_image, conf_threshold=conf_threshold)
        if not detections_2d:
            return []

        height, width = rgb_image.shape[:2]
        f = 0.5 * height / np.tan(fovy * np.pi / 360.0)
        cx = width / 2.0
        cy = height / 2.0

        grounded_3d = []
        for det in detections_2d:
            u_min, v_min, u_max, v_max = det["bbox"]
            u_c, v_c = det["center_pixel"]

            # Sample depth around the bounding box center (5x5 median patch)
            u_int = int(np.clip(u_c, 2, width - 3))
            v_int = int(np.clip(v_c, 2, height - 3))
            depth_patch = depth_map[v_int - 2 : v_int + 3, u_int - 2 : u_int + 3]
            valid_depths = depth_patch[depth_patch > 0.05]

            if len(valid_depths) == 0:
                continue

            d_c = float(np.median(valid_depths))
            if not (0.10 < d_c < 4.0):
                continue

            # Inverse projective geometry: (u, v, d) -> (x_c, y_c, z_c) -> (X_w, Y_w, Z_w)
            x_c = (u_c - cx) * d_c / f
            y_c = -(v_c - cy) * d_c / f
            z_c = -d_c
            p_cam = np.array([x_c, y_c, z_c], dtype=np.float64)
            p_world = cam_rot @ p_cam + cam_pos

            grounded_3d.append({
                "name": det["label"],
                "confidence": det["confidence"],
                "position_3d": [round(float(p), 3) for p in p_world],
                "bbox_2d": det["bbox"],
                "depth_m": round(d_c, 3),
            })

        return grounded_3d
