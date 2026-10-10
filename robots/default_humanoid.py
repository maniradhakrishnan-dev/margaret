"""
Default 27-DoF Research Humanoid Adapter
Ideal for rapid prototyping and baseline testing.
"""

import os
from typing import List, Optional
from robots.base_robot import BaseRobot

class DefaultHumanoidRobot(BaseRobot):
    @property
    def name(self) -> str:
        return "default_humanoid"

    @property
    def model_path(self) -> str:
        return os.path.abspath("arena/arena.xml")

    @property
    def arm_joint_names(self) -> List[str]:
        return ["shoulder1_right", "shoulder2_right", "elbow_right"]

    @property
    def finger_actuator_names(self) -> List[str]:
        return []

    @property
    def end_effector_body(self) -> str:
        return "right_hand"

    @property
    def keyframe_name(self) -> Optional[str]:
        return None

    @property
    def shoulder_pos(self) -> list:
        return [0.0, -0.17, 1.31]

    @property
    def max_reach_m(self) -> float:
        return 0.85

    @property
    def table_surface_z(self) -> float:
        return 0.75
