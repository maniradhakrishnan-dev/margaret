"""
Unitree G1 Menagerie Humanoid Adapter
Equipped with 29-DoF kinematics, 7-DoF arm, and 3-finger dexterous hands.
"""

import os
from typing import List, Optional
from robots.base_robot import BaseRobot

class UnitreeG1Robot(BaseRobot):
    @property
    def name(self) -> str:
        return "unitree_g1"

    @property
    def model_path(self) -> str:
        return os.path.abspath("models/menagerie/unitree_g1/scene_workbench.xml")

    @property
    def arm_joint_names(self) -> List[str]:
        return [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

    @property
    def finger_actuator_names(self) -> List[str]:
        return [
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
        ]

    @property
    def end_effector_body(self) -> str:
        return "right_hand_middle_0_link"

    @property
    def left_arm_joint_names(self) -> List[str]:
        return [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ]

    @property
    def left_finger_actuator_names(self) -> List[str]:
        return [
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
        ]

    @property
    def left_end_effector_body(self) -> str:
        return "left_hand_middle_0_link"

    def get_arm_joints(self, hand: str = "right") -> List[str]:
        return self.left_arm_joint_names if hand.lower() == "left" else self.arm_joint_names

    def get_finger_actuators(self, hand: str = "right") -> List[str]:
        return self.left_finger_actuator_names if hand.lower() == "left" else self.finger_actuator_names

    def get_end_effector(self, hand: str = "right") -> str:
        return self.left_end_effector_body if hand.lower() == "left" else self.end_effector_body

    @property
    def keyframe_name(self) -> Optional[str]:
        return "stand"

    @property
    def shoulder_pos(self) -> list:
        return [0.0, -0.10, 1.08]

    @property
    def max_reach_m(self) -> float:
        return 0.68

    @property
    def table_surface_z(self) -> float:
        return 0.62

    @property
    def floor_threshold_z(self) -> float:
        return 0.40

    def get_optimal_arm(self, pos: List[float]) -> str:
        """
        Unitree G1 humanoid arms are mounted along Y-axis:
        Right arm is optimal for Y <= 0.04m
        Left arm is optimal for Y > 0.04m
        """
        return "left" if pos[1] > 0.04 else "right"

    @property
    def staging_pos(self) -> List[float]:
        return [0.32, 0.0, 0.62]

    def get_prompt_embodiment_description(self) -> str:
        return (
            "Robot Platform: Unitree G1 29-DoF Flagship Humanoid Robot standing in front of workbench.\n"
            "Kinematic Setup:\n"
            "- Right Arm: optimal for workbench coordinates Y <= 0.04\n"
            "- Left Arm: optimal for workbench coordinates Y > 0.04\n"
            "- Shared Central Workspace: Y in [-0.04, 0.04] (optimal for handovers and stacking)\n"
            "- Table surface: Z = 0.62m. Objects rest on table at Z = 0.65m - 0.67m.\n"
            "- Dropped Objects on Floor: An object has fallen off the table ONLY if Z < 0.40m."
        )
