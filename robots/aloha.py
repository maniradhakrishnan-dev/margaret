"""
ALOHA Bimanual Desktop Dual-Arm Robot Adapter
Equipped with dual 6-DoF ViperX 300s arms and parallel-jaw pinchers bolted to a rigid desktop frame.
Optimized for 100% isolated bimanual manipulation benchmarking with zero tipping risk.
"""

import os
from typing import List, Optional
import mujoco
from robots.base_robot import BaseRobot


class AlohaRobot(BaseRobot):
    """
    ALOHA Bimanual Dual-Arm Platform.
    Left and right ViperX 300s manipulators facing a central tabletop workspace.
    """

    @property
    def name(self) -> str:
        return "aloha"

    @property
    def model_path(self) -> str:
        return os.path.abspath("models/menagerie/aloha/scene_workbench.xml")

    @property
    def arm_joint_names(self) -> List[str]:
        """Right arm joints."""
        return [
            "right/waist",
            "right/shoulder",
            "right/elbow",
            "right/forearm_roll",
            "right/wrist_angle",
            "right/wrist_rotate",
        ]

    @property
    def finger_actuator_names(self) -> List[str]:
        """Right finger actuator."""
        return ["right/gripper"]

    @property
    def end_effector_body(self) -> str:
        return "right/gripper"

    @property
    def left_arm_joint_names(self) -> List[str]:
        """Left arm joints."""
        return [
            "left/waist",
            "left/shoulder",
            "left/elbow",
            "left/forearm_roll",
            "left/wrist_angle",
            "left/wrist_rotate",
        ]

    @property
    def left_finger_actuator_names(self) -> List[str]:
        """Left finger actuator."""
        return ["left/gripper"]

    @property
    def left_end_effector_body(self) -> str:
        return "left/gripper"

    def get_arm_joints(self, hand: str = "right") -> List[str]:
        return self.left_arm_joint_names if hand.lower() == "left" else self.arm_joint_names

    def get_finger_actuators(self, hand: str = "right") -> List[str]:
        return self.left_finger_actuator_names if hand.lower() == "left" else self.finger_actuator_names

    def get_end_effector(self, hand: str = "right") -> str:
        return self.left_end_effector_body if hand.lower() == "left" else self.end_effector_body

    def gripper_open_val(self, hand: str = "right") -> float:
        """ALOHA ViperX gripper control range: 0.037 is fully open (3.7cm opening)."""
        return 0.037

    def gripper_closed_val(self, hand: str = "right") -> float:
        """ALOHA ViperX gripper control range: 0.005 is firmly clamped around object."""
        return 0.005

    @property
    def keyframe_name(self) -> Optional[str]:
        return "neutral_pose"

    @property
    def shoulder_pos(self) -> list:
        """Right arm base mount position."""
        return [0.469, -0.019, 0.15]

    @property
    def max_reach_m(self) -> float:
        """Maximum physical reach of ViperX 300s manipulator."""
        return 0.75

    @property
    def table_surface_z(self) -> float:
        """Tabletop surface height in meters."""
        return 0.00

    @property
    def floor_threshold_z(self) -> float:
        """Props fall below table if Z < -0.10m on ALOHA rig."""
        return -0.10

    def get_optimal_arm(self, pos: List[float]) -> str:
        """
        ALOHA manipulators are mounted along X-axis:
        Left arm base at X = -0.469m (optimal for X <= 0.0m)
        Right arm base at X = +0.469m (optimal for X > 0.0m)
        """
        return "left" if pos[0] < 0.0 else "right"

    @property
    def staging_pos(self) -> List[float]:
        """Shared central staging position between both ViperX arms."""
        return [0.0, 0.04, 0.04]

    def get_prompt_embodiment_description(self) -> str:
        return (
            "Robot Platform: ALOHA Bimanual Desktop Rig (Dual 6-DoF ViperX 300s Arms bolted to tabletop frame).\n"
            "Rig Kinematic Workspace Setup:\n"
            "- Left Arm: mounted at left side (X = -0.469m), optimal for workbench coordinates X <= 0.0\n"
            "- Right Arm: mounted at right side (X = +0.469m), optimal for workbench coordinates X > 0.0\n"
            "- Shared Central Workspace: X in [-0.08, 0.08], Y in [-0.05, 0.15] (optimal for handovers and stacking)\n"
            "- Table surface: Z = 0.00m. Props rest on the workbench tabletop at Z = 0.03m - 0.05m.\n"
            "- Dropped Objects: Props resting at Z ~ 0.034m are NORMAL on the table! An object has fallen off the table ONLY if Z < -0.10m.\n"
            "- NEVER call recover_floor_object for objects resting on the table (Z >= -0.10m)!"
        )

    def setup_data(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initializes ALOHA dual arms into the neutral manipulation keyframe."""
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "neutral_pose")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
            mujoco.mj_forward(model, data)
