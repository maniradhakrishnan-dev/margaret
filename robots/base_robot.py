"""
Abstract Base Robot Interface
Enables hot-swapping between different robot form factors (Unitree G1, ALOHA, Quadrupeds, Custom Humanoids).
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import mujoco

class BaseRobot(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Name / Identifier of the robot."""
        pass

    @property
    @abstractmethod
    def model_path(self) -> str:
        """Path to the MJCF XML arena."""
        pass

    @property
    @abstractmethod
    def arm_joint_names(self) -> List[str]:
        """Names of arm joints controlled by IK."""
        pass

    @property
    @abstractmethod
    def finger_actuator_names(self) -> List[str]:
        """Names of finger actuators for dexterous grip."""
        pass

    @property
    @abstractmethod
    def end_effector_body(self) -> str:
        """Name of the body or site used as the end effector."""
        pass

    @property
    @abstractmethod
    def keyframe_name(self) -> Optional[str]:
        """Default keyframe name (e.g. 'stand')."""
        pass

    @property
    @abstractmethod
    def shoulder_pos(self) -> list:
        """Approximate 3D shoulder origin for reach checking."""
        pass

    @property
    @abstractmethod
    def max_reach_m(self) -> float:
        """Maximum physical reach of the arm in meters."""
        pass

    @property
    @abstractmethod
    def table_surface_z(self) -> float:
        """Workbench tabletop surface height in meters."""
        pass

    @property
    def floor_threshold_z(self) -> float:
        """Height threshold below which an object has fallen off the table."""
        return self.table_surface_z - 0.15

    def get_optimal_arm(self, pos: List[float]) -> str:
        """Determines the kinematically optimal arm for target coordinates [x, y, z]."""
        return "left" if pos[1] > 0.04 else "right"

    @property
    def staging_pos(self) -> List[float]:
        """Shared central staging position [x, y, z] for handovers or stacking."""
        return [0.32, 0.0, self.table_surface_z]

    def get_prompt_embodiment_description(self) -> str:
        """Embodiment description injected into Margaret's prompt."""
        return (
            f"Robot Platform: {self.name}.\n"
            f"- Table surface: Z = {self.table_surface_z:.2f}m.\n"
            f"- Floor threshold: Z < {self.floor_threshold_z:.2f}m."
        )

    def get_arm_joints(self, hand: str = "right") -> List[str]:
        return self.arm_joint_names

    def get_finger_actuators(self, hand: str = "right") -> List[str]:
        return self.finger_actuator_names

    def get_end_effector(self, hand: str = "right") -> str:
        return self.end_effector_body

    def gripper_open_val(self, hand: str = "right") -> float:
        """Control value to actuate fingers into fully open state."""
        return 0.0

    def gripper_closed_val(self, hand: str = "right") -> float:
        """Control value to actuate fingers into firmly closed state."""
        return -1.1 if hand.lower() == "left" else 1.1

    def setup_data(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Custom initialization hook (e.g., applying keyframe)."""
        if self.keyframe_name:
            kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, self.keyframe_name)
            if kf_id != -1:
                mujoco.mj_resetDataKeyframe(model, data, kf_id)
                if model.key_ctrl is not None and len(model.key_ctrl[kf_id]) == model.nu:
                    data.ctrl[:] = model.key_ctrl[kf_id]
                mujoco.mj_forward(model, data)
