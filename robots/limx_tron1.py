"""
LimX Dynamics TRON 1 Biped-Wheeled Robot Adapter
Dual-leg articulated wheeled biped with point-foot / wheel drive hybrid locomotion.
Equipped with 6 leg positioning joints and 2 differential continuous wheel drives.
"""

import os
from typing import List, Optional, Tuple
import mujoco
from robots.base_robot import BaseRobot


class Tron1Robot(BaseRobot):
    """
    LimX Dynamics TRON 1 Embodiment Adapter.
    Features:
    - 6-DoF Articulated Legs (abad, hip, knee for left and right)
    - 2-DoF Differential Drive Wheels for high-speed dynamic locomotion
    - Head camera & Onboard IMU for visual perception and balance estimation
    """

    @property
    def name(self) -> str:
        return "limx_tron1"

    @property
    def model_path(self) -> str:
        return os.path.abspath("models/menagerie/limx_tron1/scene.xml")

    @property
    def leg_joint_names(self) -> List[str]:
        return [
            "abad_L_Joint",
            "hip_L_Joint",
            "knee_L_Joint",
            "abad_R_Joint",
            "hip_R_Joint",
            "knee_R_Joint",
        ]

    @property
    def left_leg_joint_names(self) -> List[str]:
        return ["abad_L_Joint", "hip_L_Joint", "knee_L_Joint"]

    @property
    def right_leg_joint_names(self) -> List[str]:
        return ["abad_R_Joint", "hip_R_Joint", "knee_R_Joint"]

    @property
    def wheel_joint_names(self) -> List[str]:
        return ["wheel_L_Joint", "wheel_R_Joint"]

    @property
    def wheel_actuator_names(self) -> List[str]:
        return ["wheel_L_Joint", "wheel_R_Joint"]

    @property
    def leg_actuator_names(self) -> List[str]:
        return [
            "abad_L_Joint",
            "hip_L_Joint",
            "knee_L_Joint",
            "abad_R_Joint",
            "hip_R_Joint",
            "knee_R_Joint",
        ]

    @property
    def wheel_radius(self) -> float:
        """Wheel outer radius in meters."""
        return 0.127

    @property
    def wheel_track(self) -> float:
        """Distance between left and right wheel contact centers in meters."""
        return 0.38

    # BaseRobot Interface Conformity
    @property
    def arm_joint_names(self) -> List[str]:
        """Maps to right leg joints for standard kinematic query compatibility."""
        return self.right_leg_joint_names

    @property
    def finger_actuator_names(self) -> List[str]:
        """TRON 1 does not carry grippers in base mobile form factor."""
        return []

    @property
    def end_effector_body(self) -> str:
        """Base chassis or wheel link."""
        return "wheel_R_Link"

    @property
    def keyframe_name(self) -> Optional[str]:
        return "stand"

    @property
    def shoulder_pos(self) -> list:
        return [0.0, 0.0, 0.65]

    @property
    def max_reach_m(self) -> float:
        return 0.65

    @property
    def table_surface_z(self) -> float:
        return 0.00  # Operates at ground level

    @property
    def floor_threshold_z(self) -> float:
        return -0.10

    def compute_differential_drive(self, linear_vel: float, angular_vel: float) -> Tuple[float, float]:
        """
        Converts commanded linear velocity (m/s) and angular yaw rate (rad/s)
        into left and right wheel angular velocities (rad/s).
        v_left = (v - omega * track / 2) / r
        v_right = (v + omega * track / 2) / r
        """
        half_track = self.wheel_track / 2.0
        v_l = linear_vel - angular_vel * half_track
        v_r = linear_vel + angular_vel * half_track
        omega_l = v_l / self.wheel_radius
        omega_r = v_r / self.wheel_radius
        return omega_l, omega_r

    def apply_wheel_velocities(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        linear_vel: float,
        angular_vel: float,
    ) -> None:
        """Sets wheel actuator control inputs for mobile locomotion."""
        omega_l, omega_r = self.compute_differential_drive(linear_vel, angular_vel)
        act_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_L_Joint")
        act_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_R_Joint")
        if act_l != -1:
            data.ctrl[act_l] = omega_l
        if act_r != -1:
            data.ctrl[act_r] = omega_r

    def get_prompt_embodiment_description(self) -> str:
        return (
            f"Robot Platform: LimX Dynamics TRON 1 (Wheeled-Biped Hybrid).\n"
            f"- Kinematics: 6-DoF articulated legs + 2-DoF continuous differential drive wheels.\n"
            f"- Locomotion: High-speed planar wheel drive with active leg height/balance articulation.\n"
            f"- Sensors: Head inspection camera, 6-axis base IMU."
        )
