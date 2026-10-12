"""
Mobile Navigation & Terrain Traversability Controller
Provides closed-loop planar differential drive locomotion, waypoint navigation,
obstacle clearance steering, and ramp climbing for wheeled and wheeled-biped embodiments.
"""

import time
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import mujoco

from robots.base_robot import BaseRobot


class MobileNavigationController:
    """
    Differential Drive Navigation Controller.
    Executes real-time physics-stepped velocity commands on robot wheels,
    tracks base odometry, and steers around obstacle corridors.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot: BaseRobot,
        lock: Optional[Any] = None,
    ):
        self.model = model
        self.data = data
        self.robot = robot
        self.lock = lock

        # Cache wheel actuators
        self.has_wheels = hasattr(robot, "wheel_actuator_names") and bool(robot.wheel_actuator_names)
        self.act_l_id = -1
        self.act_r_id = -1
        if self.has_wheels:
            self.act_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, robot.wheel_actuator_names[0])
            self.act_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, robot.wheel_actuator_names[1])

        # Base body ID for odometry
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
        if self.base_body_id == -1:
            self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    def get_robot_pose(self) -> Dict[str, Any]:
        """Returns live position [x, y, z] and yaw heading (radians)."""
        if self.base_body_id != -1:
            pos = self.data.xpos[self.base_body_id].copy()
            mat = self.data.xmat[self.base_body_id].reshape(3, 3)
            yaw = math.atan2(mat[1, 0], mat[0, 0])
        else:
            pos = np.array([0.0, 0.0, 0.0])
            yaw = 0.0
        return {"position": pos.tolist(), "yaw": yaw, "yaw_deg": math.degrees(yaw)}

    def drive(
        self,
        linear_vel: float,
        angular_vel: float,
        duration_s: float = 0.5,
        steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Dispatches differential drive wheel speeds and advances the physical simulation.
        Positive linear_vel moves forward. Positive angular_vel turns counter-clockwise (left).
        """
        start_pose = self.get_robot_pose()
        if steps is None:
            timestep = max(0.0005, getattr(self.model.opt, "timestep", 0.002))
            steps = max(10, int(duration_s / timestep))

        for step in range(steps):
            if self.lock:
                with self.lock:
                    if hasattr(self.robot, "apply_balance_stabilization"):
                        self.robot.apply_balance_stabilization(self.model, self.data)
                    if hasattr(self.robot, "apply_wheel_velocities"):
                        self.robot.apply_wheel_velocities(self.model, self.data, linear_vel, angular_vel)
                    mujoco.mj_step(self.model, self.data)
            else:
                if hasattr(self.robot, "apply_balance_stabilization"):
                    self.robot.apply_balance_stabilization(self.model, self.data)
                if hasattr(self.robot, "apply_wheel_velocities"):
                    self.robot.apply_wheel_velocities(self.model, self.data, linear_vel, angular_vel)
                mujoco.mj_step(self.model, self.data)

        # Decelerate / clear wheel control
        if self.lock:
            with self.lock:
                if self.act_l_id != -1:
                    self.data.ctrl[self.act_l_id] = 0.0
                if self.act_r_id != -1:
                    self.data.ctrl[self.act_r_id] = 0.0
                if hasattr(self.robot, "apply_balance_stabilization"):
                    self.robot.apply_balance_stabilization(self.model, self.data)
                mujoco.mj_step(self.model, self.data)
        else:
            if self.act_l_id != -1:
                self.data.ctrl[self.act_l_id] = 0.0
            if self.act_r_id != -1:
                self.data.ctrl[self.act_r_id] = 0.0
            if hasattr(self.robot, "apply_balance_stabilization"):
                self.robot.apply_balance_stabilization(self.model, self.data)
            mujoco.mj_step(self.model, self.data)


        end_pose = self.get_robot_pose()
        delta_dist = math.hypot(
            end_pose["position"][0] - start_pose["position"][0],
            end_pose["position"][1] - start_pose["position"][1],
        )

        return {
            "success": True,
            "linear_vel": linear_vel,
            "angular_vel": angular_vel,
            "duration_s": duration_s,
            "distance_traveled_m": round(delta_dist, 3),
            "start_position": [round(x, 3) for x in start_pose["position"]],
            "final_position": [round(x, 3) for x in end_pose["position"]],
            "heading_deg": round(end_pose["yaw_deg"], 1),
        }

    def move_forward(self, distance: float, speed: float = 0.6) -> Dict[str, Any]:
        """Drives forward by the specified distance in meters."""
        duration = max(0.2, abs(distance) / max(0.1, speed))
        direction = 1.0 if distance >= 0 else -1.0
        return self.drive(linear_vel=direction * abs(speed), angular_vel=0.0, duration_s=duration)

    def turn(self, angle_deg: float, turn_rate_deg_s: float = 45.0) -> Dict[str, Any]:
        """Turns the robot in place by angle_deg degrees. Positive is left, negative is right."""
        angle_rad = math.radians(angle_deg)
        duration = max(0.2, abs(angle_deg) / max(10.0, turn_rate_deg_s))
        omega = math.radians(turn_rate_deg_s) * (1.0 if angle_deg >= 0 else -1.0)
        return self.drive(linear_vel=0.0, angular_vel=omega, duration_s=duration)

    def steer_around_obstacle(self, direction: str = "left", bypass_distance: float = 1.0) -> Dict[str, Any]:
        """
        Executes an S-curve maneuver to clear an obstacle ahead:
        1. Steer laterally toward open corridor (left or right)
        2. Drive past obstacle flank
        3. Realign forward heading
        """
        side_sign = 1.0 if direction.lower() == "left" else -1.0
        print(f"🔄 [TRON 1 Navigation]: Steering {direction.upper()} to bypass obstacle corridor...")

        # Turn into bypass lane
        t1 = self.turn(side_sign * 25.0)
        # Traverse past obstacle
        t2 = self.move_forward(bypass_distance * 0.7)
        # Straighten heading
        t3 = self.turn(-side_sign * 25.0)
        # Resume forward advance
        t4 = self.move_forward(bypass_distance * 0.5)

        current = self.get_robot_pose()
        return {
            "success": True,
            "action": f"steer_around_{direction}",
            "current_position": [round(x, 3) for x in current["position"]],
            "heading_deg": round(current["yaw_deg"], 1),
        }

    def climb_ramp(self, speed: float = 0.8, duration_s: float = 2.0) -> Dict[str, Any]:
        """Drives up or across an incline ramp with positive wheel propulsion."""
        print(f"⛰️ [TRON 1 Terrain]: Climbing incline ramp at {speed} m/s...")
        return self.drive(linear_vel=speed, angular_vel=0.0, duration_s=duration_s)

