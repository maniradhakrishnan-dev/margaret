"""
Smooth Actuator Reach Controller
Interpolates position-controlled actuators toward target IK angles.
Thread-safe: Controls actuator targets without calling mj_step concurrently.
"""

import time
import mujoco
import numpy as np
import threading
from typing import Optional, Any, List, Union
from skills.kinematics import ArmKinematics
from robots.base_robot import BaseRobot

class ReachController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, robot: Optional[BaseRobot] = None, lock: Optional[threading.Lock] = None):
        self.model = model
        self.data = data
        self.robot = robot
        self.lock = lock
        self.kinematics = ArmKinematics(model, robot)
        self._is_reaching = False

    def _resolve_hand(self, target_pos: Any, requested_hand: str = "auto") -> str:
        req = requested_hand.lower() if requested_hand else "auto"
        if req in ["left", "right"]:
            return req
        if self.robot and hasattr(self.robot, "get_optimal_arm"):
            if isinstance(target_pos, (list, tuple, np.ndarray)):
                return self.robot.get_optimal_arm(list(target_pos))
            return self.robot.get_optimal_arm([0.0, float(target_pos), 0.0])
        # Auto fallback:
        if isinstance(target_pos, (list, tuple, np.ndarray)):
            return "left" if target_pos[1] > 0.05 else "right"
        return "left" if float(target_pos) > 0.05 else "right"

    def execute_reach(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        hand: str = "auto",
        duration_s: float = 1.2,
        steps: int = 40,
        target_object: Optional[str] = None,
        vision: Optional[Any] = None,
    ) -> dict:
        """
        Plans smooth trajectory with obstacle clearance and drives arm actuators in radians.
        If target_object and vision are provided, seamlessly upgrades to closed-loop visual servoing.
        """
        if target_object and vision:
            return self.execute_visual_servo_reach(
                target_x=target_x,
                target_y=target_y,
                target_z=target_z,
                target_object=target_object,
                vision=vision,
                hand=hand,
                max_duration_s=duration_s * 1.5,
            )

        target = np.array([target_x, target_y, target_z], dtype=np.float64)
        active_hand = self._resolve_hand(target, hand)

        arm_names = self.robot.get_arm_joints(active_hand) if self.robot else ["shoulder1_right", "shoulder2_right", "elbow_right"]
        actuator_map = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in arm_names
        }

        # Pre-open gripper fingers before approaching target
        if self.robot and hasattr(self.robot, "get_finger_actuators"):
            f_names = self.robot.get_finger_actuators(active_hand)
            open_val = self.robot.gripper_open_val(active_hand) if hasattr(self.robot, "gripper_open_val") else 0.0
            for fn in f_names:
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, fn)
                if aid != -1:
                    if self.lock:
                        with self.lock:
                            self.data.ctrl[aid] = open_val
                    else:
                        self.data.ctrl[aid] = open_val

        # Check current hand position
        if self.lock:
            with self.lock:
                current_hand = self.kinematics.get_hand_pos(self.data, hand=active_hand)
        else:
            current_hand = self.kinematics.get_hand_pos(self.data, hand=active_hand)

        # Plan waypoints: if starting behind table lip on standing humanoid, lift arm above table height first
        waypoints = []
        if self.robot and self.robot.name == "unitree_g1":
            if current_hand[0] < 0.15 and target_x > 0.15 and current_hand[2] < 0.74:
                wp_lift = np.array([0.18, target_y, 0.78], dtype=np.float64)
                waypoints.append((wp_lift, 0.6, 25))
        waypoints.append((target, duration_s, steps))

        for wp_target, wp_dur, wp_steps in waypoints:
            if self.lock:
                with self.lock:
                    success, target_angles = self.kinematics.solve_arm_ik(self.data, wp_target, hand=active_hand)
            else:
                success, target_angles = self.kinematics.solve_arm_ik(self.data, wp_target, hand=active_hand)

            if not target_angles:
                return {"success": False, "hand": active_hand, "error": "IK solver failed to converge."}

            # Read current controls
            if self.lock:
                with self.lock:
                    current_ctrls = {
                        name: float(self.data.ctrl[act_id]) if act_id != -1 else 0.0
                        for name, act_id in actuator_map.items()
                    }
            else:
                current_ctrls = {
                    name: float(self.data.ctrl[act_id]) if act_id != -1 else 0.0
                    for name, act_id in actuator_map.items()
                }

            # Ramp actuators smoothly
            dt_step = wp_dur / max(wp_steps, 1)
            for alpha in np.linspace(0.0, 1.0, num=wp_steps):
                for name, act_id in actuator_map.items():
                    if act_id != -1 and name in target_angles:
                        val = (1.0 - alpha) * current_ctrls[name] + alpha * target_angles[name]
                        if self.lock:
                            with self.lock:
                                self.data.ctrl[act_id] = val
                        else:
                            self.data.ctrl[act_id] = val
                if self.lock:
                    with self.lock:
                        mujoco.mj_step(self.model, self.data)
                    time.sleep(dt_step)
                else:
                    n_sim = max(1, int(dt_step / self.model.opt.timestep))
                    for _ in range(n_sim):
                        mujoco.mj_step(self.model, self.data)

        # Allow motors to settle with active distance polling
        if self.lock:
            t0 = time.time()
            final_pos = current_hand
            while time.time() - t0 < 0.8:
                with self.lock:
                    for _ in range(10):
                        mujoco.mj_step(self.model, self.data)
                    final_pos = self.kinematics.get_hand_pos(self.data, hand=active_hand)
                dist = np.linalg.norm(target - final_pos)
                if dist < 0.08:
                    break
                time.sleep(0.04)
        else:
            for _ in range(150):
                mujoco.mj_step(self.model, self.data)
            final_pos = self.kinematics.get_hand_pos(self.data, hand=active_hand)

        if self.lock:
            with self.lock:
                final_pos = self.kinematics.get_hand_pos(self.data, hand=active_hand)
        else:
            final_pos = self.kinematics.get_hand_pos(self.data, hand=active_hand)

        dist = np.linalg.norm(target - final_pos)

        return {
            "success": bool(dist < 0.12),
            "hand": active_hand,
            "target": [round(float(x), 3) for x in target],
            "reached_position": [round(float(x), 3) for x in final_pos],
            "distance_error_cm": round(float(dist * 100.0), 1)
        }

    def execute_visual_servo_reach(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        target_object: Optional[str] = None,
        vision: Optional[Any] = None,
        hand: str = "auto",
        max_duration_s: float = 2.0,
        control_freq_hz: float = 50.0,
        pos_tolerance_m: float = 0.05,
        course_correction_cb: Optional[Any] = None,
        step_hook: Optional[Any] = None,
    ) -> dict:
        """
        Continuous Visual Servoing Closed-Loop Controller (50Hz).
        Continuously tracks target object coordinates in real-time.
        If the object moves, is nudged, or displaced mid-flight, dynamically updates
        IK and curves the arm trajectory without stopping or blind execution.
        """
        current_target = np.array([target_x, target_y, target_z], dtype=np.float64)
        active_hand = self._resolve_hand(current_target, hand)

        arm_names = self.robot.get_arm_joints(active_hand) if self.robot else ["shoulder1_right", "shoulder2_right", "elbow_right"]
        actuator_map = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in arm_names
        }

        f_actuator_ids = []
        if self.robot and hasattr(self.robot, "get_finger_actuators"):
            f_names = self.robot.get_finger_actuators(active_hand)
            open_val = self.robot.gripper_open_val(active_hand) if hasattr(self.robot, "gripper_open_val") else 0.0
            for fn in f_names:
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, fn)
                if aid != -1:
                    f_actuator_ids.append((aid, open_val))
                    if self.lock:
                        with self.lock:
                            self.data.ctrl[aid] = open_val
                    else:
                        self.data.ctrl[aid] = open_val

        dt = 1.0 / control_freq_hz
        total_steps = int(max_duration_s * control_freq_hz)
        dynamic_corrections = 0
        self._is_reaching = True

        # Lift arm above table boundary first if starting tucked on G1
        if self.lock:
            with self.lock:
                current_hand = self.kinematics.get_hand_pos(self.data, hand=active_hand)
        else:
            current_hand = self.kinematics.get_hand_pos(self.data, hand=active_hand)

        if self.robot and self.robot.name == "unitree_g1" and current_hand[0] < 0.15 and current_target[0] > 0.15:
            wp_lift = np.array([0.18, current_target[1], 0.76], dtype=np.float64)
            if self.lock:
                with self.lock:
                    _, lift_angles = self.kinematics.solve_arm_ik(self.data, wp_lift, hand=active_hand)
            else:
                _, lift_angles = self.kinematics.solve_arm_ik(self.data, wp_lift, hand=active_hand)
            if lift_angles:
                for _ in range(15):
                    for name, act_id in actuator_map.items():
                        if act_id != -1 and name in lift_angles:
                            if self.lock:
                                with self.lock:
                                    self.data.ctrl[act_id] = lift_angles[name]
                            else:
                                self.data.ctrl[act_id] = lift_angles[name]
                    if self.lock:
                        time.sleep(dt)
                    else:
                        n_sim = max(1, int(dt / self.model.opt.timestep))
                        for _ in range(n_sim):
                            mujoco.mj_step(self.model, self.data)

        for step in range(total_steps):
            # Optional hook for external simulator perturbation mid-motion
            if step_hook:
                step_hook(step, current_hand)

            # 1. Continuous Visual Tracking: Sense live position of the target object
            if target_object and vision and hasattr(vision, "track_object_position_3d"):
                tracked_pos = vision.track_object_position_3d(target_object)
                if tracked_pos is not None:
                    drift = np.linalg.norm(tracked_pos - current_target)
                    if drift > 0.015:
                        dynamic_corrections += 1
                        current_target = tracked_pos.copy()
                        print(f"🎯 [Visual Servoing]: Target '{target_object}' displaced by {drift*100:.1f}cm! Curving trajectory mid-motion to {current_target.round(3)}")
                        if course_correction_cb:
                            course_correction_cb(target_object, current_target)

            # 2. Check current hand position and error vector
            if self.lock:
                with self.lock:
                    current_hand = self.kinematics.get_hand_pos(self.data, hand=active_hand)
            else:
                current_hand = self.kinematics.get_hand_pos(self.data, hand=active_hand)

            error = current_target - current_hand
            dist = np.linalg.norm(error)

            # Convergence check
            if dist <= pos_tolerance_m and step > 15:
                break

            # Maintain pre-opened fingers throughout approach
            for aid, open_val in f_actuator_ids:
                if self.lock:
                    with self.lock:
                        self.data.ctrl[aid] = open_val
                else:
                    self.data.ctrl[aid] = open_val

            # 3. Dynamic Trajectory Generation: Solve IK toward target and smoothly slew actuators
            if self.lock:
                with self.lock:
                    success, target_angles = self.kinematics.solve_arm_ik(self.data, current_target, hand=active_hand, max_iters=25)
            else:
                success, target_angles = self.kinematics.solve_arm_ik(self.data, current_target, hand=active_hand, max_iters=25)

            if success and target_angles:
                max_dtheta = 0.28  # Dynamic slew rate limit per step
                for name, act_id in actuator_map.items():
                    if act_id != -1 and name in target_angles:
                        target_val = target_angles[name]
                        if self.lock:
                            with self.lock:
                                curr = float(self.data.ctrl[act_id])
                                delta = np.clip(target_val - curr, -max_dtheta, max_dtheta)
                                self.data.ctrl[act_id] = curr + delta
                        else:
                            curr = float(self.data.ctrl[act_id])
                            delta = np.clip(target_val - curr, -max_dtheta, max_dtheta)
                            self.data.ctrl[act_id] = curr + delta

            # Step physics simulation
            if self.lock:
                with self.lock:
                    mujoco.mj_step(self.model, self.data)
                time.sleep(dt)
            else:
                n_sim = max(1, int(dt / self.model.opt.timestep))
                for _ in range(n_sim):
                    mujoco.mj_step(self.model, self.data)

        # Final settle & evaluation
        if self.lock:
            t_settle = time.time()
            final_pos = current_hand
            while time.time() - t_settle < 0.5:
                with self.lock:
                    for _ in range(10):
                        mujoco.mj_step(self.model, self.data)
                    final_pos = self.kinematics.get_hand_pos(self.data, hand=active_hand)
                if np.linalg.norm(current_target - final_pos) < 0.08:
                    break
                time.sleep(0.04)
        else:
            for _ in range(120):
                mujoco.mj_step(self.model, self.data)
            final_pos = self.kinematics.get_hand_pos(self.data, hand=active_hand)

        final_dist = np.linalg.norm(current_target - final_pos)
        self._is_reaching = False
        return {
            "success": bool(final_dist < 0.12),
            "hand": active_hand,
            "target": [round(float(x), 3) for x in current_target],
            "reached_position": [round(float(x), 3) for x in final_pos],
            "distance_error_cm": round(float(final_dist * 100.0), 1),
            "dynamic_corrections": dynamic_corrections,
            "mode": "closed_loop_visual_servoing",
        }

    def execute_return_to_rest(self, hand: str = "both", duration_s: float = 1.0, steps: int = 30) -> dict:
        """
        Smoothly drives arm actuators back to default rest/keyframe pose.
        Supports 'right', 'left', or 'both'.
        """
        hands = ["right", "left"] if hand.lower() in ["both", "all", "auto"] else [hand.lower()]
        dt = duration_s / max(1, steps)

        target_ctrls = {}
        for h in hands:
            arm_joints = self.robot.get_arm_joints(h) if self.robot else []
            for jname in arm_joints:
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
                if aid != -1:
                    val = float(self.model.key_ctrl[0, aid]) if self.model.nkey > 0 else 0.0
                    target_ctrls[aid] = val

        for alpha in np.linspace(0.0, 1.0, steps):
            for aid, target_val in target_ctrls.items():
                if self.lock:
                    with self.lock:
                        curr = float(self.data.ctrl[aid])
                        self.data.ctrl[aid] = (1.0 - alpha) * curr + alpha * target_val
                else:
                    curr = float(self.data.ctrl[aid])
                    self.data.ctrl[aid] = (1.0 - alpha) * curr + alpha * target_val

            if self.lock:
                time.sleep(dt)
            else:
                n_sim = max(1, int(dt / self.model.opt.timestep))
                for _ in range(n_sim):
                    mujoco.mj_step(self.model, self.data)

        return {"success": True, "status": "returned_to_rest", "hands": hands}


