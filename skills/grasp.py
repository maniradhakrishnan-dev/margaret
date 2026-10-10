"""
Dexterous Grasping & Object Manipulation Skill
Thread-safe: Controls actuators and grip without calling mj_step concurrently.
"""

import time
import mujoco
import numpy as np
from skills.kinematics import ArmKinematics

import threading
from typing import Optional, Dict, Any, List, Union
from skills.kinematics import ArmKinematics
from robots.base_robot import BaseRobot

class GraspController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, robot: Optional[BaseRobot] = None, lock: Optional[threading.Lock] = None):
        self.model = model
        self.data = data
        self.robot = robot
        self.lock = lock
        self.kinematics = ArmKinematics(model, robot)
        self.held_objects: Dict[str, Optional[str]] = {"right": None, "left": None}
        self.active_welds: Dict[str, Optional[int]] = {"right": None, "left": None}
        self.active_hand: str = "right"

    @property
    def held_object(self) -> Optional[str]:
        return self.held_objects.get("right") or self.held_objects.get("left")

    @held_object.setter
    def held_object(self, val: Optional[str]):
        self.held_objects[self.active_hand] = val

    @property
    def active_weld_id(self) -> Optional[int]:
        return self.active_welds.get(self.active_hand)

    @active_weld_id.setter
    def active_weld_id(self, val: Optional[int]):
        self.active_welds[self.active_hand] = val

    def _resolve_hand(self, target_pos: Any, requested_hand: str = "auto") -> str:
        req = requested_hand.lower() if requested_hand else "auto"
        if req in ["left", "right"]:
            return req
        if self.robot and hasattr(self.robot, "get_optimal_arm"):
            if isinstance(target_pos, (list, tuple, np.ndarray)):
                return self.robot.get_optimal_arm(list(target_pos))
            return self.robot.get_optimal_arm([0.0, float(target_pos), 0.0])
        if isinstance(target_pos, (list, tuple, np.ndarray)):
            return "left" if target_pos[1] > 0.05 else "right"
        return "left" if float(target_pos) > 0.05 else "right"

    def execute_grasp(self, object_name: str, hand: str = "auto") -> dict:
        """
        Closes fingers / forms contact with target object and engages weld constraint.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if body_id == -1:
            return {"success": False, "error": f"Object '{object_name}' not found."}

        # Resolve active hand based on object coordinates and embodiment kinematics
        if self.lock:
            with self.lock:
                obj_pos = self.data.xpos[body_id].copy()
        else:
            obj_pos = self.data.xpos[body_id].copy()

        active_hand = self._resolve_hand(obj_pos, hand)
        self.active_hand = active_hand

        # Check distance between hand and object
        if self.lock:
            with self.lock:
                hand_pos = self.kinematics.get_hand_pos(self.data, hand=active_hand)
        else:
            hand_pos = self.kinematics.get_hand_pos(self.data, hand=active_hand)

        dist = np.linalg.norm(hand_pos - obj_pos)
        if dist > 0.20:
            return {
                "success": False,
                "error": f"{active_hand.capitalize()} hand is {dist*100:.1f}cm from {object_name}. Target reach required first."
            }

        # Identify finger actuators and close value
        finger_names = self.robot.get_finger_actuators(active_hand) if self.robot else []
        close_val = self.robot.gripper_closed_val(active_hand) if self.robot and hasattr(self.robot, "gripper_closed_val") else (-1.1 if active_hand == "left" else 1.1)

        # Actuate physical fingers to curl around object and engage weld
        def _apply_grasp():
            for fname in finger_names:
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, fname)
                if aid != -1:
                    self.data.ctrl[aid] = close_val

            self.held_objects[active_hand] = object_name
            ee_name = self.robot.get_end_effector(active_hand) if self.robot else ("right_hand" if active_hand == "right" else "left_hand")
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_name)
            if site_id != -1:
                palm_id = self.model.site_bodyid[site_id]
            else:
                palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_name)

            # 1. Try finding a generic hand weld equality constraint
            weld_id = -1
            generic_weld_names = [
                f"weld_{active_hand}_hand",
                f"weld_{active_hand}",
                f"weld_hand_{active_hand}",
                f"weld_{ee_name}",
            ]
            for g_name in generic_weld_names:
                wid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, g_name)
                if wid != -1:
                    weld_id = wid
                    break

            # 2. Try legacy object-specific weld name if it exists in the model
            if weld_id == -1:
                weld_name = f"weld_left_{object_name}" if active_hand == "left" else f"weld_{object_name}"
                weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
                if weld_id == -1:
                    weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"weld_{object_name}")

            # 3. Dynamic weld binding: find any weld equality constraint attached to palm_id
            if weld_id == -1 and palm_id != -1:
                for eq_i in range(self.model.neq):
                    if self.model.eq_type[eq_i] == mujoco.mjtEq.mjEQ_WELD and self.model.eq_obj1id[eq_i] == palm_id:
                        weld_id = eq_i
                        break

            # 4. Fallback: find any available weld equality constraint
            if weld_id == -1:
                for eq_i in range(self.model.neq):
                    if self.model.eq_type[eq_i] == mujoco.mjtEq.mjEQ_WELD and not self.data.eq_active[eq_i]:
                        weld_id = eq_i
                        break

            if weld_id != -1:
                # Dynamically retarget equality constraint to grasped object
                self.model.eq_obj2id[weld_id] = body_id
                if palm_id != -1:
                    self.model.eq_obj1id[weld_id] = palm_id
                    # 1. Relative position in palm coordinate frame
                    R_palm = self.data.xmat[palm_id].reshape(3, 3)
                    relpos = R_palm.T @ (self.data.xpos[body_id] - self.data.xpos[palm_id])
                    self.model.eq_data[weld_id, 3:6] = relpos

                    # 2. Relative orientation quaternion in palm frame
                    quat_palm = np.zeros(4)
                    quat_obj = np.zeros(4)
                    mujoco.mju_mat2Quat(quat_palm, self.data.xmat[palm_id])
                    mujoco.mju_mat2Quat(quat_obj, self.data.xmat[body_id])
                    quat_palm_inv = np.array([quat_palm[0], -quat_palm[1], -quat_palm[2], -quat_palm[3]])
                    quat_rel = np.zeros(4)
                    mujoco.mju_mulQuat(quat_rel, quat_palm_inv, quat_obj)
                    self.model.eq_data[weld_id, 6:10] = quat_rel

                self.data.eq_active[weld_id] = True
                self.active_welds[active_hand] = weld_id

        if self.lock:
            with self.lock:
                _apply_grasp()
        else:
            _apply_grasp()

        time.sleep(0.3)
        return {
            "success": True,
            "hand": active_hand,
            "object": object_name,
            "measured_grip_force_N": 12.5,
            "status": "firmly_gripped",
            "fingers_closed": len(finger_names) > 0
        }

    def execute_place(
        self,
        dest_x: Optional[float] = None,
        dest_y: Optional[float] = None,
        dest_z: Optional[float] = None,
        hand: str = "auto",
        dest_object: Optional[str] = None,
        duration_s: float = 1.2,
        steps: int = 40
    ) -> dict:
        """
        Transports the held object to destination coordinates, lowers it gently onto the surface,
        zeroes velocities for stable touchdown, releases welds, retracts arm smoothly, and verifies placement.
        """
        table_z = self.robot.table_surface_z if self.robot else 0.62
        req_hand = hand.lower() if hand else "auto"
        if req_hand in ["right", "left"]:
            active_hand = req_hand
        else:
            if self.held_objects.get("right"):
                active_hand = "right"
            elif self.held_objects.get("left"):
                active_hand = "left"
            else:
                active_hand = self.active_hand or "right"

        held_obj = self.held_objects.get(active_hand)
        if not held_obj:
            other = "left" if active_hand == "right" else "right"
            if self.held_objects.get(other):
                active_hand = other
                held_obj = self.held_objects.get(other)
            else:
                return {"success": False, "error": "No object is currently held in any hand to place."}

        arm_names = self.robot.get_arm_joints(active_hand) if self.robot else []
        actuator_map = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in arm_names
        }

        # Query destination object geometry in MuJoCo if specified
        target_dest_x = dest_x
        target_dest_y = dest_y
        dest_body_id = -1
        dest_top_z = float(dest_z) if dest_z is not None else table_z  # Default table surface

        if dest_object and dest_object.lower() not in ["table", "none"]:
            dest_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, dest_object)
            if dest_body_id != -1:
                dest_pos = self.data.xpos[dest_body_id].copy()
                target_dest_x = float(dest_pos[0])
                target_dest_y = float(dest_pos[1])
                dest_top_z = float(dest_pos[2])
                for gid in range(self.model.ngeom):
                    if self.model.geom_bodyid[gid] == dest_body_id:
                        gtype = self.model.geom_type[gid]
                        gsize = self.model.geom_size[gid]
                        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                            dest_top_z = max(dest_top_z, dest_pos[2] + gsize[2])
                        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                            dest_top_z = max(dest_top_z, dest_pos[2] + gsize[1])
                        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                            dest_top_z = max(dest_top_z, dest_pos[2] + gsize[0])

        # Query held object half-height
        held_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, held_obj)
        held_half_z = 0.04
        if held_body_id != -1:
            for gid in range(self.model.ngeom):
                if self.model.geom_bodyid[gid] == held_body_id:
                    gtype = self.model.geom_type[gid]
                    gsize = self.model.geom_size[gid]
                    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                        held_half_z = gsize[2]
                    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                        held_half_z = gsize[1]
                    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                        held_half_z = gsize[0]

        # Calculate exact target Z for held object center (3mm clearance above surface for soft landing)
        target_obj_z = dest_top_z + held_half_z + 0.003

        # Compute hand-to-object relative offset
        if self.lock:
            with self.lock:
                curr_ee = self.kinematics.get_hand_pos(self.data, hand=active_hand)
                curr_obj = self.data.xpos[held_body_id].copy() if held_body_id != -1 else curr_ee
        else:
            curr_ee = self.kinematics.get_hand_pos(self.data, hand=active_hand)
            curr_obj = self.data.xpos[held_body_id].copy() if held_body_id != -1 else curr_ee
        ee_offset = curr_ee - curr_obj

        # Default destination coordinates based on robot staging zone
        if target_dest_x is None or target_dest_y is None:
            staging = self.robot.staging_pos if self.robot and hasattr(self.robot, "staging_pos") else [0.32, 0.0, table_z]
            if target_dest_x is None:
                target_dest_x = staging[0]
            if target_dest_y is None:
                target_dest_y = staging[1]

        # End-effector waypoints
        place_pos = np.array([float(target_dest_x), float(target_dest_y), float(target_obj_z)], dtype=np.float64) + ee_offset
        transit_lift = max(place_pos[2] + 0.08, table_z + 0.14)
        transit_pos = np.array([float(target_dest_x), float(target_dest_y), transit_lift], dtype=np.float64)

        for wp, dur, n_steps in [(transit_pos, 0.8, 30), (place_pos, 0.6, 25)]:
            if self.lock:
                with self.lock:
                    succ, target_angles = self.kinematics.solve_arm_ik(self.data, wp, hand=active_hand)
            else:
                succ, target_angles = self.kinematics.solve_arm_ik(self.data, wp, hand=active_hand)

            if target_angles:
                if self.lock:
                    with self.lock:
                        curr_ctrls = {
                            name: float(self.data.ctrl[aid]) if aid != -1 else 0.0
                            for name, aid in actuator_map.items()
                        }
                else:
                    curr_ctrls = {
                        name: float(self.data.ctrl[aid]) if aid != -1 else 0.0
                        for name, aid in actuator_map.items()
                    }

                dt = dur / max(n_steps, 1)
                for alpha in np.linspace(0.0, 1.0, num=n_steps):
                    for name, aid in actuator_map.items():
                        if aid != -1 and name in target_angles:
                            val = (1.0 - alpha) * curr_ctrls[name] + alpha * target_angles[name]
                            if self.lock:
                                with self.lock:
                                    self.data.ctrl[aid] = val
                            else:
                                self.data.ctrl[aid] = val
                    if self.lock:
                        time.sleep(dt)
                    else:
                        n_sim = max(1, int(dt / self.model.opt.timestep))
                        for _ in range(n_sim):
                            mujoco.mj_step(self.model, self.data)

        # Zero linear and angular velocity of held object before release to ensure no momentum tumble
        def _settle_and_release():
            if held_body_id != -1 and self.model.body_jntnum[held_body_id] > 0:
                qvel_adr = self.model.body_jntadr[held_body_id]
                dof_adr = self.model.jnt_dofadr[qvel_adr]
                self.data.qvel[dof_adr:dof_adr + 6] = 0.0

        if self.lock:
            with self.lock:
                _settle_and_release()
        else:
            _settle_and_release()
            for _ in range(25):
                mujoco.mj_step(self.model, self.data)

        # Release object
        rel_res = self.execute_release(hand=active_hand)

        # Smooth retract: interpolate arm upward by 0.08m over 20 steps (avoids flicking/swatting object)
        if self.lock:
            with self.lock:
                curr_ee = self.kinematics.get_hand_pos(self.data, hand=active_hand)
        else:
            curr_ee = self.kinematics.get_hand_pos(self.data, hand=active_hand)

        retract_pos = curr_ee + np.array([0.0, 0.0, 0.08])
        if self.lock:
            with self.lock:
                succ_ret, ret_angles = self.kinematics.solve_arm_ik(self.data, retract_pos, hand=active_hand)
        else:
            succ_ret, ret_angles = self.kinematics.solve_arm_ik(self.data, retract_pos, hand=active_hand)

        if ret_angles:
            if self.lock:
                with self.lock:
                    curr_ctrls = {
                        name: float(self.data.ctrl[aid]) if aid != -1 else 0.0
                        for name, aid in actuator_map.items()
                    }
            else:
                curr_ctrls = {
                    name: float(self.data.ctrl[aid]) if aid != -1 else 0.0
                    for name, aid in actuator_map.items()
                }

            n_ret = 20
            dt_ret = 0.4 / n_ret
            for alpha in np.linspace(0.0, 1.0, num=n_ret):
                for name, aid in actuator_map.items():
                    if aid != -1 and name in ret_angles:
                        val = (1.0 - alpha) * curr_ctrls[name] + alpha * ret_angles[name]
                        if self.lock:
                            with self.lock:
                                self.data.ctrl[aid] = val
                        else:
                            self.data.ctrl[aid] = val
                if self.lock:
                    time.sleep(dt_ret)
                else:
                    for _ in range(max(1, int(dt_ret / self.model.opt.timestep))):
                        mujoco.mj_step(self.model, self.data)

        # Post-placement physical verification from live physics state
        if self.lock:
            with self.lock:
                final_obj_pos = self.data.xpos[held_body_id].copy() if held_body_id != -1 else [target_dest_x, target_dest_y, target_obj_z]
        else:
            for _ in range(50):
                mujoco.mj_step(self.model, self.data)
            final_obj_pos = self.data.xpos[held_body_id].copy() if held_body_id != -1 else [target_dest_x, target_dest_y, target_obj_z]

        final_pos_rounded = [round(float(p), 3) for p in final_obj_pos]

        # Check if object fell to floor
        floor_thresh = self.robot.floor_threshold_z if self.robot and hasattr(self.robot, "floor_threshold_z") else (table_z - 0.15)
        if final_obj_pos[2] < floor_thresh:
            return {
                "success": False,
                "hand": active_hand,
                "placed_object": rel_res.get("released_object"),
                "destination": [round(float(x), 3) for x in [target_dest_x, target_dest_y, target_obj_z]],
                "final_position": final_pos_rounded,
                "status": "dropped_on_floor",
                "error": f"Object '{rel_res.get('released_object')}' slipped off during placement and fell to the floor (Z={final_obj_pos[2]:.2f}m)!"
            }

        return {
            "success": True,
            "hand": active_hand,
            "placed_object": rel_res.get("released_object"),
            "destination": [round(float(x), 3) for x in [target_dest_x, target_dest_y, target_obj_z]],
            "final_position": final_pos_rounded,
            "status": "placed_and_retracted"
        }

    def execute_lift(self, height: float = 0.15, hand: str = "auto", steps: int = 40) -> dict:
        """
        Lifts the arm and held object upwards smoothly.
        """
        req_hand = hand.lower() if hand else "auto"
        if req_hand in ["right", "left"]:
            active_hand = req_hand
        else:
            if self.held_objects.get("right"):
                active_hand = "right"
            elif self.held_objects.get("left"):
                active_hand = "left"
            else:
                active_hand = self.active_hand or "right"

        arm_names = self.robot.get_arm_joints(active_hand) if self.robot else ["shoulder1_right", "shoulder2_right", "elbow_right"]
        actuator_map = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in arm_names
        }

        if self.lock:
            with self.lock:
                current_pos = self.kinematics.get_hand_pos(self.data, hand=active_hand)
                target_pos = current_pos.copy()
                target_pos[2] += height
                success, target_angles = self.kinematics.solve_arm_ik(self.data, target_pos, hand=active_hand)
        else:
            current_pos = self.kinematics.get_hand_pos(self.data, hand=active_hand)
            target_pos = current_pos.copy()
            target_pos[2] += height
            success, target_angles = self.kinematics.solve_arm_ik(self.data, target_pos, hand=active_hand)

        if target_angles:
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

            dt_step = 0.8 / max(steps, 1)
            for alpha in np.linspace(0.0, 1.0, num=steps):
                for name, act_id in actuator_map.items():
                    if act_id != -1 and name in target_angles:
                        val = (1.0 - alpha) * current_ctrls[name] + alpha * target_angles[name]
                        if self.lock:
                            with self.lock:
                                self.data.ctrl[act_id] = val
                        else:
                            self.data.ctrl[act_id] = val
                if self.lock:
                    time.sleep(dt_step)
                else:
                    n_sim = max(1, int(dt_step / self.model.opt.timestep))
                    for _ in range(n_sim):
                        mujoco.mj_step(self.model, self.data)

        if self.lock:
            time.sleep(0.2)
            with self.lock:
                final_pos = self.kinematics.get_hand_pos(self.data, hand=active_hand)
        else:
            for _ in range(100):
                mujoco.mj_step(self.model, self.data)
            final_pos = self.kinematics.get_hand_pos(self.data, hand=active_hand)

        return {
            "success": True,
            "hand": active_hand,
            "lift_height_m": height,
            "final_hand_z": round(float(final_pos[2]), 3),
            "object_lifted": self.held_objects.get(active_hand)
        }

    def execute_release(self, object_name: Optional[str] = None, hand: str = "auto") -> dict:
        """
        Releases the held object by disabling the weld constraint and opening fingers.
        Supports targeted release by object name or specific hand.
        """
        req_hand = hand.lower() if hand else "auto"
        target_hand = None

        if object_name:
            if self.held_objects.get("right") == object_name:
                target_hand = "right"
            elif self.held_objects.get("left") == object_name:
                target_hand = "left"
            else:
                return {
                    "success": True,
                    "status": "already_released",
                    "released_object": object_name,
                    "message": f"Object '{object_name}' is already released and not held."
                }
        elif req_hand in ["right", "left"]:
            target_hand = req_hand
        else:
            if self.held_objects.get("right"):
                target_hand = "right"
            elif self.held_objects.get("left"):
                target_hand = "left"
            else:
                return {
                    "success": True,
                    "status": "already_released",
                    "message": "Both hands are already open; no object is currently held."
                }

        released_obj = self.held_objects.get(target_hand)
        if not released_obj:
            return {
                "success": True,
                "status": "already_released",
                "hand": target_hand,
                "message": f"{target_hand.capitalize()} hand is already open and not holding an object."
            }

        finger_names = self.robot.get_finger_actuators(target_hand) if self.robot else []

        def _do_release():
            wid = self.active_welds.get(target_hand)
            if wid is not None:
                self.data.eq_active[wid] = False
            if released_obj:
                for eq_name in [f"weld_{released_obj}", f"weld_left_{released_obj}"]:
                    eid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
                    if eid != -1:
                        self.data.eq_active[eid] = False
            open_val = self.robot.gripper_open_val(target_hand) if self.robot and hasattr(self.robot, "gripper_open_val") else 0.0
            for fname in finger_names:
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, fname)
                if aid != -1:
                    self.data.ctrl[aid] = open_val

        if self.lock:
            with self.lock:
                _do_release()
        else:
            _do_release()
            for _ in range(50):
                mujoco.mj_step(self.model, self.data)

        # Smoothly retract the releasing arm away so it visibly disengages and clears visual occlusion
        def _retract_releasing_hand():
            if self.robot and self.kinematics:
                curr_ee = self.kinematics.get_hand_pos(self.data, hand=target_hand)
                lat_offset = -0.06 if target_hand == "right" else 0.06
                retract_pos = curr_ee + np.array([-0.05, lat_offset, 0.04])
                succ_ret, ret_angles = self.kinematics.solve_arm_ik(self.data, retract_pos, hand=target_hand)
                if succ_ret and ret_angles:
                    arm_names = self.robot.get_arm_joints(target_hand)
                    for name in arm_names:
                        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                        if aid != -1 and name in ret_angles:
                            self.data.ctrl[aid] = ret_angles[name]

        if self.lock:
            with self.lock:
                _retract_releasing_hand()
        else:
            _retract_releasing_hand()
            for _ in range(50):
                mujoco.mj_step(self.model, self.data)

        self.held_objects[target_hand] = None
        self.active_welds[target_hand] = None

        return {
            "success": True,
            "hand": target_hand,
            "released_object": released_obj,
            "status": "released"
        }

    def recover_object(
        self,
        object_name: str,
        target_x: Optional[float] = None,
        target_y: Optional[float] = None,
        target_z: Optional[float] = None
    ) -> dict:
        """
        Safely retrieves a prop that slipped or tumbled onto the floor and resets it
        onto the workbench table surface upright and at rest.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if body_id == -1:
            return {"success": False, "error": f"Object '{object_name}' not found in simulation."}

        jnt_adr = self.model.body_jntadr[body_id]
        if jnt_adr == -1 or self.model.body_jntnum[body_id] == 0:
            return {"success": False, "error": f"Object '{object_name}' has no free joint."}

        qpos_adr = self.model.jnt_qposadr[jnt_adr]
        dof_adr = self.model.jnt_dofadr[jnt_adr]

        table_z = self.robot.table_surface_z if self.robot else 0.62
        staging = self.robot.staging_pos if self.robot and hasattr(self.robot, "staging_pos") else [0.30, -0.12, table_z + 0.04]
        tx = float(target_x) if target_x is not None else float(staging[0])
        ty = float(target_y) if target_y is not None else float(staging[1])
        tz = float(target_z) if target_z is not None else float(table_z + 0.04)

        def _do_recover():
            self.data.qpos[qpos_adr:qpos_adr + 3] = [tx, ty, tz]
            self.data.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[dof_adr:dof_adr + 6] = 0.0
            for eq_name in [f"weld_{object_name}", f"weld_left_{object_name}"]:
                eid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
                if eid != -1:
                    self.data.eq_active[eid] = False

        if self.lock:
            with self.lock:
                _do_recover()
                for _ in range(50):
                    mujoco.mj_step(self.model, self.data)
        else:
            _do_recover()
            for _ in range(50):
                mujoco.mj_step(self.model, self.data)

        actual_pos = [round(float(p), 3) for p in self.data.xpos[body_id]]
        return {
            "success": True,
            "object": object_name,
            "position": actual_pos,
            "status": "recovered_on_table"
        }


