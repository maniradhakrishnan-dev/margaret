"""
Jacobian-Based Numerical Inverse Kinematics (IK) Solver
Solves target joint angles with proper radian-to-degree conversion for position servos.
"""

import mujoco
import numpy as np
from typing import Dict, Tuple, Optional
from robots.base_robot import BaseRobot

class ArmKinematics:
    def __init__(self, model: mujoco.MjModel, robot: Optional[BaseRobot] = None):
        self.model = model
        self.scratch_data = mujoco.MjData(model)
        self.robot = robot

        if robot:
            self.arm_joints = robot.arm_joint_names
            self.ee_name = robot.end_effector_body
        else:
            self.arm_joints = ["shoulder1_right", "shoulder2_right", "elbow_right"]
            self.ee_name = "right_palm"

        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.ee_name)
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.ee_name)
        if self.ee_site_id == -1 and self.ee_body_id == -1:
            self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand")

        self.joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.arm_joints
        ]
        self.dof_addrs = [model.jnt_dofadr[j] for j in self.joint_ids if j != -1]
        self.q_addrs = [model.jnt_qposadr[j] for j in self.joint_ids if j != -1]

    def _resolve_arm(self, hand: str = "right"):
        hand_lower = hand.lower() if hand else "right"
        if self.robot:
            arm_joints = self.robot.get_arm_joints(hand_lower)
            ee_name = self.robot.get_end_effector(hand_lower)
        else:
            if hand_lower == "left":
                arm_joints = ["shoulder1_left", "shoulder2_left", "elbow_left"]
                ee_name = "left_hand"
            else:
                arm_joints = ["shoulder1_right", "shoulder2_right", "elbow_right"]
                ee_name = "right_palm"

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_name)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        if site_id == -1 and body_id == -1:
            fallback = "left_hand" if hand_lower == "left" else "right_hand"
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, fallback)

        j_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in arm_joints]
        dofs = [self.model.jnt_dofadr[j] for j in j_ids if j != -1]
        qadrs = [self.model.jnt_qposadr[j] for j in j_ids if j != -1]
        return arm_joints, site_id, body_id, j_ids, dofs, qadrs

    def get_hand_pos(self, data: mujoco.MjData, hand: str = "right") -> np.ndarray:
        """Returns the current 3D position of the specified hand's end effector."""
        mujoco.mj_forward(self.model, data)
        _, site_id, body_id, _, _, _ = self._resolve_arm(hand)
        if site_id != -1 and site_id < self.model.nsite:
            return data.site_xpos[site_id].copy()
        if body_id != -1 and body_id < self.model.nbody:
            return data.xpos[body_id].copy()
        return np.zeros(3)

    def solve_arm_ik(
        self,
        current_data: mujoco.MjData,
        target_pos: np.ndarray,
        hand: str = "right",
        damping: float = 0.05,
        max_iters: int = 80,
        tol: float = 0.008
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Solves IK and returns target angles in RADIANS for position actuators.
        """
        arm_joints, site_id, body_id, joint_ids, dof_addrs, q_addrs = self._resolve_arm(hand)
        self.scratch_data.qpos[:] = current_data.qpos[:]
        mujoco.mj_forward(self.model, self.scratch_data)

        target = np.array(target_pos, dtype=np.float64)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)

        for _ in range(max_iters):
            mujoco.mj_forward(self.model, self.scratch_data)

            if site_id != -1 and site_id < self.model.nsite:
                hand_pos = self.scratch_data.site_xpos[site_id]
                mujoco.mj_jacSite(self.model, self.scratch_data, jacp, jacr, site_id)
            elif body_id != -1 and body_id < self.model.nbody:
                hand_pos = self.scratch_data.xpos[body_id]
                mujoco.mj_jacBody(self.model, self.scratch_data, jacp, jacr, body_id)
            else:
                break

            error = target - hand_pos
            if np.linalg.norm(error) < tol:
                break

            # Sub-Jacobian for arm joints only
            J_arm = jacp[:, dof_addrs]
            A = J_arm @ J_arm.T + (damping ** 2) * np.eye(3)
            dq_arm = J_arm.T @ np.linalg.solve(A, error)

            # Apply step to arm joints
            for i, j_id in enumerate(joint_ids):
                if j_id != -1:
                    qpos_addr = q_addrs[i]
                    self.scratch_data.qpos[qpos_addr] += 0.4 * dq_arm[i]
                    
                    # Clamp to joint range
                    j_range = self.model.jnt_range[j_id]
                    self.scratch_data.qpos[qpos_addr] = np.clip(
                        self.scratch_data.qpos[qpos_addr], j_range[0], j_range[1]
                    )

        # Output target joint angles in Radians for MuJoCo position actuators
        target_angles = {}
        for name, j_id in zip(arm_joints, joint_ids):
            if j_id != -1:
                qpos_addr = self.model.jnt_qposadr[j_id]
                target_angles[name] = float(self.scratch_data.qpos[qpos_addr])

        final_hand = self.get_hand_pos(self.scratch_data, hand=hand)
        final_dist = np.linalg.norm(target - final_hand)
        return (final_dist < 0.12), target_angles

