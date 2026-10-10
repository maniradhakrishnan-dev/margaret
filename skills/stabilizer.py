"""
Virtual Test Rig & Posture Stabilizer
Keeps the humanoid torso upright in front of the workbench during manipulation.
"""

import mujoco
import numpy as np

class PostureStabilizer:
    def __init__(self, model: mujoco.MjModel, target_pos=(0.0, 0.0, 1.26)):
        self.model = model
        self.target_pos = np.array(target_pos, dtype=np.float64)
        self.torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        
        # Smooth, critically damped spring-damper gains
        self.kp_pos = 350.0  # N/m
        self.kd_pos = 45.0   # Ns/m
        self.kp_rot = 120.0  # Nm/rad
        self.kd_rot = 15.0   # Nms/rad

    def apply_standing_hold(self, data: mujoco.MjData):
        """Applies gentle stabilizing forces to keep the robot upright facing the table."""
        if self.torso_body_id == -1:
            return

        # Current torso position and velocity
        torso_pos = data.xpos[self.torso_body_id]
        torso_vel = data.cvel[self.torso_body_id][:3]

        # Desired position restoration force (keeps feet firmly planted on the floor)
        pos_error = self.target_pos - torso_pos
        force = self.kp_pos * pos_error - self.kd_pos * torso_vel

        # Upright orientation restoration
        q = data.qpos[3:7]
        rot_error = np.array([-q[1], -q[2], -q[3]])
        ang_vel = data.cvel[self.torso_body_id][3:]
        torque = self.kp_rot * rot_error - self.kd_rot * ang_vel

        # Apply external force and torque directly to the torso body
        data.xfrc_applied[self.torso_body_id, :3] = force
        data.xfrc_applied[self.torso_body_id, 3:] = torque
