"""
Dynamic Environment Disturbance Engine for Margaret Stress-Testing.
Allows perturbing and disturbing the physical environment at runtime:
- Displacing objects (e.g. human moves a prop 10cm to the side)
- Knocking objects over (tumbling/falling)
- Dropping objects to the floor
- Applying physical impulses and forces
- Emitting EventNotifications into Margaret's Adaptive Embodied Harness
"""

import time
import numpy as np
import mujoco
from typing import Optional, List, Dict, Any


class EnvironmentDisturber:
    """
    Disturbs the physical environment during active simulation to test
    Margaret's closed-loop perception, event arbiter, and replanning agility.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

    def get_all_prop_names(self) -> List[str]:
        """Returns all procedural prop names in the scene."""
        props = []
        for bid in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if name and name.startswith("prop_"):
                props.append(name)
        return props

    def resolve_target_name(self, query: str) -> Optional[str]:
        """Resolves natural language or partial names like 'red box' to 'prop_red_box_1'."""
        query_clean = query.strip().lower().replace(" ", "_")
        props = self.get_all_prop_names()
        # 1. Exact match
        for p in props:
            if p.lower() == query_clean:
                return p
        # 2. Substring match
        for p in props:
            if query_clean in p.lower():
                return p
        # 3. Token match (e.g. 'red box' matches 'prop_red_box_2')
        query_tokens = set(query.strip().lower().replace("-", " ").split())
        best_match, max_overlap = None, 0
        for p in props:
            p_tokens = set(p.lower().replace("_", " ").split())
            overlap = len(query_tokens.intersection(p_tokens))
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = p
        if max_overlap > 0:
            return best_match
        return None

    def displace_object(
        self,
        object_name: str,
        delta_x: float = 0.0,
        delta_y: float = 0.0,
        delta_z: float = 0.0,
        brain: Optional[Any] = None,
    ) -> bool:
        """
        Displaces an object by (dx, dy, dz) in physical simulation,
        simulating human intervention or table shaking.
        """
        resolved = self.resolve_target_name(object_name) or object_name
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, resolved)
        if body_id == -1:
            print(f"⚠️ [Disturber Error]: Body '{object_name}' (resolved: '{resolved}') not found.")
            return False
        object_name = resolved

        jnt_id = self.model.body_jntadr[body_id]
        if jnt_id == -1:
            return False

        qadr = self.model.jnt_qposadr[jnt_id]
        old_pos = self.data.qpos[qadr : qadr + 3].copy()
        self.data.qpos[qadr] += delta_x
        self.data.qpos[qadr + 1] += delta_y
        self.data.qpos[qadr + 2] += delta_z
        mujoco.mj_forward(self.model, self.data)
        new_pos = self.data.xpos[body_id].copy()

        print(f"💥 [Environment Disturbance]: '{object_name}' displaced from {old_pos.round(3)} to {new_pos.round(3)}!")

        # Post asynchronous event notification to Margaret
        if brain:
            brain.post_event({
                "event_type": "ENVIRONMENT_CHANGE",
                "description": f"Object '{object_name}' was physically displaced to [{new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f}]",
                "payload": {"object_name": object_name, "old_pos": old_pos.tolist(), "new_pos": new_pos.tolist()},
                "timestamp": time.time(),
            })
        return True

    def knock_over_object(
        self,
        object_name: str,
        angular_impulse: float = 2.5,
        brain: Optional[Any] = None,
    ) -> bool:
        """Knocks over an object by imparting rotational angular momentum."""
        resolved = self.resolve_target_name(object_name) or object_name
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, resolved)
        if body_id == -1:
            print(f"⚠️ [Disturber Error]: Body '{object_name}' (resolved: '{resolved}') not found.")
            return False
        object_name = resolved

        dof_adr = self.model.body_dofadr[body_id]
        if dof_adr == -1:
            return False

        # Apply angular velocity impulse around pitch and roll axes
        self.data.qvel[dof_adr + 3] += angular_impulse
        self.data.qvel[dof_adr + 4] += angular_impulse * 0.7
        mujoco.mj_forward(self.model, self.data)

        print(f"💥 [Environment Disturbance]: Knocked over '{object_name}' with impulse {angular_impulse} rad/s!")
        if brain:
            brain.post_event({
                "event_type": "ENVIRONMENT_CHANGE",
                "description": f"Object '{object_name}' was knocked over and tumbled",
                "payload": {"object_name": object_name, "angular_impulse": angular_impulse},
                "timestamp": time.time(),
            })
        return True

    def drop_object_off_table(
        self,
        object_name: str,
        brain: Optional[Any] = None,
    ) -> bool:
        """Pushes an object off the table so it falls onto the floor."""
        resolved = self.resolve_target_name(object_name) or object_name
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, resolved)
        if body_id == -1:
            print(f"⚠️ [Disturber Error]: Body '{object_name}' (resolved: '{resolved}') not found.")
            return False
        object_name = resolved

        jnt_id = self.model.body_jntadr[body_id]
        qadr = self.model.jnt_qposadr[jnt_id]
        # Move Y coordinate off table boundary (table width is ~0.35m)
        self.data.qpos[qadr + 1] = 0.55
        self.data.qpos[qadr + 2] = 0.05  # Floor level
        dof_adr = self.model.body_dofadr[body_id]
        if dof_adr != -1:
            self.data.qvel[dof_adr : dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

        print(f"💥 [Environment Disturbance]: Dropped '{object_name}' off table onto the floor!")
        if brain:
            brain.post_event({
                "event_type": "ENVIRONMENT_CHANGE",
                "description": f"Object '{object_name}' fell off the table onto the floor",
                "payload": {"object_name": object_name, "status": "on_floor"},
                "timestamp": time.time(),
            })
        return True

    def shake_all_props(
        self,
        jitter_std: float = 0.03,
        brain: Optional[Any] = None,
    ) -> int:
        """Simulates table tremor / bump by randomly displacing all free props."""
        count = 0
        for bid in range(self.model.nbody):
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if not bname or not bname.startswith("prop_"):
                continue
            jnt_id = self.model.body_jntadr[bid]
            if jnt_id == -1:
                continue
            qadr = self.model.jnt_qposadr[jnt_id]
            dx = float(np.random.normal(0, jitter_std))
            dy = float(np.random.normal(0, jitter_std))
            self.data.qpos[qadr] += dx
            self.data.qpos[qadr + 1] += dy
            count += 1

        mujoco.mj_forward(self.model, self.data)
        print(f"💥 [Environment Disturbance]: Table tremor perturbed {count} objects by ~{jitter_std*100:.1f}cm!")
        if brain:
            brain.post_event({
                "event_type": "ENVIRONMENT_CHANGE",
                "description": f"Table tremor shifted {count} props across workbench",
                "payload": {"props_affected": count, "magnitude": jitter_std},
                "timestamp": time.time(),
            })
        return count

