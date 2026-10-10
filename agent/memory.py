"""
Spatial & Episodic Memory for the Embodied Agent
Maintains scene graphs, perceived 3D objects, and execution history.
"""

import time
from typing import Any, Dict, List, Optional

class SpatialMemory:
    def __init__(self):
        # Maps object_name -> {"pos": [x, y, z], "last_seen": timestamp, ...}
        self.objects: Dict[str, Dict[str, Any]] = {}
        self.robot_state: Dict[str, Any] = {
            "right_hand": {
                "held": None,
                "reached_object": None,
                "pos": [0.0, -0.22, 0.60],
                "lifted": False
            },
            "left_hand": {
                "held": None,
                "reached_object": None,
                "pos": [0.0, 0.22, 0.60],
                "lifted": False
            },
            # Top-level backwards compatibility
            "held_object": None,
            "reached": False,
            "reached_object": None,
            "lifted": False,
            "is_stable": True
        }

    def set_hand_held(self, hand: str, obj_name: Optional[str]):
        h = "right_hand" if hand.lower() == "right" else "left_hand"
        self.robot_state[h]["held"] = obj_name
        self.robot_state["held_object"] = self.robot_state["right_hand"]["held"] or self.robot_state["left_hand"]["held"]

    def set_hand_reached(self, hand: str, obj_name: Optional[str]):
        h = "right_hand" if hand.lower() == "right" else "left_hand"
        self.robot_state[h]["reached_object"] = obj_name
        self.robot_state["reached"] = bool(obj_name)
        self.robot_state["reached_object"] = obj_name

    def set_hand_lifted(self, hand: str, lifted: bool = True):
        h = "right_hand" if hand.lower() == "right" else "left_hand"
        self.robot_state[h]["lifted"] = lifted
        self.robot_state["lifted"] = lifted

    def update_object(self, name: str, pos: List[float], metadata: Optional[Dict] = None):
        self.objects[name] = {
            "pos": list(pos),
            "last_seen": time.time(),
            "metadata": metadata or {}
        }

    def get_object(self, name: str) -> Optional[Dict[str, Any]]:
        return self.objects.get(name)

    def list_objects(self) -> List[Dict[str, Any]]:
        return [{"name": k, **v} for k, v in self.objects.items()]

    def update_hand_pos(self, hand: str, pos: List[float]):
        h = "right_hand" if hand.lower() == "right" else "left_hand"
        self.robot_state[h]["pos"] = [round(float(p), 3) for p in pos]
        held = self.robot_state[h].get("held")
        if held and held in self.objects:
            self.objects[held]["pos"] = [round(float(p), 3) for p in pos]

    def format_scene_summary(self) -> str:
        if not self.objects:
            return "No objects detected in memory yet."
        lines = ["Perceived 3D Objects in Room:"]
        for name, data in self.objects.items():
            pos = data["pos"]
            lines.append(f"  • {name}: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")
        return "\n".join(lines)

    def format_status_summary(self) -> str:
        rh = self.robot_state.get("right_hand", {})
        lh = self.robot_state.get("left_hand", {})
        lines = [
            f"Right Hand: held={rh.get('held')}, reached={rh.get('reached_object')}, lifted={rh.get('lifted')}, pos={rh.get('pos')}",
            f"Left Hand: held={lh.get('held')}, reached={lh.get('reached_object')}, lifted={lh.get('lifted')}, pos={lh.get('pos')}"
        ]
        return "\n".join(lines)

class EpisodicHistory:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.current_mission_history: List[Dict[str, Any]] = []
        self.current_goal: Optional[str] = None

    def start_new_mission(self, user_goal: str):
        """Initializes a fresh history buffer for the new interactive mission."""
        self.current_goal = user_goal
        self.current_mission_history = []

    def record_step(self, thought: str, action_name: str, args: Dict[str, Any], observation: Any):
        entry = {
            "timestamp": time.time(),
            "thought": thought,
            "action": action_name,
            "args": args,
            "observation": observation
        }
        self.history.append(entry)
        self.current_mission_history.append(entry)

    def get_recent_history(self, n: int = 5) -> List[Dict[str, Any]]:
        return self.history[-n:]

    def get_current_mission_history(self, n: int = 5) -> List[Dict[str, Any]]:
        return self.current_mission_history[-n:]
