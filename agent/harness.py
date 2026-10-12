"""
Adaptive Embodied Agentic Harness for Physical Robotics.
Ingests multi-source streams:
1. User Stream: Natural language goals, instructions, asynchronous interrupts.
2. Environment Stream: Grounded 3D scene objects, visual attributes, spatial changes.
3. Robot Description & Joint Stream: Kinematics, degrees of freedom, joint limits, live telemetry.
4. Extensible Tool/Skill Stream: Dynamically introspected skills from ToolRegistry.

Enforces strict Pydantic schemas to eliminate LLM hallucinations and produce verified
motor commands on the robot actuators with closed-loop physical feedback.
"""

import os
import json
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple, Union
import mujoco
import numpy as np

from agent.schemas import (
    TaskDecompositionPlan,
    ActionDecision,
    EventNotification,
    EventDecision,
    ActionFeedback,
    to_gemini_schema,
)
from agent.tools import ToolRegistry
from agent.memory import SpatialMemory, EpisodicHistory
from robots.base_robot import BaseRobot


def load_project_env():
    """Loads .env file into os.environ if present."""
    env_path = os.path.abspath(".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    val = v.strip("\"' ")
                    os.environ[k.strip()] = val


class AdaptiveEmbodiedHarness:
    """
    Unified Adaptive Embodied Harness for Autonomous Robots.
    Coordinates User, Environment, Robot Embodiment, and Event Streams.
    """

    def __init__(
        self,
        tools: ToolRegistry,
        memory: SpatialMemory,
        history: EpisodicHistory,
        robot: Optional[BaseRobot] = None,
        model_physics: Optional[mujoco.MjModel] = None,
        data_physics: Optional[mujoco.MjData] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        load_project_env()
        self.tools = tools
        self.memory = memory
        self.history = history
        self.robot = robot
        self.model_physics = model_physics
        self.data_physics = data_physics
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite")

        # Asynchronous event stream queue
        self._event_queue: List[EventNotification] = []
        self.active_plan: Optional[TaskDecompositionPlan] = None
        self.current_goal: str = ""

    # =========================================================================
    # EVENT & INTERRUPT MANAGEMENT
    # =========================================================================
    def post_event(self, event: Union[EventNotification, Dict[str, Any]]) -> None:
        """Enqueues an asynchronous sensory or user event (e.g. human interrupt, scene shift, fault)."""
        if isinstance(event, dict):
            event = EventNotification.model_validate(event)
        if event.timestamp == 0.0:
            event.timestamp = time.time()
        self._event_queue.append(event)
        print(f"🔔 [Event Posted]: {event.event_type} - {event.description}")

        # Synchronize spatial memory with event payload immediately
        if event.event_type == "ENVIRONMENT_CHANGE" and event.payload:
            obj_name = event.payload.get("object_name")
            new_pos = event.payload.get("new_pos")
            if obj_name and new_pos and self.memory:
                self.memory.update_object(obj_name, new_pos)
            # If object was physically moved by user / disturbance, clear held state
            if obj_name and self.memory:
                for h in ["right_hand", "left_hand"]:
                    if self.memory.robot_state.get(h, {}).get("held") == obj_name:
                        self.memory.robot_state[h]["held"] = None
                        self.memory.robot_state[h]["lifted"] = False
                if self.memory.robot_state.get("held_object") == obj_name:
                    self.memory.robot_state["held_object"] = None

    def has_pending_events(self) -> bool:
        return len(self._event_queue) > 0

    def get_pending_events(self) -> List[EventNotification]:
        return list(self._event_queue)

    def clear_events(self) -> None:
        self._event_queue.clear()

    # =========================================================================
    # MULTI-SOURCE EMBODIED CONTEXT INGESTION
    # =========================================================================
    def get_embodied_context(self) -> Dict[str, Any]:
        """
        Synthesizes the complete multi-source state:
        - User Goal & Pending Events
        - Live Environment Scene Graph
        - Robot Description & Joint Telemetry
        - Dynamically Available Skills
        """
        # 1. Environment Stream
        env_objects = {}
        for name, data in self.memory.objects.items():
            meta = data.get("metadata", {})
            env_objects[name] = {
                "position_3d": data.get("pos"),
                "shape": meta.get("category", data.get("category", "object")),
                "color": meta.get("color", data.get("color", "unknown")),
                "status": meta.get("status", data.get("status", "on_table")),
            }

        # 2. Robot Description Stream
        if self.robot:
            robot_desc = {
                "platform": self.robot.name,
                "table_surface_z": self.robot.table_surface_z,
                "floor_threshold_z": self.robot.floor_threshold_z,
                "max_reach_m": self.robot.max_reach_m,
                "staging_pos": self.robot.staging_pos,
                "arms": ["right", "left"],
                "right_end_effector": self.robot.get_end_effector("right"),
                "left_end_effector": self.robot.get_end_effector("left"),
            }
        else:
            robot_desc = {
                "platform": "generic_robot",
                "table_surface_z": 0.62,
                "floor_threshold_z": 0.47,
                "max_reach_m": 0.55,
                "staging_pos": [0.32, 0.0, 0.62],
            }

        # 3. Robot Joint Telemetry Stream
        joint_telemetry = {}
        if self.model_physics and self.data_physics and self.robot:
            for hand in ["right", "left"]:
                joints = self.robot.get_arm_joints(hand)
                for jname in joints:
                    jid = mujoco.mj_name2id(self.model_physics, mujoco.mjtObj.mjOBJ_JOINT, jname)
                    if jid != -1:
                        qadr = self.model_physics.jnt_qposadr[jid]
                        vadr = self.model_physics.jnt_dofadr[jid]
                        joint_telemetry[jname] = {
                            "q_rad": round(float(self.data_physics.qpos[qadr]), 3),
                            "v_rad_s": round(float(self.data_physics.qvel[vadr]), 3),
                        }

        # 4. Dynamically Introspected Tool / Skill Stream
        available_tools = []
        for tname in self.tools.list_tools():
            tool_obj = self.tools._tools.get(tname)
            desc = tool_obj.description if tool_obj else ""
            available_tools.append(f"{tname}: {desc}")

        # 5. Pending Events
        events_summary = [
            f"[{e.event_type}]: {e.description}" for e in self._event_queue
        ]

        return {
            "current_goal": self.current_goal,
            "environment_objects": env_objects,
            "robot_state": self.memory.robot_state,
            "robot_description": robot_desc,
            "joint_telemetry": joint_telemetry,
            "available_tools": available_tools,
            "pending_events": events_summary,
        }

    # =========================================================================
    # REASONING & SCHEMA VALIDATION VIA GEMINI
    # =========================================================================
    def _call_gemini_pydantic(
        self,
        prompt: str,
        system_instruction: str,
        response_schema: Any,
        model_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Invokes Gemini with strict Pydantic responseSchema to prevent hallucinations."""
        if not self.api_key:
            return None

        target_model = (model_name or self.model).replace("models/", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={self.api_key}"

        clean_schema = to_gemini_schema(response_schema)
        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
                "responseSchema": clean_schema,
            },
        }

        def _safe_serialize(val):
            if hasattr(val, "tolist"):
                return val.tolist()
            if hasattr(val, "item"):
                return val.item()
            return str(val)

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload, default=_safe_serialize).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text)
        except Exception as e:
            if target_model != self.fallback_model:
                time.sleep(0.3)
                return self._call_gemini_pydantic(
                    prompt, system_instruction, response_schema, model_name=self.fallback_model
                )
            return None

    # =========================================================================
    # ASYNCHRONOUS EVENT EVALUATION
    # =========================================================================
    def evaluate_events(self) -> EventDecision:
        """
        Evaluates pending events against active goal and determines whether the robot
        can safely continue, needs to replan, or must halt immediately.
        """
        if not self._event_queue:
            return EventDecision(thought="No pending events.", decision="CONTINUE", reason="Normal execution.")

        context = self.get_embodied_context()
        system_prompt = """You are the Margaret Event Arbiter.
Your job is to analyze unexpected events (user interrupts, scene shifts, robot joint limits/faults, or new tools)
occurring during active robotic manipulation and decide how the robot must react.

Decision Options:
- CONTINUE: Event is informational or does not invalidate active physical sub-goals.
- REPLAN: The environment or user intent has changed such that current sub-goals must be regenerated.
- HALT_SAFETY: A dangerous condition, physical conflict, or critical user abort command was issued."""

        prompt = f"""Current Mission Goal: "{self.current_goal}"
Active Sub-Goals: {[sg.model_dump() for sg in self.active_plan.sub_goals] if self.active_plan else []}
Pending Events:
{json.dumps(context['pending_events'], indent=2)}

Embodied Environment State:
{json.dumps(context['environment_objects'], indent=2)}

Decide how Margaret should respond."""

        raw = self._call_gemini_pydantic(prompt, system_prompt, EventDecision)
        self.clear_events()
        if raw:
            try:
                return EventDecision.model_validate(raw)
            except Exception:
                pass
        return EventDecision(thought="Default safety policy.", decision="HALT_SAFETY", reason="Failed to evaluate event.")

    # =========================================================================
    # TASK DECOMPOSITION (FRONTIER MODEL)
    # =========================================================================
    def decompose_task(self, goal_text: str) -> Optional[TaskDecompositionPlan]:
        """Decomposes arbitrary user goals dynamically with multi-source awareness."""
        self.current_goal = goal_text
        context = self.get_embodied_context()

        system_prompt = f"""You are Margaret's Cognitive Task Decomposition Engine.
Robot Embodiment:
{json.dumps(context['robot_description'], indent=2)}

Available Tools / Skills:
{json.dumps(context['available_tools'], indent=2)}

Analyze the user's natural language goal and decompose it into an ordered sequence of physical sub-goals.
Do NOT assume hardcoded objects or environments; ground purely in perceived entities."""

        prompt = f"""Natural Language User Instruction: "{goal_text}"
Perceived Environment Objects:
{json.dumps(context['environment_objects'], indent=2)}
Robot State:
{json.dumps(context['robot_state'], indent=2)}"""

        raw = self._call_gemini_pydantic(prompt, system_prompt, TaskDecompositionPlan)
        if raw:
            try:
                self.active_plan = TaskDecompositionPlan.model_validate(raw)
                return self.active_plan
            except Exception as e:
                print(f"❌ [Harness Decomposition Validation Error]: {e}")
                return None
        return None

    # =========================================================================
    # CLOSED-LOOP PHYSICAL ACTION EXECUTION & FEEDBACK
    # =========================================================================
    def execute_physical_decision(self, decision: ActionDecision) -> ActionFeedback:
        """
        Translates validated ActionDecision into actual motor commands via tool registry,
        capturing physical execution feedback without hallucination.
        """
        action_name = decision.action
        if action_name == "mission_complete":
            return ActionFeedback(success=True, action="mission_complete")
        if action_name == "abort":
            return ActionFeedback(success=False, action="abort", error="Aborted by agent reasoning.")

        # Map schema parameters to tool arguments
        args = {}
        if action_name == "reach_hand":
            args = {"target_x": decision.target_x, "target_y": decision.target_y, "target_z": decision.target_z, "hand": decision.arm}
        elif action_name == "grasp_object":
            target_obj = decision.object_name or (self.active_plan.target_object if self.active_plan else None)
            args = {"object_name": target_obj, "hand": decision.arm}
        elif action_name == "lift_hand":
            args = {"height": decision.lift_height or 0.15, "hand": decision.arm}
        elif action_name == "place_object":
            args = {"target_x": decision.target_x, "target_y": decision.target_y, "target_z": decision.target_z, "hand": decision.arm, "dest_object": decision.dest_object}
        elif action_name == "release_object":
            args = {"object_name": decision.object_name, "hand": decision.arm}
        elif action_name == "recover_floor_object":
            target_obj = decision.object_name or (self.active_plan.target_object if self.active_plan else None)
            args = {"object_name": target_obj, "target_x": decision.target_x, "target_y": decision.target_y}
        elif action_name == "inspect_scene":
            args = {}

        # Actuate through registered physical tool
        res = self.tools.execute(action_name, args)
        success = res.get("success", False)
        tool_res = res.get("result", {})
        err = res.get("error") or (tool_res.get("error") if isinstance(tool_res, dict) else None)

        reached_pos = None
        dist_err = None
        grip_force = None
        if isinstance(tool_res, dict):
            reached_pos = tool_res.get("reached_position")
            dist_err = tool_res.get("distance_error_cm")
            grip_force = tool_res.get("measured_grip_force_N")

        return ActionFeedback(
            success=bool(success and not err),
            action=action_name,
            reached_position=reached_pos,
            distance_error_cm=dist_err,
            measured_grip_force_N=grip_force,
            error=str(err) if err else None,
        )


# Backward compatibility alias
AgenticLoopHarness = AdaptiveEmbodiedHarness
