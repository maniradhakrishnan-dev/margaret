"""
Multi-Agent Embodied Architecture for Physical Robotics.
Separates concerns across three distinct agents:
1. FrontierIntentAgent: Frontier model that decomposes user natural language intent into high-level sub-goals.
2. LocalPerceptionAgent: Live sensor/camera perception agent that grounds and produces real-time environment feedback.
3. ExecutiveActionAgent: Robot command controller that bridges decomposed intent with grounded environment feedback.

NO HARDCODED USER INTENTS. NO PRECODED ENVIRONMENTS.
"""

import os
import json
import time
from typing import Any, Dict, List, Optional

from agent.schemas import (
    TaskDecompositionPlan,
    ActionDecision,
    SceneEntity,
)
from agent.tools import ToolRegistry
from agent.memory import SpatialMemory, EpisodicHistory
from agent.world_model import WorldModel


class FrontierIntentAgent:
    """
    Frontier Model Agent: Responsible for high-level semantic task decomposition.
    Decomposes arbitrary user natural language instructions into verified sub-goals
    WITHOUT any hardcoded prop names, synonyms, or pre-assumed environments.
    """

    def __init__(self, graph_caller):
        self.caller = graph_caller

    def decompose(self, user_goal: str, environment_summary: Optional[str] = None) -> Optional[TaskDecompositionPlan]:
        """
        Decomposes arbitrary user goals into structured physical sub-goals.
        Grounds purely against the live environment summary provided by the local perception agent.
        """
        system_instruction = """You are the Frontier Intent Decomposition Agent for an autonomous robot operating in physical space.
Your role is to understand arbitrary natural language user commands, extract the underlying physical intent,
and decompose the command into a sequenced list of physical sub-goals (e.g. inspect, reach, grasp, lift, place, release, recover).

CRITICAL CONSTRAINTS:
- Do NOT assume any hardcoded objects, colors, or environments.
- Ground your analysis against whichever entities and environmental feedback are provided.
- If the user refers to an object by visual description (e.g. 'the red one', 'the cylindrical container', 'the blue block'),
  bind it to the best matching entity reported in the environmental feedback.
- If an object is already held by the robot, plan placing/moving/releasing it directly without redundant grasping.
- Always output strictly according to the TaskDecompositionPlan schema."""

        prompt = f"""Natural Language User Instruction: "{user_goal}"

Current Grounded Environment Feedback:
{environment_summary or "No objects perceived yet. Initial scene inspection required."}

Decompose this instruction into a verified physical task plan."""

        raw_json = self.caller._call_gemini_pydantic(
            prompt, system_instruction, TaskDecompositionPlan
        )

        if raw_json:
            try:
                return TaskDecompositionPlan.model_validate(raw_json)
            except Exception as e:
                print(f"❌ [FrontierIntentAgent Validation Error]: {e}")
                return None

        print("❌ [FrontierIntentAgent Error]: Model offline or unavailable. Physical execution halted.")
        return None


class LocalPerceptionAgent:
    """
    Local Perception Agent: Responsible for continuous sensory perception and feedback.
    Senses the physical environment via camera sensors, depth deprojection, and proprioception.
    Produces live Environment Feedback for the agent without hardcoding.
    """

    def __init__(self, tools: ToolRegistry, memory: SpatialMemory):
        self.tools = tools
        self.memory = memory

    def update_and_give_feedback(self) -> Dict[str, Any]:
        """
        Executes live scene inspection, updates spatial memory, and formats structured
        Environment Feedback for the reasoning and executive agents.
        """
        inspection_result = {}
        if "inspect_scene" in self.tools.list_tools():
            res = self.tools.execute("inspect_scene", {})
            inspection_result = res.get("result", {})

        # Build grounded environment feedback
        objects_feedback = []
        for name, data in self.memory.objects.items():
            objects_feedback.append({
                "name": name,
                "position_3d": data.get("pos"),
                "status": data.get("status", "unknown"),
                "shape": data.get("category", "object"),
                "color": data.get("color", "unknown"),
            })

        robot_state = {
            "right_hand": self.memory.robot_state.get("right_hand", {}),
            "left_hand": self.memory.robot_state.get("left_hand", {}),
        }

        # Human-readable summary for model prompt
        summary_lines = [
            f"Perceived Physical Entities in Environment ({len(objects_feedback)} total):"
        ]
        for obj in objects_feedback:
            pos_str = f"[{obj['position_3d'][0]:.3f}, {obj['position_3d'][1]:.3f}, {obj['position_3d'][2]:.3f}]" if obj.get('position_3d') else "unknown"
            summary_lines.append(
                f"  • Entity '{obj['name']}': shape={obj['shape']}, color={obj['color']}, position={pos_str}, status={obj['status']}"
            )

        summary_lines.append("\nRobot Embodiment State:")
        rh_held = robot_state["right_hand"].get("held") or "empty"
        lh_held = robot_state["left_hand"].get("held") or "empty"
        summary_lines.append(f"  • Right Hand: held={rh_held}, reached={robot_state['right_hand'].get('reached')}")
        summary_lines.append(f"  • Left Hand:  held={lh_held}, reached={robot_state['left_hand'].get('reached')}")

        return {
            "objects": objects_feedback,
            "robot_state": robot_state,
            "summary_text": "\n".join(summary_lines),
            "raw_inspection": inspection_result,
        }


class ExecutiveActionAgent:
    """
    Executive Action Agent: Responsible for deciding and dispatching physical robot commands.
    Bridges the high-level plan from the Frontier Intent Agent and the live sensory feedback
    from the Local Perception Agent.
    """

    def __init__(self, graph):
        self.graph = graph

    def decide_next_action(
        self,
        step_idx: int,
        plan: TaskDecompositionPlan,
        environment_feedback: Dict[str, Any],
    ) -> Optional[ActionDecision]:
        """
        Prompts the embodied reasoning engine to select the next physical robot action
        grounded in the live environment feedback and active plan.
        """
        return self.graph.node_embodied_reasoning(step_idx)

    def execute_action(self, decision: ActionDecision) -> Dict[str, Any]:
        """
        Dispatches the action decision through verified robot controllers.
        """
        # Delegated directly to graph tool dispatcher with physics safety checks
        return self.graph._dispatch_action(decision)
