"""
Pure-Python Cognitive ReAct Agent Brain (Built from Scratch)
Executes Reason -> Act -> Observe loops for Physical Embodied AI.
"""

import os
import json
import time
from typing import Any, Dict, List, Optional
from agent.tools import ToolRegistry
from agent.memory import SpatialMemory, EpisodicHistory
from agent.world_model import WorldModel
from agent.harness import AgenticLoopHarness, load_project_env
from agent.graph import LiveEmbodiedGraph
from agent.schemas import TaskDecompositionPlan

SYSTEM_PROMPT = """You are Margaret, the Cognitive Embodied AI Brain for an autonomous robot operating in a 3D physical world.
Your job is to solve high-level tasks by decomposing them into physical actions and observing sensory feedback.
"""

class AgentBrain:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        world_model: Optional[WorldModel] = None,
        api_key: Optional[str] = None,
    ):
        load_project_env()
        self.name = "Margaret"
        self.tools = tool_registry
        self.world_model = world_model or WorldModel()
        self.memory = SpatialMemory()
        self.history = EpisodicHistory()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.graph = LiveEmbodiedGraph(
            self.tools,
            self.memory,
            self.history,
            world_model=self.world_model,
            api_key=self.api_key,
        )
        self.harness = self.graph  # Backward compatibility
        self.frontier_agent = self.graph.frontier_agent
        self.perception_agent = self.graph.perception_agent
        self.executive_agent = self.graph.executive_agent

    def run_mission(self, user_goal: str, max_steps: int = 10) -> Dict[str, Any]:
        """Runs the multi-agent on-demand Live Embodied Graph orchestrator to solve user's goal."""
        return self.graph.execute_live_graph(user_goal, max_steps=max_steps)

    def decompose_intent(self, user_goal: str) -> Optional[TaskDecompositionPlan]:
        """Frontier model decomposes arbitrary user natural language intent into structured physical sub-goals."""
        return self.frontier_agent.decompose(user_goal)

    def post_event(self, event: Any) -> None:
        """Posts an external event (human interrupt, scene shift, hardware alert) to Margaret."""
        self.graph.post_event(event)
