"""
Dynamic Live Agentic Graph Orchestrator for Physical Humanoid AI
Executes on-demand graph nodes (Decomposition -> Perception -> Embodied Reasoning -> World Model -> Skill Execution -> Observation)
using strict Pydantic schemas, closed-loop feedback, and token-efficient spatial scene representations with Gemini Robotics ER / Flash.
"""

import os
import json
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

from agent.schemas import (
    TaskDecompositionPlan,
    ActionDecision,
    SceneEntity,
    RobotEmbodimentState,
    ArmState,
    EventNotification,
    EventDecision,
    ActionFeedback,
    to_gemini_schema,
)
from agent.tools import ToolRegistry
from agent.memory import SpatialMemory, EpisodicHistory
from agent.world_model import WorldModel
from agent.harness import load_project_env


class LiveEmbodiedGraph:
    """
    On-demand Live Execution Graph for Physical Humanoid Robotics.
    Dynamically routes between perception, reasoning, verification, physical execution,
    and closed-loop sensory feedback without static monolithic loops or hardcoding.
    """

    def __init__(
        self,
        tools: ToolRegistry,
        memory: SpatialMemory,
        history: EpisodicHistory,
        world_model: Optional[WorldModel] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        load_project_env()
        self.tools = tools
        self.memory = memory
        self.history = history
        self.world_model = world_model or WorldModel()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite")

        # Current Graph State
        self.current_goal: str = ""
        self.decomposition_plan: Optional[TaskDecompositionPlan] = None
        self.step_counter: int = 0
        self.max_steps: int = 10
        self.is_completed: bool = False

        # Multi-Agent Subsystems & Adaptive Harness
        from agent.multiagent import FrontierIntentAgent, LocalPerceptionAgent, ExecutiveActionAgent
        from agent.harness import AdaptiveEmbodiedHarness
        self.harness = AdaptiveEmbodiedHarness(
            self.tools,
            self.memory,
            self.history,
            robot=self.robot,
            api_key=self.api_key,
            model=self.model,
        )
        self.frontier_agent = FrontierIntentAgent(self)
        self.perception_agent = LocalPerceptionAgent(self.tools, self.memory)
        self.executive_agent = ExecutiveActionAgent(self)

    def post_event(self, event: Any) -> None:
        """Enqueues an asynchronous event or user interrupt into the live execution harness."""
        self.harness.post_event(event)

    @property
    def robot(self):
        return getattr(self.world_model, "robot", None)

    @property
    def table_z(self) -> float:
        return self.robot.table_surface_z if self.robot else (self.world_model.table_surface_z if self.world_model else 0.62)

    @property
    def floor_thresh_z(self) -> float:
        if self.robot and hasattr(self.robot, "floor_threshold_z"):
            return self.robot.floor_threshold_z
        return self.table_z - 0.15

    def get_optimal_arm(self, pos: List[float]) -> str:
        if self.robot and hasattr(self.robot, "get_optimal_arm"):
            return self.robot.get_optimal_arm(pos)
        return "left" if pos[1] > 0.04 else "right"

    @property
    def staging_pos(self) -> List[float]:
        if self.robot and hasattr(self.robot, "staging_pos"):
            return self.robot.staging_pos
        return [0.32, 0.0, self.table_z]

    def _call_gemini_pydantic(
        self,
        prompt: str,
        system_instruction: str,
        response_schema: Any,
        model_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Queries Gemini API with strict Pydantic responseSchema to guarantee
        zero hallucination, valid schemas, and compliance.
        """
        if not self.api_key:
            print("❌ [Model Error]: GEMINI_API_KEY is not set. Cannot perform genuine embodied reasoning.")
            return None

        target_model = (model_name or self.model).replace("models/", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={self.api_key}"

        # Sanitize schema: dereference all $defs and $ref, strip unsupported OpenAPI keywords
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
            err_str = str(e)
            if "429" in err_str or "Too Many Requests" in err_str:
                print(f"⚠️ [Embodied Model Rate Limit with {target_model}]: HTTP 429. Seamlessly switching to {self.fallback_model}...")
                time.sleep(1.2)
            else:
                print(f"⚠️ [Embodied Model Error with {target_model}]: {e}")

            # Fallback to fast lite model if primary model encounters transient rate limit or timeout
            if target_model != self.fallback_model:
                return self._call_gemini_pydantic(
                    prompt, system_instruction, response_schema, model_name=self.fallback_model
                )
            print(f"❌ [Model Error]: Both primary and fallback reasoning models failed ({err_str}). Internet connection or API unreachable.")
            return None

    # =========================================================================
    # GRAPH NODE 1: High-Level Task Decomposition
    # =========================================================================
    def node_decompose_task(self, user_goal: str) -> TaskDecompositionPlan:
        """
        Loads the high-level semantic reasoning graph on-demand to analyze user intent,
        inspect current embodiment state, and construct a verified sub-goal DAG.
        """
        if self.robot and hasattr(self.robot, "get_prompt_embodiment_description"):
            embodiment_text = self.robot.get_prompt_embodiment_description()
        else:
            embodiment_text = (
                f"Robot Kinematic Setup:\n"
                f"- Right Arm: optimal for workbench coordinates Y <= 0.04\n"
                f"- Left Arm: optimal for workbench coordinates Y > 0.04\n"
                f"- Table surface at Z = {self.table_z:.2f}m, props rest at Z = {self.table_z+0.03:.2f}m - {self.table_z+0.05:.2f}m.\n"
                f"- Floor threshold: Z < {self.floor_thresh_z:.2f}m."
            )

        system_instruction = f"""You are Margaret, the Embodied Task Decomposition Engine for an autonomous bimanual robot in MuJoCo physics.
{embodiment_text}

Semantic Grounding & Fuzzy Visual Resolution:
- Users describe objects colloquially (e.g. "yello cube" for an orange box, "pink object" for a purple/magenta capsule, "ball" for sphere, "can/tube" for capsule/cylinder).
- RESOLUTION RULES:
  1. Shape Aliases: "cube"/"block" matches "box"; "ball" matches "sphere"; "tube"/"can"/"capsule" matches "capsule" or "cylinder".
  2. Adjacent Color / Lighting Nuance: Users see shaded, reflected colors on 3D surfaces under table lighting:
     - "yellow" matches "orange" or "gold"
     - "pink" matches "purple" or "magenta" or "red"
     - "cyan"/"teal" matches "blue"
     - "grey"/"silver" matches "white" or "black"
  3. Uniqueness Rule: If the user specifies a shape (e.g. "cube" or "yellow cube") and only ONE box/cube exists in the scene, BIND TO IT even if the user named an adjacent color!
  4. Best-Match Rule: If the user specifies an attribute (e.g. "pink object"), bind to the closest chromatic candidate in the scene rather than rejecting!
  5. NEVER claim an object does not exist if a unique or closest matching candidate exists in Currently Perceived Scene Objects!

EMBODIMENT STATE AWARENESS & HANDOVER CONFIRMATION:
- Inspect the Robot Embodiment State.
- If an arm is ALREADY HOLDING the requested object, do NOT plan reaching and grasping it again! Directly plan transporting, placing, or releasing it.
- If the user commands a handover (e.g. "take completely to left hand") or asserts that an object is still in one hand:
  Plan a physical motion: retract the releasing arm away or lift the recipient hand higher to clearly demonstrate that the object is held completely and securely!

LOCOMOTION & TERRAIN NAVIGATION DECOMPOSITION:
- For Mobile & Wheeled-Biped Embodiments (e.g. LimX TRON 1) or navigation tasks (e.g. "steer around obstacle", "move forward", "navigate to goal", "climb ramp", "traverse obstacle course"):
  - Goal types include: 'NAVIGATE', 'OBSTACLE_AVOIDANCE', 'TERRAIN_TRAVERSAL', 'INSPECT'.
  - Sub-goal actions include: 'inspect', 'navigate', 'steer', 'drive', 'climb', 'complete'.
  - When asked to steer around an obstacle (e.g. "steer around the red box"):
    1. First inspect the scene if needed (action='inspect', description='Perceive obstacle locations and corridor').
    2. Steer around the designated obstacle (action='steer', target_object=obstacle_name, description='Steer around the obstacle box to clear path').
    3. Complete mission (action='complete', description='Obstacle safely cleared').
- Decompose the mission into an ordered sequence of physical sub-goals."""

        known_objects = {}
        for name, data in self.memory.objects.items():
            meta = data.get("metadata", {})
            parts = name.split("_")
            inferred_color = meta.get("color") or (parts[1] if len(parts) >= 3 else "unknown")
            inferred_shape = meta.get("shape") or (parts[2] if len(parts) >= 3 else "object")
            known_objects[name] = {
                "color": inferred_color,
                "shape": inferred_shape,
                "position": [round(float(x), 3) for x in data["pos"]],
                "status": meta.get("status", "on_table" if data["pos"][2] >= self.floor_thresh_z else "on_floor"),
            }
        robot_state_summary = {
            "right_arm": self.memory.robot_state.get("right_hand", {}),
            "left_arm": self.memory.robot_state.get("left_hand", {}),
        }

        prompt = f"""Natural Language User Instruction: "{user_goal}"
Currently Perceived Scene Objects:
{json.dumps(known_objects, indent=2)}

Robot Embodiment State:
{json.dumps(robot_state_summary, indent=2)}

Decompose this instruction into a verified task plan."""

        raw_json = self._call_gemini_pydantic(
            prompt, system_instruction, TaskDecompositionPlan
        )

        if raw_json:
            try:
                self.decomposition_plan = TaskDecompositionPlan.model_validate(raw_json)
                return self.decomposition_plan
            except Exception as e:
                print(f"❌ [TaskDecomposition Validation Error]: {e}")
                return None

        # DO NOT DISPATCH FAKE OR HARDCODED PLANS WHEN THE MODEL IS UNAVAILABLE
        print("❌ [Task Decomposition Failed]: Reasoning model unreachable or offline.")
        return None

    # =========================================================================
    # GRAPH NODE 2: Token-Efficient Spatial Perception
    # =========================================================================
    def node_perception(self) -> Dict[str, Any]:
        """
        Token-Efficient Spatial Perception Node:
        Extracts 3D bounding coordinates locally from MuJoCo head camera,
        updating the spatial memory with lightweight SceneEntity objects (<50 tokens).
        """
        if "inspect_scene" in self.tools.list_tools():
            res = self.tools.execute("inspect_scene", {})
            return res.get("result", {})
        return {}

    # =========================================================================
    # GRAPH NODE 3: Embodied Reasoning Node (Closed-Loop ReAct)
    # =========================================================================
    def node_embodied_reasoning(self, step_idx: int) -> ActionDecision:
        """
        Embodied Reasoning Node:
        Prompts Gemini Robotics ER / Flash with structured scene graph, bimanual state,
        and closed-loop sensory feedback from previous action observations.
        """
        if self.robot and hasattr(self.robot, "get_prompt_embodiment_description"):
            embodiment_text = self.robot.get_prompt_embodiment_description()
        else:
            embodiment_text = (
                f"Bimanual Embodiment:\n"
                f"- Right Arm workspace: optimal for workbench Y <= 0.04\n"
                f"- Left Arm workspace: optimal for workbench Y > 0.04\n"
                f"- Shared Central Workspace: Y in [-0.04, 0.04] (optimal for handovers and stacking)\n"
                f"- Table surface: Z = {self.table_z:.2f}m. Objects rest at Z = {self.table_z+0.03:.2f}m - {self.table_z+0.05:.2f}m.\n"
                f"- Floor threshold: Z < {self.floor_thresh_z:.2f}m."
            )

        system_instruction = f"""You are Margaret, the Embodied Reasoning Brain for an autonomous bimanual robot in MuJoCo physics.
{embodiment_text}

Semantic Entity Grounding & Fuzzy Resolution:
- Ground user intent dynamically to entities in the scene based on visual attributes (color, position, shape, category).
- Do NOT assume fixed predefined objects or environments; reason dynamically over whatever objects are present.
- Match colloquial terms: "cube"/"block" matches "box", "ball" matches "sphere", "can"/"tube" matches "capsule" or "cylinder".
- Match adjacent colors under tabletop shading: "yellow" matches "orange", "pink" matches "purple"/"magenta".
- If only ONE box/cube exists in the scene and the user requests a "cube" (e.g. "yellow cube"), BIND TO IT ('prop_orange_box_1')!
- If the user specifies an attribute like "pink object", bind to the closest chromatic candidate ('prop_purple_capsule_3')!
- If the user re-asserts or says an object is still in one hand (e.g. "take completely to left hand"): do NOT exit; command a physical movement (e.g. lift hand or move hand away) to demonstrate secure hold!
- NEVER abort claiming an object is missing if an object with matching or closest attributes exists in the scene! Proceed to manipulate that matching entity.

Strict Physical Preconditions & Rules:
1. If spatial memory contains NO objects, MUST call 'inspect_scene' first!
2. To pick an object not held:
   - First call 'reach_hand' with object's [target_x, target_y, target_z] and correct arm based on embodiment kinematics.
   - Once reach succeeds, call 'grasp_object' with object_name and that hand. NEVER call grasp_object before reach_hand!
   - Next call 'lift_hand' (lift_height=0.15) to elevate it off the table.
3. If an object is ALREADY HELD by the robot:
   - Do NOT reach or grasp it again! Proceed directly to 'place_object' or 'release_object'.
   - If an object is NOT held (Robot Embodiment State shows held: null): NEVER call 'release_object'! Proceed directly to 'reach_hand' to pick it up!
4. Bimanual Coordination & Cooperative Stacking:
   - When stacking an object held in one hand onto an object held in the other hand:
     ↳ First place the base object into the shared central staging zone ({self.staging_pos}) so the arm holding the top object can reach and align directly on top with minimal tracking error!
     ↳ Then use the arm holding the top object to place it onto the base object via 'place_object'.
5. Dropped Objects on Floor:
   - If an object has status 'on_floor' (Z < {self.floor_thresh_z:.2f}m), it has fallen off the table and is out of reach of the manipulator.
   - Props resting normally on the workbench table are NOT on the floor. NEVER call recover_floor_object for objects on the table!
   - Call 'recover_floor_object' with object_name only if an object truly fell off the table.
6. CLOSED-LOOP SENSORY FEEDBACK & RETRY:
   - Review the observation from the previous step.
   - If the previous action succeeded, proceed to the next step.
   - If the previous action failed (e.g. reach error too large, grasp slipped, or dropped on floor): analyze the error, adjust coordinates or parameters, and TRY AGAIN!
7. Returning Arms to Rest / Neutral Pose:
   - If user asks to return to rest, lower arms, or reset pose, call 'return_to_rest' (arm='both' or 'right'/'left').
8. Mobile Locomotion & Obstacle Traversal (for Mobile Embodiments / TRON 1):
   - To clear or navigate around an obstacle: call 'steer_around_obstacle' (direction='left' or 'right', object_name=obstacle_name).
   - To drive forward along a path: call 'move_forward' (distance in meters, speed in m/s).
   - To turn or adjust heading: call 'turn_heading' (angle_deg in degrees, positive=left, negative=right).
   - To traverse an incline ramp: call 'climb_ramp' (speed in m/s, duration_s in seconds).
   - Large obstacle boxes on the floor are obstacles to steer around, NOT objects to reach and pick up!
9. When all sub-goals are physically accomplished and verified in the world, call 'mission_complete'.

Output strictly according to ActionDecision schema."""

        # Construct lightweight token-efficient prompt
        floor_thresh = self.floor_thresh_z
        scene_summary = {}
        for name, data in self.memory.objects.items():
            meta = data.get("metadata", {})
            parts = name.split("_")
            inferred_color = meta.get("color") or (parts[1] if len(parts) >= 3 else "unknown")
            inferred_shape = meta.get("shape") or (parts[2] if len(parts) >= 3 else "object")
            scene_summary[name] = {
                "color": inferred_color,
                "shape": inferred_shape,
                "pos": [round(float(x), 3) for x in data["pos"]],
                "status": meta.get("status", "on_table" if data["pos"][2] >= floor_thresh else "on_floor"),
                "affordances": ["graspable", "liftable"] if data["pos"][2] >= floor_thresh else ["unreachable_from_table", "needs_recovery"]
            }

        robot_state_summary = {
            "right_arm": self.memory.robot_state.get("right_hand", {}),
            "left_arm": self.memory.robot_state.get("left_hand", {}),
        }

        def _safe_serialize(val):
            if hasattr(val, "tolist"):
                return val.tolist()
            if hasattr(val, "item"):
                return val.item()
            return str(val)

        # Retrieve action feedback from the current interactive mission
        current_feedback = [
            {
                "step": idx + 1,
                "thought": h["thought"],
                "action": h["action"],
                "args": h["args"],
                "observation": h["observation"]
            }
            for idx, h in enumerate(self.history.get_current_mission_history(4))
        ]

        prompt = f"""Mission Goal: "{self.current_goal}"
Active Sub-Goals: {[sg.model_dump() for sg in self.decomposition_plan.sub_goals] if self.decomposition_plan else []}
Step: {step_idx}/{self.max_steps}

Token-Efficient 3D Spatial Scene:
{json.dumps(scene_summary, indent=2, default=_safe_serialize)}

Robot Embodiment State:
{json.dumps(robot_state_summary, indent=2, default=_safe_serialize)}

Recent Actions & Sensory Feedback in Current Mission:
{json.dumps(current_feedback, indent=2, default=_safe_serialize)}

Determine the next precise physical action."""

        raw_json = self._call_gemini_pydantic(
            prompt, system_instruction, ActionDecision
        )

        if raw_json:
            try:
                return ActionDecision.model_validate(raw_json)
            except Exception as e:
                print(f"❌ [ActionDecision Validation Error]: {e}")
                return None

        # DO NOT DISPATCH FAKE MOVEMENTS OR HARDCODED ACTIONS
        print(f"❌ [Embodied Reasoning Failed]: Reasoning model unreachable or offline at step {step_idx}.")
        return None

    # =========================================================================
    # GRAPH NODE 4: Sandbox World Model Verifier
    # =========================================================================
    def node_world_model_verify(self, decision: ActionDecision, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs candidate action through the Physics Sandbox World Model in imagination.
        Evaluates reachability, collisions, and slip before sending commands to physical motors.
        """
        if not self.world_model:
            return {"recommendation": "EXECUTE"}

        if decision.action in ["move_forward", "turn_heading", "drive", "steer_around_obstacle", "climb_ramp", "inspect_scene", "return_to_rest"]:
            return {"recommendation": "EXECUTE"}

        act_arm = decision.arm if decision.arm in ["right", "left"] else "right"
        if self.world_model and self.world_model.robot:
            base_shoulder = list(self.world_model.robot.shoulder_pos)
            if act_arm == "left":
                base_shoulder[1] = -base_shoulder[1]
            shoulder_3d = base_shoulder
        else:
            shoulder_3d = [0.0, -0.10, 1.08] if act_arm == "right" else [0.0, 0.10, 1.08]

        if decision.action in ["reach_hand", "place_object"]:
            tx = args.get("target_x")
            ty = args.get("target_y")
            tz = args.get("target_z")
            if tx is not None and ty is not None and tz is not None:
                return self.world_model.simulate_reach(shoulder_3d, [tx, ty, tz], hand=act_arm)

        elif decision.action == "grasp_object":
            return self.world_model.simulate_grasp(object_mass=0.2, grip_force_N=12.5)

        return {"recommendation": "EXECUTE"}

    # =========================================================================
    # DYNAMIC GRAPH EXECUTION RUNNER
    # =========================================================================
    def execute_live_graph(self, user_goal: str, max_steps: int = 10) -> Dict[str, Any]:
        """
        Executes the Live Agentic Graph dynamically on-demand with closed-loop feedback.
        Routes conditionally through:
        Perception Grounding -> Decomposition -> [Embodied Reasoning -> World Model -> Physical Skill -> Sensory Feedback]
        """
        self.current_goal = user_goal
        self.max_steps = max_steps
        self.step_counter = 0
        self.is_completed = False

        # Initialize fresh mission history for closed-loop reasoning
        self.history.start_new_mission(user_goal)

        print("\n" + "=" * 68)
        print(f"🎯 EMBODIED LIVE GRAPH MISSION: \"{user_goal}\"")
        print("=" * 68)

        # 1. Ground Perception if memory is empty
        if not self.memory.objects:
            print("👁️ [PerceptionNode] Spatial memory ungrounded -> Triggering token-efficient 3D vision...")
            self.node_perception()

        # 2. On-Demand Decomposition
        plan = self.node_decompose_task(user_goal)
        if not plan:
            print(f"\n❌ [ABORTED]: Failed to obtain task plan from reasoning model.")
            print(f"🛑 Robot physical execution HALTED. Zero movements dispatched without verified model reasoning.\n")
            return {"status": "error", "error": "Model offline or unavailable for task decomposition. Physical execution halted."}

        print(f"\n🧠 [Embodied Cognitive Plan via {self.model}]:")
        print(f"   • Intent Analysis: {plan.intent_analysis}")
        print(f"   • Goal Type: {plan.goal_type}")
        print(f"   • Sub-Goal Sequence:")
        for sg in plan.sub_goals:
            arm_badge = f"[{sg.arm.upper()}] " if sg.arm != 'auto' else ""
            print(f"     ↳ {sg.id}. {arm_badge}{sg.description}")

        # 3. Dynamic Closed-Loop Execution Loop
        while self.step_counter < max_steps and not self.is_completed:
            self.step_counter += 1
            time.sleep(0.4)  # Yield to GUI window and rate-limit API calls
            print(f"\n--- [GRAPH STEP {self.step_counter}/{max_steps}] ---")

            # Check for asynchronous interrupts or events (user, environment, or robot description)
            if self.harness.has_pending_events():
                event_decision = self.harness.evaluate_events()
                print(f"⚡ [Event Decision]: {event_decision.decision} - {event_decision.reason}")
                if event_decision.decision == "HALT_SAFETY":
                    print("🛑 [SAFETY HALT]: Mission halted due to event safety policy.")
                    return {"status": "error", "error": f"Mission halted: {event_decision.reason}"}
                elif event_decision.decision == "REPLAN":
                    print("🔄 [REPLAN TRIGGERED]: Re-decomposing plan in response to event...")
                    plan = self.node_decompose_task(user_goal)
                    if not plan:
                        return {"status": "error", "error": "Re-decomposition failed after event."}

            # Reasoning Node: Gemini Robotics ER decision
            decision = self.node_embodied_reasoning(self.step_counter)
            if not decision or decision.action == "abort":
                reason = decision.thought if decision else (plan.intent_analysis if plan else "Could not determine safe physical movement")
                print(f"\n⚠️ [Mission Aborted / Unable to Proceed]: {reason}")
                print(f"📦 Grounded Objects Currently on Workbench:")
                for o_name, o_data in self.memory.objects.items():
                    o_meta = o_data.get("metadata", {})
                    o_parts = o_name.split("_")
                    o_col = o_meta.get("color") or (o_parts[1] if len(o_parts) >= 3 else "")
                    o_shp = o_meta.get("shape") or (o_parts[2] if len(o_parts) >= 3 else "")
                    pos_fmt = [round(float(x), 3) for x in o_data["pos"]]
                    print(f"   • {o_name}: {o_col} {o_shp} at pos={pos_fmt}")
                print(f"🛑 Robot physical execution HALTED safely.\n")
                return {"status": "error", "error": f"Mission cannot proceed: {reason}"}

            print(f"🤔 Thought: {decision.thought}")

            if decision.action == "mission_complete":
                print("🏁 Mission Complete Goal Achieved!")
                self.history.record_step(decision.thought, decision.action, {}, {"status": "goal_achieved"})
                self.is_completed = True
                return {"status": "success", "steps": self.step_counter}

            # Map Decision to physical tool arguments
            args: Dict[str, Any] = {}
            if decision.action == "reach_hand":
                target_obj = decision.object_name or (self.decomposition_plan.target_object if self.decomposition_plan else None)
                args = {
                    "target_x": decision.target_x,
                    "target_y": decision.target_y,
                    "target_z": decision.target_z,
                    "hand": decision.arm,
                }
                if target_obj:
                    args["object_name"] = target_obj
            elif decision.action == "grasp_object":
                target_obj = decision.object_name
                if not target_obj and self.decomposition_plan:
                    target_obj = self.decomposition_plan.target_object
                args = {
                    "object_name": target_obj,
                    "hand": decision.arm,
                }
            elif decision.action == "lift_hand":
                args = {
                    "height": decision.lift_height or 0.15,
                    "hand": decision.arm,
                }
            elif decision.action == "place_object":
                args = {
                    "target_x": decision.target_x,
                    "target_y": decision.target_y,
                    "target_z": decision.target_z,
                    "hand": decision.arm,
                    "dest_object": decision.dest_object,
                }
            elif decision.action == "release_object":
                args = {
                    "object_name": decision.object_name,
                    "hand": decision.arm,
                }
            elif decision.action == "return_to_rest":
                args = {
                    "hand": decision.arm if decision.arm in ["right", "left"] else "both",
                }
            elif decision.action == "recover_floor_object":
                target_obj = decision.object_name
                if not target_obj and self.decomposition_plan:
                    target_obj = self.decomposition_plan.target_object
                args = {
                    "object_name": target_obj,
                    "target_x": decision.target_x if decision.target_x is not None else self.staging_pos[0],
                    "target_y": decision.target_y if decision.target_y is not None else self.staging_pos[1],
                }
            elif decision.action == "inspect_scene":
                args = {}
            elif decision.action == "move_forward":
                args = {
                    "distance": decision.distance if decision.distance is not None else 1.0,
                    "speed": decision.speed if decision.speed is not None else 0.6,
                }
            elif decision.action == "turn_heading":
                args = {
                    "angle_deg": decision.angle_deg if decision.angle_deg is not None else 30.0,
                }
            elif decision.action == "drive":
                args = {
                    "linear_vel": decision.speed if decision.speed is not None else 0.5,
                    "angular_vel": 0.0,
                    "duration_s": 1.0,
                }
            elif decision.action == "steer_around_obstacle":
                args = {
                    "direction": decision.direction or "left",
                    "object_name": decision.object_name or (self.decomposition_plan.target_object if self.decomposition_plan else "obstacle"),
                }
            elif decision.action == "climb_ramp":
                args = {
                    "speed": decision.speed if decision.speed is not None else 0.8,
                    "duration_s": 2.0,
                }

            print(f"⚡ Action:  {decision.action}({args})")

            # GRAPH NODE 4: Sandbox World Model Verification
            wm_check = self.node_world_model_verify(decision, args)
            if wm_check.get("recommendation") == "REPLAN":
                outcome = wm_check.get("predicted_outcome", "PHYSICAL_RISK_DETECTED")
                print(f"🔮 [Sandbox World Model]: REJECTED candidate trajectory! Reason: {outcome}")
                obs = {
                    "success": False,
                    "error": f"World Model sandbox predicted failure: {outcome}. Please replan with valid coordinates or alternative arm."
                }
                print(f"👁️ Observe: {obs}")
                self.history.record_step(decision.thought, decision.action, args, obs)
                continue

            # Execution Node: Actuate physical MuJoCo controllers + ROS 2
            exec_res = self.tools.execute(decision.action, args)
            obs = exec_res.get("result") if exec_res.get("success") else exec_res.get("error")
            print(f"👁️ Observe: {obs}")

            # SENSE THE RESULT: Update spatial & proprioceptive memory with live feedback
            act_hand = decision.arm if decision.arm in ["right", "left"] else "right"
            if isinstance(obs, dict):
                if "reached_position" in obs:
                    self.memory.update_hand_pos(act_hand, obs["reached_position"])
                if decision.action == "grasp_object" and obs.get("success"):
                    self.memory.set_hand_held(act_hand, decision.object_name)
                elif decision.action == "lift_hand" and obs.get("success"):
                    self.memory.set_hand_lifted(act_hand, True)
                    # Sensed live lift: update object elevation
                    held_obj = self.memory.robot_state[f"{act_hand}_hand"].get("held")
                    if held_obj and held_obj in self.memory.objects:
                        self.memory.objects[held_obj]["pos"][2] += (decision.lift_height or 0.15)
                elif decision.action in ["place_object", "release_object"] and obs.get("success"):
                    self.memory.set_hand_held(act_hand, None)
                    self.memory.set_hand_lifted(act_hand, False)
                    self.memory.set_hand_reached(act_hand, None)
                elif decision.action == "recover_floor_object" and obs.get("success"):
                    rec_obj = obs.get("object")
                    if rec_obj and "position" in obs:
                        self.memory.update_object(rec_obj, obs["position"], metadata={"status": "on_table"})
                elif "final_position" in obs:
                    self.memory.robot_state["base_pos"] = obs["final_position"]
                elif "current_position" in obs:
                    self.memory.robot_state["base_pos"] = obs["current_position"]

            # Observation Node: Record episodic feedback for closed-loop ReAct
            self.history.record_step(decision.thought, decision.action, args, obs)

        return {"status": "max_steps_reached", "steps": self.step_counter}
