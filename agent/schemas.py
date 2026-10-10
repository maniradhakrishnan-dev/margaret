"""
Embodied AI Pydantic Schemas
Strict type definitions and validation schemas for Gemini Robotics ER models,
eliminating hallucinations and enforcing physical groundedness.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class SubGoal(BaseModel):
    id: int = Field(description="Step index of this sub-goal")
    description: str = Field(description="Physical sub-task description")
    arm: Literal["right", "left", "auto"] = Field(
        default="auto", description="Arm selected based on workspace Y boundary (right <= 0.04, left > 0.04)"
    )
    action: Literal["inspect", "reach", "grasp", "lift", "place", "release", "recover", "complete"] = Field(
        description="Physical skill action category"
    )
    target_object: Optional[str] = Field(default=None, description="Target object identifier")
    destination_object: Optional[str] = Field(default=None, description="Placement destination if applicable")


class TaskDecompositionPlan(BaseModel):
    intent_analysis: str = Field(
        description="Detailed physical, geometric, and semantic analysis of the user instruction"
    )
    goal_type: Literal[
        "PICK_AND_PLACE", "PICK_AND_LIFT", "RELEASE", "INSPECT", "REORIENT", "COMPLEX"
    ] = Field(description="High-level manipulation category")
    target_object: Optional[str] = Field(
        default=None, description="Primary object identifier to manipulate grounded from the scene"
    )
    destination_object: Optional[str] = Field(
        default=None, description="Destination target object identifier if placing"
    )
    sub_goals: List[SubGoal] = Field(
        description="Ordered sequence of physical sub-goals to accomplish the mission"
    )


class ActionDecision(BaseModel):
    thought: str = Field(
        description="Embodied physical reasoning justifying this action choice, arm allocation, and safety"
    )
    action: Literal[
        "inspect_scene",
        "reach_hand",
        "grasp_object",
        "lift_hand",
        "place_object",
        "release_object",
        "return_to_rest",
        "recover_floor_object",
        "mission_complete",
        "abort",
    ] = Field(description="Name of the physical tool / skill to actuate")
    arm: Literal["right", "left", "auto"] = Field(
        default="auto", description="Which humanoid arm to actuate: 'right' (Y <= 0.04) or 'left' (Y > 0.04)"
    )
    target_x: Optional[float] = Field(
        default=None, description="Target X coordinate in world frame (meters)"
    )
    target_y: Optional[float] = Field(
        default=None, description="Target Y coordinate in world frame (meters)"
    )
    target_z: Optional[float] = Field(
        default=None, description="Target Z coordinate in world frame (meters)"
    )
    object_name: Optional[str] = Field(
        default=None, description="Target object identifier to manipulate"
    )
    dest_object: Optional[str] = Field(
        default=None, description="Destination object identifier for placement"
    )
    lift_height: Optional[float] = Field(
        default=0.15, description="Height in meters to lift arm above current position"
    )


class SceneEntity(BaseModel):
    name: str = Field(description="Unique entity identifier in the scene")
    category: str = Field(description="Object class / shape geometry category")
    color: str = Field(description="Visual color description")
    position_3d: List[float] = Field(
        description="3D centroid [X, Y, Z] in world coordinate frame (meters)"
    )
    affordances: List[str] = Field(
        default_factory=lambda: ["graspable", "liftable", "stackable"],
        description="Physical interaction affordances"
    )


class ArmState(BaseModel):
    held: Optional[str] = None
    reached_object: Optional[str] = None
    pos: List[float]
    lifted: bool = False


class RobotEmbodimentState(BaseModel):
    right_hand: ArmState
    left_hand: ArmState
    is_stable: bool = True


class EventNotification(BaseModel):
    event_type: Literal[
        "USER_INTERRUPT",
        "ENVIRONMENT_CHANGE",
        "ROBOT_DESCRIPTION_CHANGE",
        "ROBOT_FAULT",
        "NEW_TOOL_REGISTERED",
    ] = Field(description="Category of the asynchronous or sensory event")
    description: str = Field(description="Description of what occurred in the environment, user stream, or robot")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Detailed event payload data")
    timestamp: float = Field(default=0.0, description="Epoch timestamp of event occurrence")


class EventDecision(BaseModel):
    thought: str = Field(description="Reasoning on how this event impacts the ongoing physical plan")
    decision: Literal["CONTINUE", "REPLAN", "HALT_SAFETY"] = Field(
        description="Whether to continue current execution, replan the sub-goals, or immediately halt for safety"
    )
    reason: str = Field(description="Justification for the decision")


class ActionFeedback(BaseModel):
    success: bool = Field(description="Whether the physical motor execution succeeded")
    action: str = Field(description="Physical tool action that was actuated")
    reached_position: Optional[List[float]] = Field(default=None, description="Actual reached 3D position in world space")
    distance_error_cm: Optional[float] = Field(default=None, description="Tracking error distance in centimeters")
    measured_grip_force_N: Optional[float] = Field(default=None, description="Measured physical contact/grip force in Newtons")
    error: Optional[str] = Field(default=None, description="Error or collision description if failed")
    joint_feedback: Optional[Dict[str, Any]] = Field(default=None, description="Physical joint angles or actuator telemetry")


def to_gemini_schema(schema_input) -> Dict[str, Any]:
    """
    Transforms Pydantic OpenAPI schemas into Gemini REST API-compatible schemas by:
    1. Inlining and dereferencing all nested $defs and $ref models.
    2. Removing metadata keys ($defs, title) unsupported by Gemini responseSchema.
    3. Normalizing anyOf with null into nullable primitive types.
    """
    import copy

    if hasattr(schema_input, "model_json_schema"):
        schema_dict = schema_input.model_json_schema()
    else:
        schema_dict = schema_input

    defs = schema_dict.get("$defs", {})

    def deref(item):
        if isinstance(item, dict):
            if "$ref" in item:
                ref = item["$ref"].split("/")[-1]
                target = copy.deepcopy(defs.get(ref, {}))
                return deref(target)
            new_dict = {}
            for k, v in item.items():
                if k in ["$defs", "title"]:
                    continue
                if k == "anyOf":
                    types = [x.get("type") for x in v if isinstance(x, dict) and x.get("type") != "null"]
                    if types:
                        new_dict["type"] = types[0]
                    new_dict["nullable"] = True
                    continue
                new_dict[k] = deref(v)
            return new_dict
        elif isinstance(item, list):
            return [deref(x) for x in item]
        return item

    return deref(schema_dict)
