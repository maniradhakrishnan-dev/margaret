"""
Embodied Autonomous Margaret Agent System.
Agent-First Architecture:
1. Agent initializes and configures the physical environment (plain workbench, zero obstruction sheets)
2. Agent loads the robot embodiment (Unitree G1 Humanoid) and settles standing physics
3. Agent initializes perception, world model, 50Hz closed-loop visual servoing, and cognitive ReAct brain
4. Agent opens the live 3D MuJoCo simulation window with full mouse and hotkey disturbance support
5. Agent provides an interactive console for natural language commands and real-time physical disturbances
"""

import os
import sys
import time
import threading
import math
import argparse
from typing import Optional, List, Dict, Any
import numpy as np
import mujoco
import mujoco.viewer

from arena import PhysicalEnvironment, EnvironmentDisturber
from robots import get_robot
from skills.vision import SpatialVision
from skills.reach_controller import ReachController
from skills.grasp import GraspController
from skills.navigation import MobileNavigationController
from agent.world_model import WorldModel
from agent.tools import ToolRegistry
from agent.brain import AgentBrain
from agent.schemas import EventNotification


class MargaretAgent:
    """
    Autonomous Embodied AI Agent for Physical Humanoid Manipulation.
    Controls its own environment loading, perception grounding, physical execution,
    and adaptive responses to runtime environmental disturbances.
    """

    def __init__(
        self,
        robot: str = "unitree_g1",
        num_objects: int = 3,
        furniture: str = "workbench",
        obstacle: str = "none",
        use_yolo: bool = True,
        seed: Optional[int] = None,
    ):
        self.robot_name = robot
        self.num_objects = num_objects
        self.furniture_type = furniture
        self.obstacle_type = obstacle
        self.use_yolo = use_yolo
        self.seed = seed

        # State flags
        self.is_running = False
        self.adversary_enabled = False
        self.active_mission_thread: Optional[threading.Thread] = None
        self.adversary_thread: Optional[threading.Thread] = None

        # Core Components (initialized during load_environment)
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.robot = None
        self.physics_lock = threading.RLock()
        self.disturber: Optional[EnvironmentDisturber] = None
        self.vision: Optional[SpatialVision] = None
        self.reach_ctrl: Optional[ReachController] = None
        self.grasp_ctrl: Optional[GraspController] = None
        self.nav_ctrl: Optional[MobileNavigationController] = None
        self.world_model: Optional[WorldModel] = None
        self.tool_registry = ToolRegistry()
        self.brain: Optional[AgentBrain] = None
        self.spawned_props: List[Dict[str, Any]] = []
        self.active_prop_idx = 0
        self.xml_path = ""

    def load_environment(self) -> str:
        """
        Agent configures and loads its physical world, robot embodiment, and vision model.
        Guarantees a clean, plain wooden workbench with zero upper sheets or barrier obstructions.
        """
        print("\n" + "=" * 78)
        print("🤖 [MARGARET AGENT]: Autonomous Boot Sequence Initiated")
        print("=" * 78)

        # Stage 1: Load Physical Environment (isolated in arena/)
        print("\n🌍 [Stage 1/5] Loading Physical Environment from arena/...")
        self.arena = PhysicalEnvironment(
            robot_platform=self.robot_name,
            num_objects=self.num_objects,
            furniture_type=self.furniture_type,
            obstacle_type=self.obstacle_type,
            seed=self.seed,
        )
        self.xml_path, self.spawned_props = self.arena.build(output_filename="runtime_interactive_arena.xml")
        self.model = self.arena.model
        self.data = self.arena.data
        self.disturber = self.arena.disturber

        print(f"   • Plain Workbench Arena generated: {self.xml_path}")
        print(f"   • Initial props spawned on table surface:")
        for p in self.spawned_props:
            print(f"     ↳ [{p['name']}]: shape={p['shape']}, color={p['color']}, pos={p['initial_pos']}, mass={p['mass']}kg")

        # Stage 2: Load Robot Embodiment (isolated in robots/)
        print(f"\n🤖 [Stage 2/5] Loading Robot Embodiment ({self.robot_name.upper()})...")
        self.robot = get_robot(self.robot_name)
        self.robot.setup_data(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        print(f"   • Robot model initialized with {self.model.nu} actuators ({len(self.robot.arm_joint_names)} arm DoFs).")

        # Stage 3: Load Vision System
        cam_name = "head_cam" if self.robot_name in ["unitree_g1", "limx_tron1", "tron1", "wheeled_biped"] else "overhead_cam"
        print(f"\n👁️ [Stage 3/5] Loading Vision System (Camera: {cam_name})...")
        self.vision = SpatialVision(self.model, self.data, camera_name=cam_name, lock=self.physics_lock, table_surface_z=self.robot.table_surface_z)
        if self.use_yolo:
            print("   • Initializing Local YOLO-World Open-Vocabulary Neural Model...")
            _ = self.vision.local_vision
            print("   • Local Neural Vision Model loaded and ready.")
        else:
            print("   • Optical 3D metric deprojection vision loaded.")

        # Stage 4: Load Kinematics & Motor Controllers
        print("\n🦾 [Stage 4/5] Loading Kinematics, Mobile Navigation & Closed-Loop Motor Controllers...")
        self.disturber = EnvironmentDisturber(self.model, self.data)
        self.reach_ctrl = ReachController(self.model, self.data, robot=self.robot, lock=self.physics_lock)
        self.grasp_ctrl = GraspController(self.model, self.data, robot=self.robot, lock=self.physics_lock)
        self.nav_ctrl = MobileNavigationController(self.model, self.data, robot=self.robot, lock=self.physics_lock)
        self.world_model = WorldModel(robot=self.robot)
        print("   • Mobile Navigation, Inverse Kinematics, and Closed-Loop skills ready.")

        # Stage 5: Load Cognitive ReAct Brain & World Model
        print("\n🧠 [Stage 5/5] Initializing Cognitive ReAct Brain & Sandbox World Model...")
        self._register_skills()
        self.brain = AgentBrain(self.tool_registry, world_model=self.world_model)

        # Settle robot into stable posture (100 steps)
        with self.physics_lock:
            for _ in range(100):
                if hasattr(self.robot, "apply_balance_stabilization"):
                    self.robot.apply_balance_stabilization(self.model, self.data)
                mujoco.mj_step(self.model, self.data)

        print("\n✅ [AGENT READY]: Margaret fully booted standing at plain workbench.")
        print("=" * 78)
        return self.xml_path

    def _register_skills(self):
        """Registers physical skills to Margaret's ToolRegistry."""
        @self.tool_registry.register(name="inspect_scene", description="Scans workbench with camera to detect objects and 3D positions.")
        def tool_inspect_scene():
            if self.use_yolo:
                objects = self.vision.detect_with_local_model()
            else:
                objects = self.vision.detect_table_objects()
            for obj in objects:
                self.brain.memory.update_object(
                    obj["name"],
                    obj["position_3d"],
                    metadata={"status": obj.get("status", "on_table"), "color": obj.get("color", "unknown")},
                )
            return {"detected_count": len(objects), "objects": objects}

        @self.tool_registry.register(name="reach_hand", description="Uses 50Hz closed-loop visual servoing to curve arm toward moving or static object.")
        def tool_reach_hand(target_x: float, target_y: float, target_z: float, hand: str = "auto", object_name: str = None):
            res = self.reach_ctrl.execute_reach(
                target_x=target_x,
                target_y=target_y,
                target_z=target_z,
                hand=hand,
                duration_s=1.2,
                steps=40,
                target_object=object_name,
                vision=self.vision,
            )
            actual_hand = res.get("hand", self.robot.get_optimal_arm([target_x, target_y, target_z]) if (self.robot and hasattr(self.robot, "get_optimal_arm")) else (hand if hand in ["right", "left"] else ("left" if target_y > 0.04 else "right")))
            best_obj, min_d = None, float("inf")
            for obj_name, obj_data in self.brain.memory.objects.items():
                d = np.linalg.norm(np.array(obj_data["pos"]) - np.array([target_x, target_y, target_z]))
                if d < min_d:
                    min_d = d
                    best_obj = obj_name

            if res.get("success") and best_obj and min_d < 0.15:
                self.brain.memory.set_hand_reached(actual_hand, best_obj)
            else:
                self.brain.memory.set_hand_reached(actual_hand, None)

            h_key = f"{actual_hand}_hand"
            if h_key in self.brain.memory.robot_state and "reached_position" in res:
                self.brain.memory.robot_state[h_key]["pos"] = res["reached_position"]
            self.brain.memory.robot_state["hand_pos"] = res.get("reached_position")
            return res

        @self.tool_registry.register(name="grasp_object", description="Closes fingers around target object and verifies tactile grip.")
        def tool_grasp_object(object_name: str, hand: str = "auto"):
            res = self.grasp_ctrl.execute_grasp(object_name, hand=hand)
            if res.get("success"):
                actual_hand = res.get("hand", "right")
                self.brain.memory.set_hand_held(actual_hand, object_name)
            return res

        @self.tool_registry.register(name="lift_hand", description="Lifts the arm and held object upwards from the table.")
        def tool_lift_hand(height: float = 0.15, hand: str = "auto"):
            res = self.grasp_ctrl.execute_lift(height=height, hand=hand, steps=40)
            if res.get("success"):
                actual_hand = res.get("hand", "right")
                self.brain.memory.set_hand_lifted(actual_hand, True)
                held_obj = self.grasp_ctrl.held_objects.get(actual_hand)
                if held_obj and held_obj in self.brain.memory.objects:
                    self.brain.memory.objects[held_obj]["pos"][2] += height
            return res

        @self.tool_registry.register(name="place_object", description="Transports held object and places it gently onto target.")
        def tool_place_object(target_x: float = None, target_y: float = None, target_z: float = None, hand: str = "auto", dest_object: str = None):
            res = self.grasp_ctrl.execute_place(target_x, target_y, target_z, hand=hand, dest_object=dest_object, duration_s=1.2, steps=40)
            if res.get("success"):
                actual_hand = res.get("hand", "right")
                self.brain.memory.set_hand_held(actual_hand, None)
                self.brain.memory.set_hand_reached(actual_hand, None)
                self.brain.memory.set_hand_lifted(actual_hand, False)
            return res

        @self.tool_registry.register(name="release_object", description="Opens fingers and releases held object.")
        def tool_release_object(object_name: str = None, hand: str = "auto"):
            res = self.grasp_ctrl.execute_release(object_name=object_name, hand=hand)
            if res.get("success"):
                actual_hand = res.get("hand", "right")
                self.brain.memory.set_hand_held(actual_hand, None)
                self.brain.memory.set_hand_reached(actual_hand, None)
                self.brain.memory.set_hand_lifted(actual_hand, False)
            return res

        @self.tool_registry.register(name="return_to_rest", description="Smoothly returns one or both humanoid arms to the neutral resting pose.")
        def tool_return_to_rest(hand: str = "both"):
            res = self.reach_ctrl.execute_return_to_rest(hand=hand)
            if res.get("success"):
                hands_reset = ["right", "left"] if hand.lower() in ["both", "all", "auto"] else [hand.lower()]
                for h in hands_reset:
                    self.brain.memory.set_hand_reached(h, None)
                    self.brain.memory.set_hand_lifted(h, False)
            return res

        # Mobile Locomotion & Obstacle Traversal Skills (TRON 1 & Mobile Platforms)
        @self.tool_registry.register(name="move_forward", description="Propels robot forward across terrain by specified distance in meters.")
        def tool_move_forward(distance: float = 1.0, speed: float = 0.6):
            if self.nav_ctrl:
                return self.nav_ctrl.move_forward(distance=distance, speed=speed)
            return {"success": False, "error": "No navigation controller configured"}

        @self.tool_registry.register(name="turn_heading", description="Rotates robot in place by angle in degrees (positive is left, negative is right).")
        def tool_turn_heading(angle_deg: float = 30.0):
            if self.nav_ctrl:
                return self.nav_ctrl.turn(angle_deg=angle_deg)
            return {"success": False, "error": "No navigation controller configured"}

        @self.tool_registry.register(name="drive", description="Applies differential drive linear velocity (m/s) and angular yaw rate (rad/s) for duration.")
        def tool_drive(linear_vel: float = 0.5, angular_vel: float = 0.0, duration_s: float = 1.0):
            if self.nav_ctrl:
                return self.nav_ctrl.drive(linear_vel=linear_vel, angular_vel=angular_vel, duration_s=duration_s)
            return {"success": False, "error": "No navigation controller configured"}

        @self.tool_registry.register(name="steer_around_obstacle", description="Steers around an obstacle corridor on the left or right and realigns forward heading.")
        def tool_steer_around_obstacle(direction: str = "left", object_name: str = None):
            if self.nav_ctrl:
                return self.nav_ctrl.steer_around_obstacle(direction=direction)
            return {"success": False, "error": "No navigation controller configured"}

        @self.tool_registry.register(name="climb_ramp", description="Climbs up or traverses an incline ramp with active wheel propulsion.")
        def tool_climb_ramp(speed: float = 0.8, duration_s: float = 2.0):
            if self.nav_ctrl:
                return self.nav_ctrl.climb_ramp(speed=speed, duration_s=duration_s)
            return {"success": False, "error": "No navigation controller configured"}

    def inspect_scene(self) -> List[Dict[str, Any]]:
        """Scans the physical arena with head camera and grounds all detected obstacles/objects in memory."""
        if self.robot_name in ["limx_tron1", "tron1", "wheeled_biped"]:
            print("👁️ [TRON 1]: Scanning obstacle course terrain with onboard head camera...")
            objects = []
            robot_pos = self.nav_ctrl.get_robot_pose()["position"] if self.nav_ctrl else [0.0, 0.0, 0.0]
            with self.physics_lock:
                for p in self.spawned_props:
                    bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, p["name"])
                    if bid != -1:
                        cur_pos = self.data.xpos[bid].round(3).tolist()
                    else:
                        cur_pos = p["initial_pos"]
                    dx = cur_pos[0] - robot_pos[0]
                    dy = cur_pos[1] - robot_pos[1]
                    dist = round(math.hypot(dx, dy), 2)
                    objects.append({
                        "name": p["name"],
                        "shape": p["shape"],
                        "color": p["color"],
                        "position_3d": cur_pos,
                        "distance_m": dist,
                        "status": "cleared" if dx < -0.3 else ("ahead" if dx > 0 else "alongside"),
                    })
            for obj in objects:
                self.brain.memory.update_object(
                    obj["name"],
                    obj["position_3d"],
                    metadata={"status": obj["status"], "color": obj["color"], "shape": obj["shape"], "distance_m": obj["distance_m"]},
                )
            print(f"👁️ [TRON 1]: Grounded {len(objects)} course features (obstacles, ramp, goal) in spatial memory.")
            return objects

        print("👁️ [Margaret]: Scanning plain workbench with head camera...")
        with self.physics_lock:
            if self.use_yolo:
                objects = self.vision.detect_with_local_model()
            else:
                objects = self.vision.detect_table_objects()
        for obj in objects:
            self.brain.memory.update_object(
                obj["name"],
                obj["position_3d"],
                metadata={"status": obj.get("status", "on_table"), "color": obj.get("color", "unknown")},
            )
        return objects

    def get_current_target_name(self) -> str:
        """Returns the currently active target prop name."""
        if not self.spawned_props:
            return "none"
        return self.spawned_props[self.active_prop_idx % len(self.spawned_props)]["name"]

    def execute_mission(self, user_command: str):
        """Dispatches an autonomous cognitive mission to Margaret asynchronously."""
        def _run():
            try:
                self.brain.run_mission(user_command)
            except Exception as e:
                print(f"❌ [Mission Error]: {e}")
            finally:
                print(f"\n🤖 [Margaret Ready] > ", end="", flush=True)

        self.active_mission_thread = threading.Thread(target=_run, daemon=True)
        self.active_mission_thread.start()

    def nudge_active_prop(self, delta_x: float = 0.0, delta_y: float = 0.0, delta_z: float = 0.0):
        """Directly slides the active prop by (dx, dy, dz) meters across the table."""
        tgt = self.get_current_target_name()
        with self.physics_lock:
            if self.grasp_ctrl:
                for h in ["right", "left"]:
                    if self.grasp_ctrl.held_objects.get(h) == tgt:
                        print(f"🔓 [Auto-Release]: Disengaging robot grip on '{tgt}' before nudge...")
                        self.grasp_ctrl.execute_release(tgt, hand=h)
                        if self.brain and self.brain.memory:
                            self.brain.memory.set_hand_held(h, None)
                            self.brain.memory.set_hand_lifted(h, False)
            self.disturber.displace_object(tgt, delta_x=delta_x, delta_y=delta_y, delta_z=delta_z, brain=self.brain)

    def disturb(self, action: str, target: Optional[str] = None, val: float = 0.12):
        """
        Injects a physical disturbance into the active world.
        Updates physics immediately and posts an EventNotification to Margaret's brain.
        """
        target_name = target or self.get_current_target_name()
        resolved = self.disturber.resolve_target_name(target_name) or target_name

        with self.physics_lock:
            # If object is currently gripped by robot, release hold so disturbance works freely
            if self.grasp_ctrl:
                for h in ["right", "left"]:
                    if self.grasp_ctrl.held_objects.get(h) == resolved:
                        print(f"🔓 [Auto-Release]: Disengaging robot grip on '{resolved}' for disturbance...")
                        self.grasp_ctrl.execute_release(resolved, hand=h)
                        if self.brain and self.brain.memory:
                            self.brain.memory.set_hand_held(h, None)
                            self.brain.memory.set_hand_lifted(h, False)

            if action in ["displace", "nudge", "move"]:
                print(f"\n💥 [DISTURBANCE]: Nudging '{resolved}' by +{val*100:.1f}cm across the table...")
                self.disturber.displace_object(resolved, delta_x=0.0, delta_y=val, brain=self.brain)

            elif action in ["knock", "tumble"]:
                print(f"\n💥 [DISTURBANCE]: Knocking over '{resolved}' with angular momentum...")
                self.disturber.knock_over_object(resolved, angular_impulse=3.0, brain=self.brain)

            elif action in ["drop", "floor"]:
                print(f"\n💥 [DISTURBANCE]: Dropping '{resolved}' off the table onto the floor...")
                self.disturber.drop_object_off_table(resolved, brain=self.brain)

            elif action in ["shake", "tremor"]:
                print(f"\n💥 [DISTURBANCE]: Shaking table! Perturbing all props simultaneously...")
                self.disturber.shake_all_props(jitter_std=0.035, brain=self.brain)

            else:
                print(f"⚠️ Unknown disturbance action: '{action}'")

    def toggle_adversary(self, state: Optional[bool] = None):
        """Toggles the autonomous adversary daemon that perturbs objects mid-reach."""
        if state is not None:
            self.adversary_enabled = state
        else:
            self.adversary_enabled = not self.adversary_enabled

        status_str = "ENABLED (Auto-perturbs targets mid-reach)" if self.adversary_enabled else "DISABLED"
        print(f"\n🦹 [Adversary Mode]: {status_str}")

        if self.adversary_enabled and (self.adversary_thread is None or not self.adversary_thread.is_alive()):
            def _adversary_loop():
                while self.adversary_enabled and self.is_running:
                    time.sleep(0.5)
                    # Check if an arm reach motion is actively executing
                    if self.reach_ctrl and getattr(self.reach_ctrl, "_is_reaching", False):
                        time.sleep(0.4)
                        tgt = self.get_current_target_name()
                        print(f"\n🦹 [Adversary Triggered]: Perturbing '{tgt}' by +15cm mid-reach!")
                        self.disturb("displace", tgt, val=0.15)
                        time.sleep(3.0)

            self.adversary_thread = threading.Thread(target=_adversary_loop, daemon=True)
            self.adversary_thread.start()

    def print_menu(self):
        """Displays the clean, comprehensive Agent-First command center."""
        print("\n" + "=" * 78)
        print("🤖 MARGARET AGENT SYSTEM (Embodied Autonomous Physical AI)")
        print("=" * 78)
        print(f"Embodiment: {self.robot_name.upper()} Humanoid")
        print("Workbench:  Plain Wooden Table (clean, zero obstructions)")
        print(f"Objects:    {len(self.spawned_props)} grounded props on table:")
        for idx, p in enumerate(self.spawned_props):
            marker = "👉" if idx == self.active_prop_idx else "  "
            print(f"   {marker} [{idx+1}] {p['name']} ({p['color']} {p['shape']}, mass: {p['mass']}kg)")

        print("-" * 78)
        print("COMMANDS:")
        print("  [Natural Language Tasks to Margaret]:")
        print(f"    • 'pick up the {self.spawned_props[0]['color']} {self.spawned_props[0]['shape']}'")
        print(f"    • 'grasp {self.get_current_target_name()} and lift it'")
        print("    • 'inspect scene'")
        print("")
        print("  [Live Environment Disturbances (Test Margaret's Real-Time Adaptation!)]:")
        print("    • 'select <obj|1..n>'   -> Switch active prop (e.g. 'select 2', 'select red box')")
        print("    • 'disturb <obj> [dx]' -> Nudge object by dx meters (e.g. 'disturb red box 0.15')")
        print("    • 'knock <obj>'         -> Knock over / tumble object (e.g. 'knock cylinder')")
        print("    • 'drop <obj>'          -> Drop object off table onto floor (or 'disturb drop <obj>')")
        print("    • 'shake'               -> Bump/shake table (tremor perturbs all props)")
        print("    • 'adversary on/off'    -> Auto-nudge target mid-reach to test visual servoing!")
        print("")
        print("  [Interactive 3D Window Controls]:")
        print("    • [Double-Click]            -> Select any prop on table (auto-snaps if clicked near!)")
        print("    • [1, 2, 3...] or [N]       -> Instantly select prop 1, 2, 3... or cycle next")
        print("    • [Ctrl + Right-Click Drag] -> Drag selected prop across table in 3D!")
        print("    • [Arrow Keys (←, →, ↑, ↓)] -> Smoothly glide selected prop (±4cm)")
        print("    • [D] Displace prop (+12cm) | [K] Knock over prop | [F] Drop prop onto floor")
        print("    • [T] Shake table")
        print("")
        print("  [System]:")
        print("    • 'objects' -> Show live 3D coordinates | 'status' -> Show robot arms")
        print("    • 'help'    -> Reprint this menu        | 'exit'   -> Shutdown")
        print("=" * 78 + "\n")

    def _terminal_cli_worker(self):
        """Background interactive CLI console thread."""
        time.sleep(1.0)
        self.print_menu()

        while self.is_running:
            try:
                raw = input("🤖 [Margaret Ready] > ").strip()
                if not raw:
                    continue

                low = raw.lower()
                if low in ["exit", "quit", "q"]:
                    print("\n👋 Shutting down Margaret Agent System...")
                    self.is_running = False
                    break

                elif low in ["help", "h", "?"]:
                    self.print_menu()

                elif low in ["objects", "props"]:
                    print("\n📦 Live Grounded Objects:")
                    for p in self.spawned_props:
                        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, p["name"])
                        pos = self.data.xpos[bid].round(3).tolist() if bid != -1 else p["initial_pos"]
                        print(f"   • {p['name']}: pos={pos}, color={p['color']}, shape={p['shape']}")
                    print()

                elif low == "status":
                    print("\n🤖 Margaret Robot Embodiment State:")
                    print(f"   • Right Hand: {self.brain.memory.robot_state.get('right_hand', {})}")
                    print(f"   • Left Hand:  {self.brain.memory.robot_state.get('left_hand', {})}")
                    print(f"   • Held Objects: {self.grasp_ctrl.held_objects if self.grasp_ctrl else {}}")
                    print()

                # Explicit environment disturbance commands
                elif low.startswith("disturb") or low.startswith("chaos") or low.startswith("nudge"):
                    parts = raw.split()
                    if len(parts) > 1 and parts[1].lower() in ["drop", "floor"]:
                        tgt = " ".join(parts[2:]) if len(parts) > 2 else None
                        self.disturb("drop", tgt)
                    elif len(parts) > 1 and parts[1].lower() in ["knock", "tumble"]:
                        tgt = " ".join(parts[2:]) if len(parts) > 2 else None
                        self.disturb("knock", tgt)
                    elif len(parts) > 1 and parts[1].lower() in ["shake", "tremor"]:
                        self.disturb("shake")
                    else:
                        val = 0.12
                        tgt = None
                        for token in parts[1:]:
                            try:
                                val = float(token)
                            except ValueError:
                                tgt = (tgt + " " + token).strip() if tgt else token
                        self.disturb("displace", tgt, val=val)

                elif low in ["shake", "tremor", "earthquake"]:
                    self.disturb("shake")

                elif (low.startswith("knock ") or low.startswith("tumble ")) and not any(w in low for w in ["the", "arm", "hand", "down"]):
                    parts = raw.split(maxsplit=1)
                    tgt = parts[1] if len(parts) > 1 else None
                    self.disturb("knock", tgt)

                elif (low.startswith("drop ") or low.startswith("floor ")) and not any(w in low for w in ["slowly", "table", "in the", "on the", "onto", "down", "gently", "it", "please"]):
                    parts = raw.split(maxsplit=1)
                    tgt = parts[1] if len(parts) > 1 else None
                    self.disturb("drop", tgt)

                elif low.startswith("adversary") or low.startswith("chaos"):
                    parts = low.split()
                    state = None
                    if len(parts) > 1:
                        state = parts[1] in ["on", "true", "1", "enable"]
                    self.toggle_adversary(state)

                elif low in ["next", "cycle"]:
                    self.active_prop_idx = (self.active_prop_idx + 1) % len(self.spawned_props)
                    tgt = self.get_current_target_name()
                    p_sel = self.spawned_props[self.active_prop_idx]
                    print(f"🎯 Switched active target prop [{self.active_prop_idx+1}]: '{tgt}' ({p_sel['color']} {p_sel['shape']})")

                elif low.startswith("select ") or low.startswith("choose ") or low.startswith("pick-prop "):
                    query = raw.split(maxsplit=1)[1].strip().lower()
                    selected_idx = None
                    if query.isdigit():
                        q_num = int(query) - 1
                        if 0 <= q_num < len(self.spawned_props):
                            selected_idx = q_num
                    else:
                        for idx, p in enumerate(self.spawned_props):
                            pname = p["name"].lower()
                            pcolor = p["color"].lower()
                            pshape = p["shape"].lower()
                            if query == pname or query in pname or query == f"{pcolor} {pshape}" or query == pcolor or query == pshape:
                                selected_idx = idx
                                break
                    if selected_idx is not None:
                        self.active_prop_idx = selected_idx
                        p_sel = self.spawned_props[selected_idx]
                        print(f"🎯 Selected active prop [{selected_idx+1}]: '{p_sel['name']}' ({p_sel['color']} {p_sel['shape']})")
                        print("   👉 Drag with [Ctrl + Right-Click] or nudge with [Arrow Keys (←, →, ↑, ↓)]")
                    else:
                        avail = ", ".join(f"[{i+1}] {p['name']}" for i, p in enumerate(self.spawned_props))
                        print(f"⚠️ Could not find prop matching '{query}'. Available props: {avail}")

                else:
                    # Natural language mission for Margaret!
                    # Commands like "drop the green object in the table slowly", "move arm", "pick up...", "place..."
                    print(f"\n🎯 [Margaret Mission Dispatched]: \"{raw}\"")
                    self.execute_mission(raw)

            except (EOFError, KeyboardInterrupt):
                self.is_running = False
                break

    def launch(self, initial_mission: Optional[str] = None):
        """
        Main entrypoint: boots environment, robot, and 3D simulation window on main thread.
        Runs CLI prompt on interactive background thread.
        """
        # 1. Load Environment & Robot
        self.load_environment()

        # 2. Perception Grounding
        self.inspect_scene()

        # 3. Setup Hotkey Handler for 3D Window
        def key_callback(keycode: int):
            # GLFW Arrow keys: LEFT=263, RIGHT=262, UP=265, DOWN=264
            if keycode == 262:  # GLFW_KEY_RIGHT
                self.nudge_active_prop(delta_x=0.0, delta_y=0.04)
                return
            elif keycode == 263:  # GLFW_KEY_LEFT
                self.nudge_active_prop(delta_x=0.0, delta_y=-0.04)
                return
            elif keycode == 265:  # GLFW_KEY_UP
                self.nudge_active_prop(delta_x=0.04, delta_y=0.0)
                return
            elif keycode == 264:  # GLFW_KEY_DOWN
                self.nudge_active_prop(delta_x=-0.04, delta_y=0.0)
                return

            # Number keys 1, 2, 3... to select prop directly
            if 49 <= keycode <= 57:
                idx = keycode - 49
                if idx < len(self.spawned_props):
                    self.active_prop_idx = idx
                    tgt = self.get_current_target_name()
                    p_info = self.spawned_props[idx]
                    print(f"\n🎯 [KEY {idx+1}]: Selected active prop '{tgt}' ({p_info['color']} {p_info['shape']})")
                    print("   👉 Drag with [Ctrl + Right-Click] or glide with [Arrow Keys (←, →, ↑, ↓)]")
                return

            try:
                key_char = chr(keycode).upper()
            except (ValueError, OverflowError):
                return

            if key_char == "D":
                tgt = self.get_current_target_name()
                self.disturb("displace", tgt, val=0.12)
            elif key_char == "K":
                tgt = self.get_current_target_name()
                self.disturb("knock", tgt)
            elif key_char == "F":
                tgt = self.get_current_target_name()
                self.disturb("drop", tgt)
            elif key_char == "T":
                self.disturb("shake")
            elif key_char == "N":
                self.active_prop_idx = (self.active_prop_idx + 1) % len(self.spawned_props)
                tgt = self.get_current_target_name()
                p_info = self.spawned_props[self.active_prop_idx]
                print(f"\n🎯 [HOTKEY N]: Switched active prop to: '{tgt}' ({p_info['color']} {p_info['shape']})")
                print("   👉 Drag with [Ctrl + Right-Click] or glide with [Arrow Keys (←, →, ↑, ↓)]")
            elif key_char == "H":
                self.print_menu()

        self.is_running = True

        # 4. Start Interactive Console Thread
        cli_thread = threading.Thread(target=self._terminal_cli_worker, daemon=True)
        cli_thread.start()

        # 5. Optional Initial Mission
        if initial_mission:
            time.sleep(1.2)
            cmd = initial_mission
            if cmd.lower() == "auto":
                cmd = f"pick up {self.get_current_target_name()}"
            self.execute_mission(cmd)

        # 6. Launch Native 3D Simulation Window on Main Thread (GLFW requirement)
        print("\n🚀 Launching Native 3D Simulation Window...")
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback) as viewer:
            # Set camera directly facing workbench in foreground with props crystal clear
            viewer.cam.lookat[:] = [0.35, 0.0, 0.65]
            viewer.cam.distance = 1.55
            viewer.cam.azimuth = 155
            viewer.cam.elevation = -25

            last_drag_sel = None
            last_gui_sel = None
            last_synced_idx = None

            # Collect valid table prop body IDs (prevent perturbing robot links!)
            prop_bids = {
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, p["name"]): p["name"]
                for p in self.spawned_props
            }
            prop_bids_inv = {name: bid for bid, name in prop_bids.items()}
            workbench_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "workbench")

            while viewer.is_running() and self.is_running:
                step_start = time.time()

                # Step physics safely under physics_lock
                steps_per_frame = max(1, int(0.016 / self.model.opt.timestep))
                with self.physics_lock:
                    # Mouse Perturbation Management (Locked to table props only!)
                    if hasattr(viewer, "perturb"):
                        current_sel = viewer.perturb.select

                        # 1. Snap-selection: If user double-clicked on table surface near a prop, snap to nearest prop!
                        if workbench_bid != -1 and current_sel == workbench_bid:
                            click_x = self.data.xpos[workbench_bid][0] + viewer.perturb.localpos[0]
                            click_y = self.data.xpos[workbench_bid][1] + viewer.perturb.localpos[1]
                            min_dist = float("inf")
                            best_idx = None
                            for idx, p in enumerate(self.spawned_props):
                                p_bid = prop_bids_inv.get(p["name"])
                                if p_bid is not None and p_bid != -1:
                                    p_pos = self.data.xpos[p_bid]
                                    d_xy = float(np.hypot(p_pos[0] - click_x, p_pos[1] - click_y))
                                    if d_xy < min_dist:
                                        min_dist = d_xy
                                        best_idx = idx
                            if best_idx is not None and min_dist < 0.25:
                                self.active_prop_idx = best_idx
                                best_pname = self.spawned_props[best_idx]["name"]
                                best_bid = prop_bids_inv[best_pname]
                                viewer.perturb.select = best_bid
                                current_sel = best_bid
                                last_gui_sel = best_bid
                                last_synced_idx = best_idx
                                p_info = self.spawned_props[best_idx]
                                print(f"\n🎯 [3D Window]: Clicked table near '{best_pname}' ({p_info['color']} {p_info['shape']}) -> Selected!")

                        # 2. If user double-clicked directly on a prop in GUI, update active_prop_idx
                        if current_sel in prop_bids and current_sel != last_gui_sel:
                            last_gui_sel = current_sel
                            sel_name = prop_bids[current_sel]
                            for idx, p in enumerate(self.spawned_props):
                                if p["name"] == sel_name:
                                    self.active_prop_idx = idx
                                    last_synced_idx = idx
                                    print(f"\n🎯 [3D Window]: Mouse selected '{sel_name}'! (Hold Ctrl + Right-Click to drag)")
                                    break

                        # 3. If active_prop_idx was changed (via keyboard 1..9, N or CLI), sync viewer selection
                        if self.active_prop_idx != last_synced_idx:
                            last_synced_idx = self.active_prop_idx
                            active_tgt = self.get_current_target_name()
                            active_bid = prop_bids_inv.get(active_tgt, -1)
                            if active_bid != -1:
                                viewer.perturb.select = active_bid
                                last_gui_sel = active_bid
                                current_sel = active_bid

                        # 4. If selection is not a prop (e.g. robot link or empty space), clamp to active target prop!
                        if current_sel not in prop_bids:
                            active_tgt = self.get_current_target_name()
                            active_bid = prop_bids_inv.get(active_tgt, -1)
                            if active_bid != -1:
                                viewer.perturb.select = active_bid
                                last_gui_sel = active_bid
                                current_sel = active_bid

                        # 5. Apply drag motion while user holds mouse button (Ctrl + Right-Click Drag)
                        if viewer.perturb.active and current_sel in prop_bids:
                            sel_name = prop_bids[current_sel]

                            # If dragged object is held by robot, release robot grip immediately BEFORE updating position!
                            if self.grasp_ctrl and sel_name:
                                for h in ["right", "left"]:
                                    if self.grasp_ctrl.held_objects.get(h) == sel_name:
                                        print(f"🔓 [Mouse Drag]: Disengaging robot grip on '{sel_name}'...")
                                        self.grasp_ctrl.execute_release(sel_name, hand=h)
                                        if self.brain and self.brain.memory:
                                            self.brain.memory.set_hand_held(h, None)
                                            self.brain.memory.set_hand_lifted(h, False)

                            jadr = self.model.body_jntadr[current_sel]
                            if jadr != -1 and self.model.jnt_type[jadr] == mujoco.mjtJoint.mjJNT_FREE:
                                qadr = self.model.jnt_qposadr[jadr]
                                ref = viewer.perturb.refpos
                                if not (ref[0] == 0.0 and ref[1] == 0.0 and ref[2] == 0.0):
                                    # Direct translation tracking cursor in 3D
                                    self.data.qpos[qadr] = ref[0]
                                    self.data.qpos[qadr + 1] = ref[1]
                                    min_z = (self.robot.table_surface_z + 0.02) if self.robot else 0.64
                                    self.data.qpos[qadr + 2] = max(min_z, ref[2])  # Maintain on/above tabletop
                                    # Zero velocities so it glides smoothly without spinning
                                    dofadr = self.model.body_dofadr[current_sel]
                                    if dofadr != -1:
                                        self.data.qvel[dofadr : dofadr + 6] = 0.0
                                    mujoco.mj_forward(self.model, self.data)

                            # External perturbation on object in physical world (no 3D coordinates broadcast)
                            if last_drag_sel != current_sel:
                                last_drag_sel = current_sel
                                if self.brain:
                                    self.brain.post_event({
                                        "event_type": "ENVIRONMENT_CHANGE",
                                        "description": f"External disturbance detected: Prop '{sel_name}' is being moved",
                                        "payload": {"object_name": sel_name},
                                        "timestamp": time.time(),
                                    })
                        else:
                            # Clear external applied forces once mouse is released so object rests naturally
                            self.data.xfrc_applied[:] = 0.0
                            last_drag_sel = None

                    # 6. Live HUD overlay in 3D Window
                    if hasattr(viewer, "set_texts"):
                        active_tgt = self.get_current_target_name()
                        p_info = self.spawned_props[self.active_prop_idx % len(self.spawned_props)]
                        hud_col1 = f"TARGET: [{self.active_prop_idx+1}] {p_info['color'].upper()} {p_info['shape'].upper()} ({active_tgt})"
                        hud_col2 = "KEYS: [1..9/N]: Select | [Ctrl+RClick/Arrows]: Drag"
                        viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, hud_col1, hud_col2)])

                    for _ in range(steps_per_frame):
                        if hasattr(self.robot, "apply_balance_stabilization"):
                            self.robot.apply_balance_stabilization(self.model, self.data)
                        mujoco.mj_step(self.model, self.data)

                    viewer.sync()

                dt_sleep = (1.0 / 60.0) - (time.time() - step_start)
                if dt_sleep > 0:
                    time.sleep(dt_sleep)

        self.is_running = False
        print("\n✅ Simulation session ended. Margaret Agent standing down.")


def parse_args():
    parser = argparse.ArgumentParser(description="Margaret Embodied AI Autonomous Agent")
    parser.add_argument("--robot", type=str, default="unitree_g1", choices=["unitree_g1", "aloha", "limx_tron1", "tron1", "wheeled_biped"], help="Robot embodiment")
    parser.add_argument("--aloha", action="store_true", default=False, help="Shorthand for --robot aloha (desktop bimanual ViperX dual-arm embodiment)")
    parser.add_argument("--tron1", action="store_true", default=False, help="Shorthand for --robot tron1 (LimX Dynamics TRON 1 biped-wheeled robot)")
    parser.add_argument("--objects", type=int, default=3, help="Number of procedural objects")
    parser.add_argument("--mission", type=str, default=None, help="Optional initial mission on launch")
    parser.add_argument("--no-yolo", dest="yolo", action="store_false", default=True, help="Disable local YOLO-World vision model (enabled by default)")
    parser.add_argument("--adversary", action="store_true", default=False, help="Start with auto-adversary disturbance mode active")
    args = parser.parse_args()
    if args.aloha:
        args.robot = "aloha"
    if args.tron1:
        args.robot = "limx_tron1"
    return args


def main():
    args = parse_args()
    agent = MargaretAgent(
        robot=args.robot,
        num_objects=args.objects,
        use_yolo=args.yolo,
    )
    if args.adversary:
        agent.adversary_enabled = True
    agent.launch(initial_mission=args.mission)


if __name__ == "__main__":
    main()
