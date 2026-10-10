"""
Master Agentic Humanoid System
Unifies Agent Brain, World Model, Vision, Kinematics, and MuJoCo Physics into one living loop.
Supports both MuJoCo Menagerie (Unitree G1 with dexterous hands) and Custom Humanoid test rigs.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import argparse
import threading
import mujoco
import mujoco.viewer
import numpy as np
from typing import Optional

from robots import get_robot
from skills.vision import SpatialVision
from skills.reach_controller import ReachController
from skills.grasp import GraspController
from agent.world_model import WorldModel
from agent.tools import ToolRegistry
from agent.brain import AgentBrain
from ros2_bridge import HumanoidRosBridge

def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous Physical AI Humanoid Agent")
    parser.add_argument(
        "--robot",
        type=str,
        default="unitree_g1",
        help="Robot platform: 'unitree_g1' (MuJoCo Menagerie Flagship) or 'default' (27-DoF Test Rig)",
    )
    parser.add_argument(
        "--mission",
        type=str,
        default=None,
        help="Optional initial mission to execute on startup. If omitted, waits for user instruction.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep session open and prompt for subsequent missions dynamically",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening interactive 3D GUI (for headless servers or test suites)",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 72)
    print("🤖 LAUNCHING MARGARET - AUTONOMOUS PHYSICAL AI AGENT")
    print("   Architecture: Cognitive ReAct + World Model + Vision + IK + MuJoCo")
    print(f"   Robot Platform: {args.robot.upper()}")
    print("=" * 72)

    # 1. Load Robot Platform & Arena
    robot = get_robot(args.robot)
    print(f"📦 Loading Arena Model: {robot.model_path}")
    model = mujoco.MjModel.from_xml_path(robot.model_path)
    data = mujoco.MjData(model)

    # 2. Initialize Robot Pose (e.g. keyframe stand)
    robot.setup_data(model, data)
    print(f"✨ Initialized {robot.name} with {model.nu} actuators ({len(robot.arm_joint_names)} arm DoFs, {len(robot.finger_actuator_names)} finger DoFs).")

    # 3. Settle initial contact dynamics
    for _ in range(400):
        mujoco.mj_step(model, data)

    # 4. Instantiate Physical AI Subsystems with Reentrant Thread Lock (None in headless mode)
    physics_lock = None if args.headless else threading.RLock()
    vision = SpatialVision(model, data, camera_name="head_cam", lock=physics_lock)
    reach_ctrl = ReachController(model, data, robot, lock=physics_lock)
    grasp_ctrl = GraspController(model, data, robot, lock=physics_lock)
    world_model = WorldModel(
        model=model,
        robot=robot,
        arm_max_reach_m=robot.max_reach_m,
        table_surface_z=robot.table_surface_z,
    )

    # 5. Build Tool Registry & Register Physical Skills
    registry = ToolRegistry()

    @registry.register(name="inspect_scene", description="Inspects the workbench with camera and detects 3D objects.")
    def tool_inspect_scene():
        objects = vision.detect_table_objects()
        for obj in objects:
            brain.memory.update_object(obj["name"], obj["position_3d"], metadata={"status": obj.get("status", "on_table")})
        return {"detected_count": len(objects), "objects": objects}

    @registry.register(name="reach_hand", description="Uses Inverse Kinematics and continuous visual servoing to move palm to 3D target coordinates.")
    def tool_reach_hand(target_x: float, target_y: float, target_z: float, hand: str = "auto", object_name: Optional[str] = None):
        # 1. World Model imagination check
        mental_check = world_model.simulate_reach(robot.shoulder_pos, [target_x, target_y, target_z])
        print(f"   🔮 [World Model Imagination]: {mental_check['predicted_outcome']} (Confidence: {mental_check['confidence']*100:.0f}%)")
        
        if mental_check["recommendation"] != "EXECUTE":
            return {"success": False, "error": f"World Model aborted action: {mental_check['predicted_outcome']}"}

        # 2. Execute physical reach in MuJoCo (with continuous visual servoing if object specified)
        res = reach_ctrl.execute_reach(
            target_x, target_y, target_z,
            hand=hand,
            duration_s=1.2,
            steps=40,
            target_object=object_name,
            vision=vision,
        )
        actual_hand = res.get("hand", hand if hand in ["right", "left"] else ("left" if target_y > 0.04 else "right"))
        
        # Find which object is closest to this target
        best_obj, min_d = None, float("inf")
        for obj_name, obj_data in brain.memory.objects.items():
            d = np.linalg.norm(np.array(obj_data["pos"]) - np.array([target_x, target_y, target_z]))
            if d < min_d:
                min_d = d
                best_obj = obj_name
        
        if res.get("success") and best_obj and min_d < 0.15:
            brain.memory.set_hand_reached(actual_hand, best_obj)
        else:
            brain.memory.set_hand_reached(actual_hand, None)
            
        h_key = f"{actual_hand}_hand"
        if h_key in brain.memory.robot_state and "reached_position" in res:
            brain.memory.robot_state[h_key]["pos"] = res["reached_position"]
        brain.memory.robot_state["hand_pos"] = res.get("reached_position")
        return res

    @registry.register(name="grasp_object", description="Closes fingers around target object and verifies tactile grip.")
    def tool_grasp_object(object_name: str, hand: str = "auto"):
        # 1. World Model friction check
        friction_check = world_model.simulate_grasp(object_mass=0.15, grip_force_N=12.5)
        print(f"   🔮 [World Model Friction]: {friction_check['status']} (Safety Margin: {friction_check['safety_margin']}x)")

        # 2. Execute physical grasp
        res = grasp_ctrl.execute_grasp(object_name, hand=hand)
        if res.get("success"):
            actual_hand = res.get("hand", "right")
            brain.memory.set_hand_held(actual_hand, object_name)
        return res

    @registry.register(name="lift_hand", description="Lifts the arm and held object upwards from the table.")
    def tool_lift_hand(height: float = 0.15, hand: str = "auto"):
        res = grasp_ctrl.execute_lift(height=height, hand=hand, steps=40)
        if res.get("success"):
            actual_hand = res.get("hand", "right")
            brain.memory.set_hand_lifted(actual_hand, True)
            held_obj = grasp_ctrl.held_objects.get(actual_hand)
            if held_obj and held_obj in brain.memory.objects:
                brain.memory.objects[held_obj]["pos"][2] += height
        return res

    @registry.register(name="place_object", description="Transports held object and places it gently onto target.")
    def tool_place_object(target_x: Optional[float] = None, target_y: Optional[float] = None, target_z: Optional[float] = None, hand: str = "auto", dest_object: Optional[str] = None):
        target_obj = grasp_ctrl.held_objects.get("right") or grasp_ctrl.held_objects.get("left")
        res = grasp_ctrl.execute_place(target_x, target_y, target_z, hand=hand, dest_object=dest_object, duration_s=1.2, steps=40)
        if res.get("success"):
            actual_hand = res.get("hand", "right")
            brain.memory.set_hand_held(actual_hand, None)
            brain.memory.set_hand_reached(actual_hand, None)
            brain.memory.set_hand_lifted(actual_hand, False)
        if target_obj:
            obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_obj)
            if obj_id != -1:
                real_pos = [round(float(p), 3) for p in data.xpos[obj_id]]
                real_status = "on_table" if real_pos[2] > 0.55 else "on_floor"
                brain.memory.update_object(target_obj, real_pos, metadata={"status": real_status})
        return res

    @registry.register(name="release_object", description="Opens fingers and releases held object.")
    def tool_release_object(object_name: Optional[str] = None, hand: str = "auto"):
        res = grasp_ctrl.execute_release(object_name=object_name, hand=hand)
        if res.get("success"):
            actual_hand = res.get("hand", "right")
            brain.memory.set_hand_held(actual_hand, None)
            brain.memory.set_hand_reached(actual_hand, None)
            brain.memory.set_hand_lifted(actual_hand, False)
        return res

    @registry.register(name="recover_floor_object", description="Retrieves an object that fell onto the floor and resets it safely onto the table.")
    def tool_recover_floor_object(object_name: str, target_x: Optional[float] = None, target_y: Optional[float] = None):
        res = grasp_ctrl.recover_object(object_name, target_x=target_x, target_y=target_y)
        if res.get("success"):
            brain.memory.update_object(object_name, res["position"], metadata={"status": "on_table"})
        return res

    # 6. Instantiate Cognitive Agent Brain (Margaret)
    brain = AgentBrain(registry, world_model=world_model)

    # 7. Initialize ROS 2 Jazzy Robotics Bridge
    ros_bridge = HumanoidRosBridge(model, data, vision=vision, lock=physics_lock, robot_name=robot.name)

    # 8. Agent Mission Worker Thread
    mission_completed_event = threading.Event()

    def agent_mission_thread():
        time.sleep(1.0)  # Allow viewer window to open and settle
        if args.mission:
            brain.run_mission(args.mission)
        if args.interactive or not args.headless:
            print("\n" + "=" * 70)
            print("💬 NATURAL LANGUAGE INTERACTIVE INTERFACE READY")
            print("   You can enter natural language instructions, e.g.:")
            print("   - 'pick up the blue block'")
            print("   - 'pickup the red object and put it on the blue object'")
            print("   - 'release the object'")
            print("   - 'pick up the green ball'")
            print("   - 'lift it higher'")
            print("   - 'inspect the workbench'")
            print("   (Type 'quit' or 'exit' to conclude)")
            print("=" * 70)
            while True:
                try:
                    next_goal = input("\n🤖 Enter instruction for Margaret: ").strip()
                    if not next_goal or next_goal.lower() in ["quit", "exit"]:
                        break
                    brain.run_mission(next_goal)
                except (EOFError, KeyboardInterrupt):
                    break
        mission_completed_event.set()

    # 9. Execution Mode (Headless vs Interactive GUI)
    if args.headless:
        print("\n⚙️ Running in Headless Simulation Mode...")
        if args.mission:
            brain.run_mission(args.mission)
        print("✅ Headless mission execution completed.")
        ros_bridge.shutdown()
        return

    # Interactive Viewer Mode: Spawn Agent Worker Thread & Open 3D Viewer
    mission_thread = threading.Thread(target=agent_mission_thread, daemon=True)
    mission_thread.start()

    # Interactive 3D Viewer Mode (Passive Viewer on Main Thread)
    print("\n🚀 Opening Interactive 3D Simulation...")
    print("💡 Watch the terminal output as the Agent reasons and calls physical tools in real time!\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.2
        viewer.cam.elevation = -18
        viewer.cam.azimuth = 145

        while viewer.is_running():
            step_start = time.time()

            # Step physics to match 60 FPS display (approx 8 physics steps of 0.002s per frame = 1:1 real time)
            steps_per_frame = max(1, int(0.016 / model.opt.timestep))
            with physics_lock:
                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)
                # Sync graphics safely under physics lock
                viewer.sync()

            # Maintain 60 Hz display refresh
            dt_sleep = (1.0 / 60.0) - (time.time() - step_start)
            if dt_sleep > 0:
                time.sleep(dt_sleep)

if __name__ == "__main__":
    main()
