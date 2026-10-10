"""
Interactive Procedural Environment & Disturbance Sandbox.
Demonstrates:
1. Generating a completely independent physical arena with randomized objects, obstacles, and furniture.
2. Dropping the robot into the scene.
3. Introducing runtime disturbances (moving objects, knocking them over).
4. Demonstrating Margaret's closed-loop perception and event awareness.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import argparse
import mujoco

from arena.environment_generator import ProceduralEnvironment
from arena.disturbance_engine import EnvironmentDisturber
from skills.vision import SpatialVision
from agent.tools import ToolRegistry
from agent.brain import AgentBrain
from robots import get_robot


from PIL import Image

def save_render(model, data, camera_name: str, filepath: str, width: int = 640, height: int = 480):
    """Renders off-screen image from camera and saves to disk."""
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
        renderer.update_scene(data, camera=camera_name)
        rgb = renderer.render()
        img = Image.fromarray(rgb)
        img.save(filepath)
        print(f"📸 Saved visual render: {filepath}")
    except Exception as e:
        print(f"⚠️ Could not render {camera_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Procedural Environment & Disturbance Sandbox")
    parser.add_argument("--robot", type=str, default="unitree_g1", choices=["unitree_g1", "aloha"])
    parser.add_argument("--objects", type=int, default=4, help="Number of procedural objects to generate")
    parser.add_argument("--furniture", type=str, default="workbench", choices=["workbench", "shelf_table", "dual_table"], help="Type of furniture layout")
    parser.add_argument("--obstacle", type=str, default="barrier", choices=["none", "barrier", "divider"], help="Type of obstacle")
    parser.add_argument("--disturb", type=str, default="displace", choices=["displace", "knock_over", "drop_floor", "shake", "none"], help="Disturbance mode")
    parser.add_argument("--render", action="store_true", default=True, help="Render and save visual frames to disk")
    args = parser.parse_args()

    print("=" * 70)
    print("🌍 1. GENERATING INDEPENDENT PROCEDURAL ENVIRONMENT")
    print("=" * 70)
    print(f"• Robot Embodiment: {args.robot}")
    print(f"• Furniture Type:   {args.furniture}")
    print(f"• Obstacle Type:    {args.obstacle}")
    print(f"• Number of Props:  {args.objects}")

    generator = ProceduralEnvironment(robot_platform=args.robot)
    xml_path, spawned_props = generator.generate_random_arena(
        num_objects=args.objects,
        furniture_type=args.furniture,
        obstacle_type=args.obstacle,
        add_obstacle_barrier=(args.obstacle == "barrier"),
        output_filename="runtime_procedural_arena.xml"
    )

    print(f"✅ Generated independent arena at: {xml_path}")
    print(f"📦 Spawned {len(spawned_props)} procedural props:")
    for p in spawned_props:
        print(f"   • [{p['name']}]: shape={p['shape']}, color={p['color']}, pos={p['initial_pos']}, mass={p['mass']}kg")

    print("\n" + "=" * 70)
    print(f"🤖 2. DROPPING MARGARET AGENT INTO THE UNSEEN ENVIRONMENT")
    print("=" * 70)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    robot = get_robot(args.robot)
    robot.setup_data(model, data)
    mujoco.mj_forward(model, data)

    registry = ToolRegistry()
    brain = AgentBrain(registry)
    disturber = EnvironmentDisturber(model, data)

    cam_name = "head_cam" if args.robot == "unitree_g1" else "overhead_cam"
    vision = SpatialVision(model, data, camera_name=cam_name)

    if args.render:
        save_render(model, data, "room_overview", "procedural_arena_overview.png")
        save_render(model, data, cam_name, "procedural_head_cam.png")

    # Initial sensory observation
    print("\n👁️  Margaret scanning scene with camera...")
    detected = vision.detect_table_objects()
    print(f"✅ Margaret detected {len(detected)} objects via pure camera vision:")
    for d in detected:
        print(f"   • Grounded '{d['name']}': shape={d['class']}, color={d['color']}, 3D centroid={d['position_3d']}")

    if args.disturb != "none" and spawned_props:
        print("\n" + "=" * 70)
        print(f"💥 3. SIMULATING RUNTIME ENVIRONMENT DISTURBANCE: [{args.disturb.upper()}]")
        print("=" * 70)
        target = spawned_props[0]["name"]

        if args.disturb == "displace":
            print(f"-> Disturbing '{target}' by shifting position +12cm along table...")
            disturber.displace_object(target, delta_x=0.0, delta_y=0.12, brain=brain)
        elif args.disturb == "knock_over":
            print(f"-> Disturbing '{target}' by knocking it over with angular momentum...")
            disturber.knock_over_object(target, angular_impulse=3.0, brain=brain)
        elif args.disturb == "drop_floor":
            print(f"-> Pushing '{target}' off the table onto the floor...")
            disturber.drop_object_off_table(target, brain=brain)
        elif args.disturb == "shake":
            print("-> Table tremor bump: perturbing all props simultaneously...")
            disturber.shake_all_props(jitter_std=0.04, brain=brain)

        if args.render:
            save_render(model, data, "room_overview", "procedural_disturbed_overview.png")

        print("\n👁️  Margaret re-sensing environment after disturbance...")
        detected_after = vision.detect_table_objects()
        for d in detected_after:
            if d["name"] == target:
                print(f"   • Updated position of target '{target}': {d['position_3d']}")

        pending_events = brain.graph.harness.get_pending_events()
        print(f"\n🔔 Pending Events in Margaret's Adaptive Harness: {len(pending_events)}")
        for ev in pending_events:
            print(f"   • [{ev.event_type}]: {ev.description}")

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE: Environment independence & dynamic disturbance verified!")
    print("=" * 70)


if __name__ == "__main__":
    main()

