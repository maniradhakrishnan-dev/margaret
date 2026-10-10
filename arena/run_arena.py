"""
3D Interactive Arena Launcher
Stage 1: Verified Environment with Humanoid, Workbench, and Physics Props.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import mujoco
import mujoco.viewer
import numpy as np

import argparse
from robots import get_robot

def main():
    parser = argparse.ArgumentParser(description="3D Interactive Arena Viewer")
    parser.add_argument("--robot", type=str, default="unitree_g1", help="Robot platform: 'unitree_g1' or 'default'")
    args = parser.parse_args()

    robot = get_robot(args.robot)
    arena_path = robot.model_path

    print("=" * 65)
    print(f"🏟️  LOADING 3D ROBOT ARENA: {robot.name.upper()}")
    print("=" * 65)
    
    if not os.path.exists(arena_path):
        raise FileNotFoundError(f"Cannot find arena file at {arena_path}")
    
    # 1. Load Arena Physics Model
    model = mujoco.MjModel.from_xml_path(arena_path)
    data = mujoco.MjData(model)
    robot.setup_data(model, data)

    print(f"✅ Arena Loaded Successfully from: {arena_path}")
    print(f"• Total Bodies in Scene: {model.nbody}")
    print(f"• Total Degrees of Freedom: {model.nv}")
    print(f"• Total Actuators / Motors: {model.nu}")
    print(f"• Total Geometries (Geoms): {model.ngeom}")

    # Inspect props on the table
    print("\n📦 Interactive Props in Scene:")
    ignored_bodies = {"world", "workbench", "table", "floor", "pelvis"}
    robot_prefixes = ("left_", "right_", "left/", "right/", "waist", "torso", "pelvis", "head", "neck", "base", "flange")
    for bid in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not bname or bname in ignored_bodies or bname.startswith(robot_prefixes):
            continue
        if model.body_jntnum[bid] > 0:
            print(f"  - [{bname}] ready with 6-DoF rigid-body physics at pos {data.xpos[bid].round(3)}!")

    # Inspect cameras
    print("\n📷 Cameras available:")
    for i in range(model.ncam):
        cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        print(f"  - [{i}] {cam_name}")

    print("\n" + "=" * 65)
    print("🎮 INTERACTIVE VIEWER CONTROLS:")
    print("  - [SPACEBAR]: Pause / Resume physics simulation.")
    print("  - [BACKSPACE]: Reset scene to initial state.")
    print("  - [Ctrl + Right-Click & Drag]: Grab any object (mug, ball, block,")
    print("    or robot limb) with your mouse to move or throw it!")
    print("  - Close the window to exit.")
    print("=" * 65 + "\n")

    # 2. Launch the 3D GUI Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.4
        viewer.cam.elevation = -18
        viewer.cam.azimuth = 140

        while viewer.is_running():
            step_start = time.time()

            # Step physics forward by 1 timestep (1 ms)
            mujoco.mj_step(model, data)

            # Sync graphical display with physics
            viewer.sync()

            # Maintain real-time simulation clock (1000 Hz)
            dt_sleep = model.opt.timestep - (time.time() - step_start)
            if dt_sleep > 0:
                time.sleep(dt_sleep)

if __name__ == "__main__":
    main()
