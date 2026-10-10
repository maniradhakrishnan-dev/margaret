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

# Path to the humanoid model
MODEL_PATH = os.path.join(
    PROJECT_ROOT, "mujoco-3.15.0-linux-x86_64/mujoco-3.15.0/model/humanoid/humanoid.xml"
)

def inspect_model(model: mujoco.MjModel):
    """Inspects the kinematic structure and actuators of the humanoid."""
    print("=" * 60)
    print("🤖 HUMANOID MODEL INSPECTION")
    print("=" * 60)
    print(f"Total Generalized Coordinates (nq): {model.nq}")
    print(f"Total Degrees of Freedom (nv):       {model.nv}")
    print(f"Total Actuators / Motors (nu):       {model.nu}")
    print(f"Total Bodies:                        {model.nbody}")
    print(f"Total Joints:                        {model.njnt}")
    print("-" * 60)
    
    print("\n--- JOINTS & DEGREES OF FREEDOM ---")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jnt_type = model.jnt_type[i]
        # Joint types: 0: free, 1: ball, 2: slide, 3: hinge
        type_str = ["free (6-DoF)", "ball (3-DoF)", "slide (1-DoF)", "hinge (1-DoF)"][jnt_type]
        print(f"[{i:2d}] {name:<22} Type: {type_str}")

    print("\n--- ACTUATORS (MOTORS) ---")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        ctrlrange = model.actuator_ctrlrange[i]
        print(f"[{i:2d}] {name:<22} Control Range: [{ctrlrange[0]:.2f}, {ctrlrange[1]:.2f}]")
    print("=" * 60)

def main():
    print(f"Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Cannot find model at {MODEL_PATH}")

    # 1. Load static model (MjModel) and dynamic state (MjData)
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # 2. Inspect joints and actuators in terminal
    inspect_model(model)

    # 3. Launch interactive 3D viewer directly from Python
    print("\n🚀 Launching Interactive Python 3D Viewer...")
    print("💡 In the viewer:")
    print("   - Press SPACEBAR to pause / unpause.")
    print("   - Double Right-Click any limb and Ctrl+Drag to apply forces.")
    print("   - Close the viewer window to exit.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running():
            step_start = time.time()

            # Apply a visible wave motion to the right arm (shoulder1_right and elbow_right)
            shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder1_right")
            elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "elbow_right")

            t = time.time() - start_time
            if shoulder_id != -1:
                # Oscillate shoulder forward and back
                data.ctrl[shoulder_id] = 1.0 * np.sin(2.0 * np.pi * 0.8 * t)
            if elbow_id != -1:
                # Oscillate elbow bend
                data.ctrl[elbow_id] = 1.0 * np.cos(2.0 * np.pi * 0.8 * t)

            # Step the physics forward by 1 time-step (default 0.002s = 2ms)
            mujoco.mj_step(model, data)

            # Sync the physics state with the 3D graphical viewer
            viewer.sync()

            # Maintain real-time simulation speed
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
