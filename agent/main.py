"""
Margaret Embodied AI Agent Main Runner.
Location: agent/main.py

Usage:
  uv run python -m agent
  or
  uv run python agent/main.py

Flow:
1. Margaret Agent boots up.
2. Margaret Agent configures and loads the physical environment (plain workbench, zero obstruction sheets).
3. Margaret Agent loads the robot embodiment (Unitree G1 humanoid).
4. Margaret Agent initializes and benchmarks the vision model (SpatialVision / YOLO-World).
5. Margaret Agent initializes kinematics, 50Hz closed-loop visual servoing, and cognitive ReAct brain.
6. Margaret Agent launches the native 3D simulation window with full mouse drag and hotkey perturbation support.
7. Margaret Agent enters the interactive console for natural language commands and runtime disturbances.
"""

import os
import sys
import argparse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.embodied_agent import MargaretAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Margaret Autonomous Embodied AI Agent")
    parser.add_argument("--robot", type=str, default="unitree_g1", choices=["unitree_g1", "aloha", "limx_tron1", "tron1", "wheeled_biped"], help="Robot embodiment")
    parser.add_argument("--aloha", action="store_true", default=False, help="Shorthand for --robot aloha (desktop bimanual ViperX dual-arm embodiment)")
    parser.add_argument("--tron1", action="store_true", default=False, help="Shorthand for --robot tron1 (LimX Dynamics TRON 1 biped-wheeled robot)")
    parser.add_argument("--objects", type=int, default=3, help="Number of procedural objects")
    parser.add_argument("--furniture", type=str, default="workbench", choices=["workbench", "shelf_table", "dual_table"], help="Furniture layout")
    parser.add_argument("--obstacle", type=str, default="none", choices=["none", "barrier", "divider"], help="Obstacle type")
    parser.add_argument("--mission", type=str, default=None, help="Initial task for Margaret to execute immediately on launch")
    parser.add_argument("--no-yolo", dest="yolo", action="store_false", default=True, help="Disable local YOLO-World neural vision model (enabled by default)")
    parser.add_argument("--adversary", action="store_true", default=False, help="Enable auto-adversary mode (perturbs target mid-reach)")
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
        furniture=args.furniture,
        obstacle=args.obstacle,
        use_yolo=args.yolo,
    )
    if args.adversary:
        agent.adversary_enabled = True
    agent.launch(initial_mission=args.mission)


if __name__ == "__main__":
    main()
