"""
Interactive Agent-First Simulation Runner with Real-Time Environment Disturbances.
Launches the MargaretAgent system:
- Plain wooden workbench table (zero upper sheets or obstacles)
- Unitree G1 Humanoid robot embodiment
- Live 3D simulation with 60 FPS physics and native mouse drag perturbation
- Interactive Agent Console for commanding missions and injecting physical disturbances
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.embodied_agent import MargaretAgent, parse_args


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
