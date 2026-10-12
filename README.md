# 🤖 Margaret: Embodiment-Agnostic Physical AI Agentic System

An end-to-end, embodiment-agnostic physical AI platform combining **Agentic Cognitive AI**, **Frontier Models (Gemini)**, **Pre-Simulation World Models**, **Multi-Robot MuJoCo Menagerie**, **Jacobian Kinematics**, **Continuous Visual Servoing**, and **ROS 2 Jazzy**.

Built entirely from scratch in clean, transparent Python — strictly **without** bloated third-party agent frameworks.

---

## 🌟 Key Architecture & Highlights

```
                                  +-----------------------------+
                                  |   High-Level Mission / Goal |
                                  |   (CLI, Voice, or ROS 2)   |
                                  +--------------+--------------+
                                                 |
                                                 v
                               +----------------------------------+
                               |     Cognitive Agent Brain        |
                               |    (Reason -> Act -> Observe)    |
                               |   Frontier Model / Local ReAct   |
                               +-----------------+----------------+
                                                 |
                                                 v
                               +----------------------------------+
                               |        Latent World Model        |
                               |    Pre-simulation reachability   |
                               |    & slip friction evaluation    |
                               +-----------------+----------------+
                                                 |
                                                 v
                               +----------------------------------+
                               |    Pluggable Robot Adapters      |
                               |    • Unitree G1 (Menagerie)      |
                               |    • Default 27-DoF Test Rig     |
                               |    • ALOHA / Unitree Go2         |
                               +-----------------+----------------+
                                                 |
                    +----------------------------+----------------------------+
                    |                                                         |
                    v                                                         v
   +----------------------------------+                      +----------------------------------+
   |        Physical Skills           |                      |       ROS 2 Jazzy Bridge         |
   | • Spatial Vision RGB-D           |                      | • /robot/joint_states (50 Hz)    |
   | • Damped Least Squares IK (7-DoF)|                      | • /robot/camera/image_raw        |
   | • Dexterous Multi-Finger Grip    |                      | • /robot/agent/status            |
   | • Contact Weld Lift Controller   |                      | • /robot/cmd_mission             |
   +----------------+-----------------+                      +----------------------------------+
                    |
                    v
   +----------------------------------+
   |   MuJoCo Physics Simulation      |
   |   Thread-Safe PD Actuation       |
   |   Realistic 3D Arena Workbench   |
   +----------------------------------+
```

1. **Agent-First Cognitive Architecture (`agent/`):**
   - Pure-Python ReAct loop (`agent/brain.py`) with type-hinted JSON tool registry (`agent/tools.py`).
   - Supports Gemini Frontier Models (`gemini-2.5-flash`) when `GEMINI_API_KEY` is present, with automatic fallback to local deterministic hierarchical planner.
   - 3D spatial memory (`agent/memory.py`) tracking scene objects and robot state.

2. **Pre-Simulation World Model (`agent/world_model.py`):**
   - Mentally rehearses candidate actions before motor actuation.
   - Rejects unfeasible reaches and calculates slip margins against gravitational loads.

3. **Google DeepMind MuJoCo Menagerie Flagship Support (`robots/`):**
   - **Unitree G1 (`unitree_g1`):** Flagship commercial humanoid with 29 degrees of freedom, 43 position-controlled actuators, 7-DoF arms, and 3-finger dexterous hands.
   - **Default Research Humanoid (`default`):** 27-DoF laboratory humanoid test rig.
   - Hot-swappable via `BaseRobot` interface and factory.

4. **Jacobian Numerical Inverse Kinematics (`skills/kinematics.py`):**
   - Sub-Jacobian Damped Least Squares solver computing exact joint angles in native SI units (radians).
   - Smoothly ramped through position servos without physics instabilities or stack overflows.

5. **Dexterous Manipulation (`skills/grasp.py`):**
   - Curls physical finger actuators on Unitree G1 around objects.
   - Dynamically activates 6-DoF equality weld constraints with tactile contact verification.

6. **Native ROS 2 Jazzy Software Bridge (`ros2_bridge/`):**
   - Telemetry publishing on `/robot/joint_states`, `/robot/camera/image_raw`, and `/robot/agent/status`.
   - Compatible with standard ROS 2 visualization tools (`rviz2`).

---

---

## 🛠️ Prerequisites & Setup

### 1. Install `uv` (Fast Python Package & Project Manager)
If you don't already have `uv` installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Install Dependencies
```bash
git clone https://github.com/maniradhakrishnan-dev/margaret.git
cd margaret
uv sync
```

### 3. Configure Environment Variables (Optional)
Margaret supports Google DeepMind Frontier reasoning models (`gemini-2.5-flash` and `gemini-2.0-flash`), while seamlessly falling back to local deterministic heuristic reasoning if no key is configured:
```bash
cp .env.example .env
# Open .env and add your GEMINI_API_KEY (optional)
```

### 4. Neural Vision Model Weights
Margaret employs **YOLO-World** for real-time open-vocabulary 3D scene perception. The neural model weights (`yolov8s-world.pt`) are **automatically downloaded** on first launch. If you wish to pre-fetch them offline:
```bash
uv run python -c "from ultralytics import YOLO; YOLO('yolov8s-world.pt')"
```

---

## 🚀 Quickstart Guide

### 1. Launch Margaret Autonomous Embodied AI Agent (Interactive 3D Viewer)
```bash
# Unitree G1 Full Humanoid (Default):
uv run margaret

# ALOHA Bimanual Desktop Platform:
uv run margaret --robot aloha
```

### 2. Interactive Live Disturbance Sandbox
Experience continuous 50Hz closed-loop visual servoing while disturbing objects in real-time:
```bash
uv run margaret --interactive
```
Interactive disturbance commands in the terminal:
- `nudge <object>`: Displaces an object by ~12cm across the tabletop
- `knock <object>`: Tips an object over with angular impulse
- `shake`: Perturbs all objects on the workbench simultaneously
- `drop <object>`: Knocks an object off the table onto the floor

### 3. Headless Verification Mode (Automated / CI)
Run autonomous physical verification without opening a graphical window:
```bash
uv run margaret --headless
```

### 5. Launch Standalone 3D Arena Viewer & Sandbox
Inspect the environment, furniture, cameras, and mouse-manipulate props:
```bash
uv run python arena/run_arena.py --robot unitree_g1
# Or explore procedural disturbances:
uv run python arena/explore_disturbed_arena.py --disturb displace
```

### 6. Run the Full Test Suite
Run all unit and integration tests:
```bash
uv run python -m unittest discover -s tests
```

---

## 📁 Repository Structure

```
.
├── agent/
│   ├── main.py              # Official Margaret CLI entry point (`uv run margaret`)
│   ├── embodied_agent.py    # Embodied AI physical agent lifecycle manager
│   ├── run_agentic_humanoid.py # Master end-to-end Physical AI orchestrator
│   ├── run_interactive_viewer.py # Interactive viewer runner
│   ├── brain.py             # Pure-Python ReAct cognitive loop & LLM integration
│   ├── memory.py            # 3D spatial scene memory & episodic step history
│   ├── tools.py             # Tool registry with automatic JSON schema generation
│   └── world_model.py       # Pre-simulation imagination & risk assessment
├── arena/
│   ├── environment_generator.py # Procedural environment and scene generator
│   ├── disturbance_engine.py    # Physical disturbance engine (nudges, tremors)
│   ├── run_arena.py             # Standalone 3D interactive arena viewer
│   ├── explore_disturbed_arena.py # Procedural disturbance sandbox
│   └── arena.xml                # Custom 3D laboratory arena and workbench
├── models/
│   ├── humanoid/            # Standalone research humanoid model
│   └── menagerie/           # MuJoCo Menagerie models (Unitree G1, Go2, ALOHA)
│       └── unitree_g1/
│           ├── scene_workbench.xml # G1 arena integrated with workbench & props
│           └── g1_with_hands.xml   # 29-DoF G1 model with dexterous hands
├── robots/
│   ├── base_robot.py        # Abstract BaseRobot interface
│   ├── unitree_g1.py        # Unitree G1 Menagerie adapter
│   ├── default_humanoid.py  # Default 27-DoF humanoid adapter
│   ├── robot_factory.py     # Robot instantiation factory
│   └── explore_humanoid.py  # Humanoid kinematics and motor inspector
├── ros2_bridge/
│   └── humanoid_node.py     # ROS 2 Jazzy bridge node (JointState, Image, Status)
├── skills/
│   ├── kinematics.py        # Jacobian Damped Least Squares IK solver
│   ├── reach_controller.py  # Smooth thread-safe PD reach controller
│   ├── grasp.py             # Multi-finger grasp & equality weld lift controller
│   └── vision.py            # Virtual head camera RGB-D perception pipeline
└── tests/
    └── test_agentic_humanoid.py # Comprehensive unit and integration test suite

---

## 📜 Native ROS 2 Jazzy Reference
For configuring native ROS 2 Jazzy on Ubuntu 24.04 LTS:
```bash
sudo apt install -y ros-jazzy-desktop
source /opt/ros/jazzy/setup.bash
```
Verified on native DDS communication.

















