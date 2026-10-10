# Margaret: Full-Stack Autonomous Embodied Physical AI Architecture

```
                                  +-------------------------------------------------------------+
                                  |                 TOP: HARNESS, SKILLS & ENVIRONMENT          |
                                  |  - Skill Registry: reach_hand, grasp, lift, place, recover  |
                                  |  - Spatial & Episodic Memory (3D Scene Graph, Bounding Box) |
                                  |  - MuJoCo Digital Twin Arena & Ground-Truth Workspace       |
                                  |  - Strict Pydantic Schema Contracts & Function Definitions  |
                                  |  - Skill Invariants & Postcondition Verification            |
                                  +-------------------------------------------------------------+
                                                                 |
                                                                 | [Available Skills, Schemas,
                                                                 |  Affordances, Spatial Memory]
                                                                 v
+---------------------------------------+      +-----------------------------------+      +---------------------------------------+
|          LEFT: INPUT STACK            |      |      CENTER: EMBODIED AGENT       |      |         RIGHT: PRODUCED ACTION        |
|                                       |      |             MARGARET              |      |              (OUTPUT)                 |
| 1. Multimodal Sensory Inputs          | ---> |      MULTI-AGENT LIVE GRAPH       | ---> | - Structured Action Dispatch           |
|    - Natural Language User Goal       |      |                                   |      |   (Pydantic ActionDecision schema)    |
|    - Continuous 30 Hz Camera Stream   |      | 1. Perception Grounding Node      |      | - Bimanual Arm Selection              |
|      (RGB-D via Local Small Model)    |      |    (Translates raw pixels/rays    |      |   ('right' vs 'left' workspace)       |
|    - Metric Depth Map & Point Clouds  |      |                                   |      | - 3D Cartesian End-Effector Waypoint  |
|    - Proprioception & Joint Telemetry |      | 2. Task Decomposition Node        |      |   [target_x, target_y, target_z]     |
|      (Joint Positions, Velocities,    |      |    (Builds dynamic sub-goal DAG)  |      | - Dexterous Grasp & Normal Force (N)  |
|       Torques, Tactile Grip Forces)   |      |                                   |      | - Stacking Clearance & Drop Recovery  |
|                                       |      | 3. Embodied Reasoner Node (ReAct) |      | - Motion Timing & Interpolation Steps |
| 2. Hierarchical Model Stack           |      |    (Allocates arms, plans contact)|      +---------------------------------------+
|    - Frontier Model: Gemini ER 2.0    |      |                                   |                          |
|      (High-level cortical reasoning)  |      | 4. Sandbox World Model Verifier   |                          | [Dispatches Motion
|    - Small Model: Flash-Lite / SLM    |      |    (Forward rollout in MjData     |                          |  to Low-Level Stack]
|      (Fast reactive fallback & checks)|      |     sandbox clone; zero mocks)    |                          v
|    - Edge Vision Model: YOLO-World    |      +-----------------------------------+                          v
|      (Local 30 Hz zero-shot detector) |                        ^                        +---------------------------------------+
+---------------------------------------+                        |                        |          MOTION CONTROLLERS           |
                    ^                                            |                        |  - 7-DoF Jacobian DLS IK Solver       |
                    |                                            | [Proprioceptive State, |  - Actuator PD / Torque Controllers   |
                    |                                            |  Kinematic Limits,     |  - Whole-Body Trajectory Interpolator |
                    |                                            |  Tactile Normal Forces,|  - Balance Guard & CoM Safety Filter  |
                    |                                            |  Hardware Alarms]      +---------------------------------------+
                    |                                            |                                    |
                    |                                            |                                    v
                    |             +-------------------------------------------------------------------+-------+
                    |             |          BOTTOM: CONTROL, RL, LOW-LEVEL HARDWARE & ROBOT DESCRIPTION      |
                    |             |                                                                           |
                    |             |  1. ROBOT PLATFORMS (MODULAR DUAL-TARGET):                                |
                    |             |     - Platform A: Desktop Bimanual Arm Rig (Fixed Base Manipulation)      |
                    |             |     - Platform B: Full Humanoid (Unitree G1 29-DoF Whole-Body Mobile)     |
                    |             |  2. CONTROL & RL:                                                         |
                    |             |     - Numerical Kinematics: 7-DoF Damped Least Squares (DLS) IK           |
                    |             |     - RL Locomotion Policy (PPO): Dynamic balance, walking, jumping        |
                    |             |  3. SIMULATION & HARDWARE:                                                |
                    |             |     - MuJoCo Multi-Body Contact Dynamics (dt = 0.002s)                    |
                    |             |     - Low-level Motor Actuation & CAN Bus Hardware Abstraction            |
                    |             |     - Middleware: ROS 2 Jazzy Bridge (/joint_states, /cmd_vel, /head_cam) |
                    |             +---------------------------------------------------------------------------+
                    |                                            |
                    +--------------------------------------------+
                             CLOSED-LOOP PHYSICAL FEEDBACK LOOP
                             - Actual Reached Position [x_act, y_act, z_act]
                             - Kinematic Tracking Error (cm)
                             - Tactile Contact & Measured Grip Force (N)
                             - Collision Contacts, Prop Slipping, Floor Drops
```

---

## 1. Directional Geometry Around the Central Agent

| Direction | Component | Role | What Flows In / Out |
| :--- | :--- | :--- | :--- |
| **LEFT $\to$ CENTER** | **Input Stack** | *What the agent senses & thinks with* | Natural language goals, continuous 30 Hz local vision, proprioception telemetry, Frontier LLM, Small SLM, and Edge Vision Model. |
| **TOP $\to$ CENTER** | **Harness & Skills** | *What tools the agent has at its disposal* | Callable skills (`reach`, `grasp`, `place`, `recover`), 3D spatial scene graph memory, Pydantic function schemas, and formal skill pre/postcondition invariants. |
| **BOTTOM $\to$ CENTER** | **Physical Embodiment** | *The physical body grounding the agent* | **Feeds UPWARD**: Proprioception (joint angles $q$, velocities $\dot{q}$, torques $\tau$), robot URDF kinematic reach limits, Center-of-Mass (CoM) stability, tactile normal forces ($F_N$), and hardware collision alarms. |
| **CENTER $\to$ RIGHT** | **Produced Action** | *The output decisions made by the agent* | **Feeds OUTWARD**: High-level action dispatches (Pydantic tool calls), bimanual arm allocation (`right` vs `left`), 3D Cartesian waypoints $(X, Y, Z)$, target grip forces, and velocity profiles. |
| **RIGHT $\to$ BOTTOM $\to$ LEFT** | **Closed-Loop Feedback** | *Physical feedback engine* | Actions actuate controllers & physics; real empirical outcomes (tracking error, contact forces, slip, drops) feed back into the input stream and agent ReAct loop. |

---

## 2. Core Architectural Pillars

### Pillar 1: Dual-Robot Platform Strategy
To decouple dexterous manipulation research from whole-body humanoid balance:

```
+----------------------------------------------------+----------------------------------------------------+
| PLATFORM A: DESKTOP BIMANUAL ARM RIG (ALOHA-style) | PLATFORM B: FULL 29-DoF HUMANOID (Unitree G1)     |
+----------------------------------------------------+----------------------------------------------------+
| • Morphology: Two 6-DoF/7-DoF arms on rigid bench  | • Morphology: 29 active DoFs (legs, waist, arms)   |
| • Kinematic Base: Bolted to tabletop (zero tipping)| • Kinematic Base: Floating base on two feet        |
| • Focus: Precision dexterous manipulation, picking,| • Focus: Whole-body mobile manipulation, walking,  |
|   stacking, tool assembly, handovers, compliance   |   jumping, crouching, dynamic balance recovery     |
| • Sim-to-Real: Direct, fast, safe, inexpensive     | • Sim-to-Real: Requires RL locomotion balance      |
|   (e.g., low-cost SO-100 / ALOHA arms)             |   policy and Whole-Body Control (WBC) safety filter|
+----------------------------------------------------+----------------------------------------------------+
```
*Both platforms share the exact same Agent Brain, Tool Registry, Vision Pipeline, and Spatial Memory.*

---

### Pillar 2: Asynchronous Fast-Slow Vision Pipeline (Real Small Model)
Solves the frequency gap (30 Hz Camera vs 0.5 Hz LLM) with zero cloud token waste:

1. **Fast Stream (Local Edge Vision @ 30 Hz)**:
   - Runs a local lightweight open-vocabulary detector (**YOLO-World** or **SmolVLM**) on the robot's local GPU/CPU.
   - Detects 2D bounding boxes from real camera RGB pixels (e.g. `"red mug"`, `"blue block"`).
   - Deprojects 2D bounding boxes into true 3D spatial $(X, Y, Z)$ centroids using the **metric depth map** and camera intrinsics matrix:
     $$\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = Z \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$
   - **Zero ground-truth physics cheating**: coordinates are derived purely from camera vision and depth.
   - Continuously updates the local 3D Scene Graph in memory.
2. **Slow Stream (Frontier Agent Brain @ 0.5–1 Hz)**:
   - Gemini receives only the compressed text 3D Scene Graph ($<40$ tokens per step).
   - Video streaming costs **0 cloud tokens**.
   - Gemini only requests a static keyframe image snapshot when diagnosing an unexpected physical failure.

---

### Pillar 3: Genuine Sandbox World Model (Zero Mocks)
Replaces heuristic distance/Z checks with an isolated forward-simulation sandbox:

1. **State Clonation**: Clones the current physical state in memory:
   ```python
   sandbox_data = copy.deepcopy(data)  # Isolated digital twin
   ```
2. **Forward Rollout in Imagination**:
   - Solves IK for candidate action and steps MuJoCo forward for $K=30$ timesteps in the sandbox clone.
3. **Empirical Physical Inspection**:
   - **Collision Contacts**: Queries `sandbox_data.contact` for forbidden pairs (e.g., arm colliding with table surface or torso).
   - **Actuator Torque Saturation**: Checks if `sandbox_data.qfrc_actuator` exceeds motor limits.
   - **Slip & Tip Detection**: Checks if the target object tilts ($>15^\circ$) or drops below table height ($Z < 0.50\,\text{m}$).
4. **Action Gating**:
   - If safe: returns `EXECUTE` with empirical clearance margins.
   - If unsafe: aborts before the real robot moves and returns an exact physical diagnosis to the agent:
     `{"recommendation": "REPLAN", "error": "Right forearm penetrates table surface by 3.2cm at step 14"}`.

---

### Pillar 4: Low-Level Control & RL Dynamics
Clarifies the exact control stack between high-level agent decisions and low-level actuators:

1. **Upper-Body Bimanual Manipulation**:
   - **Numerical Kinematics**: 7-DoF Damped Least Squares (DLS) Jacobian IK:
     $$\Delta \theta = J^T (J J^T + \lambda^2 I)^{-1} \Delta x$$
   - **Trajectory Generation**: Smooth cubic spline / multi-step interpolation (no step-response torque spikes).
   - **Actuation**: Actuator PD position servos running at $500\,\text{Hz}$ in MuJoCo.
2. **Lower-Body Humanoid Locomotion & Balance (RL)**:
   - **Current State**: Humanoid pelvis anchored in keyframe stand for upper-body manipulation isolation.
   - **RL Roadmap (PPO Policy)**:
     - Trained in MuJoCo / Isaac Gym to output leg motor torques ($12$ leg DoFs) at $50\,\text{Hz}$.
     - Observes IMU orientation, angular velocity, and joint positions to maintain upright balance while arms move.
     - **Whole-Body Safety Guard**: Monitors Center-of-Mass (CoM) relative to the foot support polygon; halts arm trajectories if tipping is detected.

---

### Pillar 5: Formal Skill Contracts & Physical Invariants (No Fake Success)
Every skill must enforce strict preconditions and postconditions measured from real physics:

| Skill | Preconditions | Postconditions (Must Hold for `success=True`) | Physical Failure Diagnosis |
| :--- | :--- | :--- | :--- |
| **`reach_hand`** | Target within arm workspace ($Y \le 0.04$ for right) | $\|p_{\text{ee}} - p_{\text{target}}\| \le 0.05\,\text{m}$ | `tracking_error_exceeded` with exact delta |
| **`grasp_object`** | Hand within $5\,\text{cm}$ of object | Measured normal force $F_N \ge 10\,\text{N}$, finger closure | `grasp_slipped` or `insufficient_grip_force` |
| **`lift_hand`** | Object firmly held | Object elevation $\Delta Z \ge 0.12\,\text{m}$, object $Z > 0.74\,\text{m}$ | `object_dropped_during_lift` |
| **`place_object`** | Object firmly held | Object resting on dest surface, zero residual velocity, arm retracted | `dropped_on_floor` or `misaligned_stack` |
| **`recover_floor`**| Object on floor ($Z < 0.40\,\text{m}$) | Object restored to tabletop ($Z \ge 0.64\,\text{m}$), upright, zero velocity | `recovery_unreachable` |

*An action is NEVER marked successful based on LLM intent — it is only successful if the physical invariants pass in simulation/hardware.*

---

## 3. Technology Stack Reference

```
+----------------------------------------------------------------------------------------------------+
|                                      FULL-STACK TECHNOLOGY MATRIX                                  |
+----------------------+---------------------------------------+-------------------------------------+
| Layer                | Technology / Framework                | Role in Architecture                |
+----------------------+---------------------------------------+-------------------------------------+
| Frontier Reasoning   | Google Gemini Robotics ER 2.0         | Cortical planning, task DAG, ReAct  |
| Edge Reactive Fallback| Gemini 3.1 Flash-Lite / Local SLM    | Sub-second fallback & rate-limit recovery|
| Structured Schemas   | Pydantic v2 (Strict JSON Schema)      | Zero-hallucination action contracts |
| Local Vision Engine  | YOLO-World / ONNX + Metric Depth      | 30 Hz real pixel detection & 3D deprojection|
| World Model          | MuJoCo MjData Sandbox Cloner          | Forward simulation & collision gate |
| Physics Engine       | MuJoCo 3.x (Multi-Body Contact)       | Ground-truth physics simulation     |
| Numerical Kinematics | NumPy Damped Least Squares (DLS) IK   | 7-DoF bimanual arm IK solver        |
| Robot Platforms      | Unitree G1 (29-DoF) & Dual-Arm Rig    | Humanoid & Tabletop embodiments     |
| Middleware           | ROS 2 Jazzy Jalisco                   | Topic streaming (/joint_states, cam)|
| Benchmark Suite      | Python Unittest / Automated A/B Matrix| Empirical proof of harness superiority|
+----------------------+---------------------------------------+-------------------------------------+
```

---

## 4. Phased Execution Roadmap

To build this systematically without mocks or fake success:

- [ ] **Phase 1: Real Sandbox World Model**
  - Replace heuristic math in [`agent/world_model.py`](file:///home/mani_radhakrishnan/ros2_installation/agent/world_model.py) with isolated `mujoco.MjData` clone forward rollouts.
  - Return genuine physical collision contacts and slip predictions.
- [ ] **Phase 2: Real Local Vision Model & Depth Deprojection**
  - Implement real pixel-based detection in [`skills/vision.py`](file:///home/mani_radhakrishnan/ros2_installation/skills/vision.py) using an edge open-vocabulary detector + depth deprojection matrix $K^{-1}$ (eliminating ground-truth physics reading).
- [ ] **Phase 3: Desktop Bimanual Dual-Arm Platform**
  - Add [`robots/bimanual_desktop_arm.py`](file:///home/mani_radhakrishnan/ros2_installation/robots/bimanual_desktop_arm.py) and MuJoCo scene for rigid-base manipulation.
- [ ] **Phase 4: Formal Invariants & Empirical Benchmark Harness**
  - Add pre/postcondition verifiers to all skills in [`skills/`](file:///home/mani_radhakrishnan/ros2_installation/skills/).
  - Create `benchmark_harness.py` to mathematically prove that our multi-agent harness outperforms single-shot open-loop baselines.
- [ ] **Phase 5: RL Balance & Sim-to-Real Hardware Deployment**
  - Implement balance safety guards and ROS 2 hardware interfaces for physical deployment.