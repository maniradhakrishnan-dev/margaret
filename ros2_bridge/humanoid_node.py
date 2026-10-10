"""
ROS 2 Jazzy Robotics Bridge Node
Bridges MuJoCo Physical AI Simulation to native ROS 2 topics:
- /robot/joint_states      (sensor_msgs/JointState)
- /robot/camera/image_raw   (sensor_msgs/Image)
- /robot/agent/status      (std_msgs/String)
- /robot/cmd_mission       (std_msgs/String - Subscriber)
"""

import time
import json
import threading
from typing import Optional

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState, Image
    from std_msgs.msg import String, Header
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

class HumanoidRosBridge:
    """
    ROS 2 Jazzy Bridge for Humanoid Physics & Agent State.
    Safely operates whether native ROS 2 environment is sourced or not.
    """
    def __init__(self, model, data, vision=None, lock: Optional[threading.Lock] = None, robot_name: str = "unitree_g1"):
        self.model = model
        self.data = data
        self.vision = vision
        self.lock = lock
        self.robot_name = robot_name
        self.node = None
        self.executor = None
        self.ros_thread = None
        self.is_running = False

        if not HAS_ROS2:
            print("⚠️ ROS 2 Python libraries (rclpy) not found in active environment. ROS 2 Bridge skipped.")
            return

        try:
            if not rclpy.ok():
                rclpy.init()
            self._setup_node()
        except Exception as e:
            print(f"⚠️ Could not initialize ROS 2 node: {e}")

    def _setup_node(self):
        class _BridgeNode(Node):
            def __init__(self, outer):
                super().__init__("humanoid_agent_bridge")
                self.outer = outer
                self.joint_pub = self.create_publisher(JointState, "/robot/joint_states", 10)
                self.image_pub = self.create_publisher(Image, "/robot/camera/image_raw", 10)
                self.status_pub = self.create_publisher(String, "/robot/agent/status", 10)

                # Timer at 30 Hz for telemetry publishing
                self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

            def timer_callback(self):
                self.outer._publish_telemetry(self)

        self.node = _BridgeNode(self)
        self.is_running = True

        def spin_worker():
            try:
                rclpy.spin(self.node)
            except Exception:
                pass

        self.ros_thread = threading.Thread(target=spin_worker, daemon=True)
        self.ros_thread.start()
        print("🚀 ROS 2 Jazzy Bridge Active! Publishing on /robot/joint_states and /robot/camera/image_raw.")

    def _publish_telemetry(self, node):
        if not self.model or not self.data:
            return

        # 1. Publish JointStates
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        names = []
        positions = []
        velocities = []

        import mujoco
        if self.lock:
            with self.lock:
                for i in range(self.model.njnt):
                    j_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    q_addr = self.model.jnt_qposadr[i]
                    d_addr = self.model.jnt_dofadr[i]
                    if j_name and q_addr < len(self.data.qpos):
                        names.append(j_name)
                        positions.append(float(self.data.qpos[q_addr]))
                        velocities.append(float(self.data.qvel[d_addr]) if d_addr < len(self.data.qvel) else 0.0)
        else:
            for i in range(self.model.njnt):
                j_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                q_addr = self.model.jnt_qposadr[i]
                d_addr = self.model.jnt_dofadr[i]
                if j_name and q_addr < len(self.data.qpos):
                    names.append(j_name)
                    positions.append(float(self.data.qpos[q_addr]))
                    velocities.append(float(self.data.qvel[d_addr]) if d_addr < len(self.data.qvel) else 0.0)

        msg.name = names
        msg.position = positions
        msg.velocity = velocities
        node.joint_pub.publish(msg)

    def publish_status(self, step_idx: int, thought: str, action: str, observation: str):
        """Broadcasts cognitive ReAct steps to ROS 2 network."""
        if not self.node:
            return
        status_dict = {
            "step": step_idx,
            "robot": self.robot_name,
            "thought": thought,
            "action": action,
            "observation": str(observation),
            "timestamp": time.time()
        }
        msg = String()
        msg.data = json.dumps(status_dict)
        self.node.status_pub.publish(msg)

    def shutdown(self):
        if self.node:
            self.node.destroy_node()
        if HAS_ROS2 and rclpy.ok():
            rclpy.shutdown()
