"""
Procedural Environment Generator for Margaret Autonomous Robot Arena.
Generates complex, randomized, and independent 3D physical environments:
- Arbitrary furniture (tables, workbenches, shelves)
- Arbitrary obstacles (dividers, barriers, clutter)
- Arbitrary objects with randomized shapes, colors, masses, and friction
- Pluggable robot embodiments (Unitree G1, ALOHA)

Completely independent of Margaret agent.
"""

import os
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Color Palette (RGB definitions for MuJoCo materials)
COLOR_PALETTE = {
    "red": [0.90, 0.15, 0.15, 1.0],
    "blue": [0.15, 0.30, 0.90, 1.0],
    "green": [0.15, 0.85, 0.30, 1.0],
    "yellow": [0.92, 0.85, 0.12, 1.0],
    "purple": [0.70, 0.20, 0.82, 1.0],
    "orange": [0.95, 0.50, 0.10, 1.0],
    "cyan": [0.10, 0.85, 0.90, 1.0],
    "white": [0.92, 0.92, 0.92, 1.0],
    "black": [0.12, 0.12, 0.12, 1.0],
    "gray": [0.55, 0.55, 0.55, 1.0],
}

SHAPE_TYPES = ["box", "cylinder", "sphere", "capsule"]


class ProceduralEnvironment:
    """
    Independent Environment Generator.
    Assembles physics arenas with arbitrary objects, obstacles, and furniture,
    then drops the robot inside for benchmarking.
    """

    def __init__(self, robot_platform: str = "unitree_g1", seed: Optional[int] = None):
        self.robot_platform = robot_platform
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def generate_random_arena(
        self,
        num_objects: int = 3,
        furniture_type: str = "workbench",
        obstacle_type: str = "none",
        add_obstacle_barrier: bool = False,
        table_height: float = 0.60,
        table_x: float = 0.35,
        table_y_extent: float = 0.36,
        output_filename: str = "procedural_arena.xml",
        **kwargs,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generates a randomized MuJoCo MJCF XML arena independent of the agent.
        Defaults to a clean, plain table and objects with zero barriers or upper shelves.
        Returns (xml_filepath, list_of_spawned_objects).
        """
        # Determine target directory based on robot embodiment
        if self.robot_platform == "unitree_g1":
            target_dir = os.path.abspath("models/menagerie/unitree_g1")
            include_line = '<include file="g1_with_hands.xml"/>'
            head_cam_line = '<camera name="head_cam" pos="0.06 0 1.25" euler="0 25 0" fovy="60"/>'
            pelvis_ground = '<weld name="ground_pelvis" body1="pelvis"/>'
            right_hand_link = "right_hand_middle_0_link"
            left_hand_link = "left_hand_middle_0_link"
            floor_pos = "0 0 0"
        elif self.robot_platform == "aloha":
            target_dir = os.path.abspath("models/menagerie/aloha")
            include_line = '<include file="aloha.xml"/>'
            head_cam_line = '<camera name="overhead_cam" pos="0 -0.30 1.02" mode="fixed" quat="0.976332 0.216277 0 0" fovy="58"/>\n    <camera name="head_cam" pos="0 -0.30 1.02" mode="fixed" quat="0.976332 0.216277 0 0" fovy="58"/>'
            pelvis_ground = ''
            right_hand_link = "right/gripper_link"
            left_hand_link = "left/gripper_link"
            table_height = 0.0
            table_x = 0.0
        elif self.robot_platform in ["limx_tron1", "tron1", "wheeled_biped"]:
            target_dir = os.path.abspath("models/menagerie/limx_tron1")
            include_line = '<include file="tron1.xml"/>'
            head_cam_line = ''
            pelvis_ground = '<weld name="ground_base" body1="base_Link"/>'
            right_hand_link = "wheel_R_Link"
            left_hand_link = "wheel_L_Link"
            table_height = 0.50
            table_x = 0.42
            floor_pos = "0 0 0"
        else:
            raise ValueError(f"Unknown robot platform: {self.robot_platform}")

        os.makedirs(target_dir, exist_ok=True)
        xml_path = os.path.join(target_dir, output_filename)

        # 1. Assets: generate material declarations
        materials_xml = [
            '<material name="wood_mat" rgba="0.6 0.4 0.25 1" reflectance="0.1" roughness="0.6"/>',
            '<material name="barrier_mat" rgba="0.3 0.3 0.35 1" reflectance="0.3"/>',
        ]
        for cname, rgba in COLOR_PALETTE.items():
            rgba_str = f"{rgba[0]:.2f} {rgba[1]:.2f} {rgba[2]:.2f} {rgba[3]:.1f}"
            materials_xml.append(f'<material name="{cname}_mat" rgba="{rgba_str}" specular="0.6" shininess="0.5"/>')

        # 2. Furniture: plain table (or optional shelves / dual tables if requested)
        furniture_type = kwargs.get("furniture_type", furniture_type)
        if self.robot_platform == "aloha":
            furniture_xml = """
    <!-- ALOHA Central Desktop Workbench (Spacious tabletop at Z = 0.0m) -->
    <body name="workbench" pos="0 0 0">
      <geom name="tabletop" type="box" size="0.65 0.45 0.375" pos="0 0 -0.375" material="wood_mat"/>
    </body>"""
        elif furniture_type == "shelf_table":
            furniture_xml = f"""
    <!-- Independent Table with Upper Shelf Riser -->
    <body name="workbench" pos="{table_x:.2f} 0 0">
      <geom name="tabletop" type="box" size="0.18 {table_y_extent:.2f} 0.02" pos="0 0 {table_height:.2f}" material="wood_mat"/>
      <geom name="leg1" type="cylinder" size="0.02 {table_height/2:.2f}" pos="0.13 {table_y_extent-0.04:.2f} {table_height/2:.2f}" rgba="0.2 0.2 0.2 1"/>
      <geom name="leg2" type="cylinder" size="0.02 {table_height/2:.2f}" pos="-0.13 {table_y_extent-0.04:.2f} {table_height/2:.2f}" rgba="0.2 0.2 0.2 1"/>
      <geom name="leg3" type="cylinder" size="0.02 {table_height/2:.2f}" pos="0.13 {-table_y_extent+0.04:.2f} {table_height/2:.2f}" rgba="0.2 0.2 0.2 1"/>
      <geom name="leg4" type="cylinder" size="0.02 {table_height/2:.2f}" pos="-0.13 {-table_y_extent+0.04:.2f} {table_height/2:.2f}" rgba="0.2 0.2 0.2 1"/>
      <!-- Upper Shelf -->
      <geom name="shelf_top" type="box" size="0.08 {table_y_extent*0.75:.2f} 0.015" pos="0.06 0 {table_height + 0.18:.2f}" material="wood_mat"/>
      <geom name="shelf_post_left" type="cylinder" size="0.01 0.09" pos="0.06 {table_y_extent*0.65:.2f} {table_height + 0.09:.2f}" rgba="0.3 0.3 0.3 1"/>
      <geom name="shelf_post_right" type="cylinder" size="0.01 0.09" pos="0.06 {-table_y_extent*0.65:.2f} {table_height + 0.09:.2f}" rgba="0.3 0.3 0.3 1"/>
    </body>"""
        elif furniture_type == "dual_table":
            furniture_xml = f"""
    <!-- Independent Primary Workbench & Side Cart -->
    <body name="workbench" pos="{table_x:.2f} 0 0">
      <geom name="tabletop" type="box" size="0.18 {table_y_extent:.2f} 0.02" pos="0 0 {table_height:.2f}" material="wood_mat"/>
      <geom name="leg1" type="cylinder" size="0.02 {table_height/2:.2f}" pos="0.13 {table_y_extent-0.04:.2f} {table_height/2:.2f}" rgba="0.2 0.2 0.2 1"/>
      <geom name="leg2" type="cylinder" size="0.02 {table_height/2:.2f}" pos="-0.13 {table_y_extent-0.04:.2f} {table_height/2:.2f}" rgba="0.2 0.2 0.2 1"/>
      <geom name="leg3" type="cylinder" size="0.02 {table_height/2:.2f}" pos="0.13 {-table_y_extent+0.04:.2f} {table_height/2:.2f}" rgba="0.2 0.2 0.2 1"/>
      <geom name="leg4" type="cylinder" size="0.02 {table_height/2:.2f}" pos="-0.13 {-table_y_extent+0.04:.2f} {table_height/2:.2f}" rgba="0.2 0.2 0.2 1"/>
    </body>
    <body name="side_table" pos="{table_x + 0.05:.2f} {table_y_extent + 0.32:.2f} 0">
      <geom name="side_tabletop" type="box" size="0.14 0.18 0.02" pos="0 0 {table_height - 0.05:.2f}" rgba="0.35 0.35 0.38 1"/>
      <geom name="side_leg1" type="cylinder" size="0.015 {(table_height - 0.05)/2:.2f}" pos="0.09 0.13 {(table_height - 0.05)/2:.2f}" rgba="0.2 0.2 0.2 1"/>
      <geom name="side_leg2" type="cylinder" size="0.015 {(table_height - 0.05)/2:.2f}" pos="-0.09 -0.13 {(table_height - 0.05)/2:.2f}" rgba="0.2 0.2 0.2 1"/>
    </body>"""
        else:
            furniture_xml = f"""
    <!-- Plain Wooden Table / Workbench -->
    <body name="workbench" pos="{table_x:.2f} 0 0">
      <geom name="tabletop" type="box" size="0.18 {table_y_extent:.2f} 0.02" pos="0 0 {table_height:.2f}" material="wood_mat"/>
      <geom name="leg1" type="cylinder" size="0.02 {table_height/2:.2f}" pos="0.13 {table_y_extent-0.04:.2f} {table_height/2:.2f}" rgba="0.2 0.2 0.2 1"/>
      <geom name="leg2" type="cylinder" size="0.02 {table_height/2:.2f}" pos="-0.13 {table_y_extent-0.04:.2f} {table_height/2:.2f}" rgba="0.2 0.2 0.2 1"/>
      <geom name="leg3" type="cylinder" size="0.02 {table_height/2:.2f}" pos="0.13 {-table_y_extent+0.04:.2f} {table_height/2:.2f}" rgba="0.2 0.2 0.2 1"/>
      <geom name="leg4" type="cylinder" size="0.02 {table_height/2:.2f}" pos="-0.13 {-table_y_extent+0.04:.2f} {table_height/2:.2f}" rgba="0.2 0.2 0.2 1"/>
    </body>"""

        # 3. Optional Obstacle Barrier
        obstacle_xml = ""
        obstacle_type = kwargs.get("obstacle_type", obstacle_type)
        if obstacle_type == "barrier" or add_obstacle_barrier:
            barrier_z = table_height + 0.06
            obstacle_xml = f"""
    <!-- Physical Obstacle Barrier -->
    <body name="obstacle_barrier" pos="{table_x:.2f} 0.0 {barrier_z:.2f}">
      <geom name="barrier_geom" type="box" size="0.015 0.06 0.04" material="barrier_mat" mass="5.0"/>
    </body>"""
        elif obstacle_type == "divider":
            barrier_z = table_height + 0.05
            obstacle_xml = f"""
    <!-- Tabletop Divider Partition -->
    <body name="obstacle_divider" pos="{table_x:.2f} -0.05 {barrier_z:.2f}">
      <geom name="divider_geom" type="box" size="0.12 0.01 0.03" material="barrier_mat" mass="3.0"/>
    </body>"""

        # 4. Generate Random Props
        objects_xml = []
        spawned_meta = []
        if self.robot_platform == "aloha":
            surface_z = 0.00
            # Props positioned in front workspace reachable by ViperX manipulators
            x_slots = [-0.18, -0.09, 0.09, 0.18]
            y_slots = np.linspace(0.10, 0.26, max(num_objects, 4)).tolist()
        else:
            surface_z = table_height + 0.02
            x_slots = [round(table_x - 0.04, 3), round(table_x, 3), round(table_x + 0.04, 3)]
            y_slots = np.linspace(-table_y_extent + 0.11, table_y_extent - 0.11, max(num_objects, 4)).tolist()
        self.rng.shuffle(y_slots)

        available_colors = list(COLOR_PALETTE.keys())
        self.rng.shuffle(available_colors)

        for i in range(num_objects):
            color = available_colors[i % len(available_colors)]
            shape = self.rng.choice(SHAPE_TYPES)
            obj_name = f"prop_{color}_{shape}_{i+1}"

            px = round(float(self.rng.choice(x_slots)), 3)
            py = round(float(y_slots[i % len(y_slots)]), 3)
            mass = round(self.rng.uniform(0.08, 0.25), 2)
            friction = "1.8 0.05 0.005"

            if self.robot_platform == "aloha":
                if shape == "box":
                    geom_def = f'<geom name="{obj_name}_geom" type="box" size="0.016 0.016 0.016" material="{color}_mat" mass="{mass}" friction="{friction}"/>'
                    pz = round(surface_z + 0.018, 3)
                elif shape == "cylinder":
                    geom_def = f'<geom name="{obj_name}_geom" type="cylinder" size="0.015 0.025" material="{color}_mat" mass="{mass}" friction="{friction}"/>'
                    pz = round(surface_z + 0.027, 3)
                elif shape == "sphere":
                    geom_def = f'<geom name="{obj_name}_geom" type="sphere" size="0.016" material="{color}_mat" mass="{mass}" friction="{friction}"/>'
                    pz = round(surface_z + 0.018, 3)
                elif shape == "capsule":
                    geom_def = f'<geom name="{obj_name}_geom" type="capsule" size="0.014 0.022" material="{color}_mat" mass="{mass}" friction="{friction}"/>'
                    pz = round(surface_z + 0.018, 3)
            else:
                if shape == "box":
                    geom_def = f'<geom name="{obj_name}_geom" type="box" size="0.032 0.032 0.032" material="{color}_mat" mass="{mass}" friction="{friction}"/>'
                    pz = round(surface_z + 0.034, 3)
                elif shape == "cylinder":
                    geom_def = f'<geom name="{obj_name}_geom" type="cylinder" size="0.032 0.045" material="{color}_mat" mass="{mass}" friction="{friction}"/>'
                    pz = round(surface_z + 0.046, 3)
                elif shape == "sphere":
                    geom_def = f'<geom name="{obj_name}_geom" type="sphere" size="0.035" material="{color}_mat" mass="{mass}" friction="{friction}"/>'
                    pz = round(surface_z + 0.036, 3)
                elif shape == "capsule":
                    geom_def = f'<geom name="{obj_name}_geom" type="capsule" size="0.028 0.035" material="{color}_mat" mass="{mass}" friction="{friction}"/>'
                    pz = round(surface_z + 0.036, 3)

            body_str = f"""
    <body name="{obj_name}" pos="{px} {py} {pz}">
      <joint name="{obj_name}_joint" type="free" damping="0.8"/>
      {geom_def}
    </body>"""
            objects_xml.append(body_str)
            spawned_meta.append({
                "name": obj_name,
                "color": color,
                "shape": shape,
                "initial_pos": [px, py, pz],
                "mass": mass,
            })

        # 5. Generic Equality Welds (Retargeted dynamically by GraspController)
        welds_xml = f"""
  <equality>
    {pelvis_ground}
    <weld name="weld_right_hand" body1="{right_hand_link}" body2="world" active="false"/>
    <weld name="weld_left_hand"  body1="{left_hand_link}"  body2="world" active="false"/>
  </equality>"""

        # 6. Assemble complete MJCF XML
        full_xml = f"""<mujoco model="independent_procedural_arena">
  {include_line}

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.2 0.2 0.2" specular="0.8 0.8 0.8"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    {chr(10).join(materials_xml)}
  </asset>

  <worldbody>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" pos="{floor_pos}"/>
    <light directional="true" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" pos="0 0 4" dir="0 0 -1"/>
    <camera name="room_overview" pos="1.5 -1.5 1.5" euler="55 45 0" fovy="55"/>
    {head_cam_line}
    {furniture_xml}
    {obstacle_xml}
    {''.join(objects_xml)}
  </worldbody>

  {welds_xml}
</mujoco>
"""

        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(full_xml)

        return xml_path, spawned_meta
