"""
Physical Environment Subsystem.
Location: arena/arena_manager.py

Dedicated, isolated physical world module:
- Independent procedural environment generation (plain workbench, zero obstruction sheets)
- Physics model and data initialization
- Runtime environment disturbance engine
- Completely decoupled from Agent AI, prompts, and reasoning graphs.
"""

import os
from typing import Optional, List, Dict, Tuple, Any
import mujoco
import numpy as np

from arena.environment_generator import ProceduralEnvironment
from arena.disturbance_engine import EnvironmentDisturber


class PhysicalEnvironment:
    """
    Independent Physical Simulation Arena.
    Manages the 3D physics scene, tabletop furniture, props, and disturbance engine.
    """

    def __init__(
        self,
        robot_platform: str = "unitree_g1",
        num_objects: int = 3,
        furniture_type: str = "workbench",
        obstacle_type: str = "none",
        seed: Optional[int] = None,
    ):
        self.robot_platform = robot_platform
        self.num_objects = num_objects
        self.furniture_type = furniture_type
        self.obstacle_type = obstacle_type
        self.seed = seed

        self.generator = ProceduralEnvironment(robot_platform=robot_platform, seed=seed)
        self.xml_path: str = ""
        self.spawned_props: List[Dict[str, Any]] = []
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.disturber: Optional[EnvironmentDisturber] = None

    def build(self, output_filename: str = "runtime_interactive_arena.xml") -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generates and builds the independent physical arena.
        Guarantees a clean, plain wooden workbench with zero upper sheets or obstacle barriers.
        """
        self.xml_path, self.spawned_props = self.generator.generate_random_arena(
            num_objects=self.num_objects,
            furniture_type=self.furniture_type,
            obstacle_type=self.obstacle_type,
            add_obstacle_barrier=(self.obstacle_type == "barrier"),
            output_filename=output_filename,
        )

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.disturber = EnvironmentDisturber(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        return self.xml_path, self.spawned_props

    def get_prop_positions(self) -> Dict[str, List[float]]:
        """Returns live 3D coordinates for all props in the arena."""
        positions = {}
        for p in self.spawned_props:
            name = p["name"]
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                positions[name] = self.data.xpos[bid].round(3).tolist()
            else:
                positions[name] = p["initial_pos"]
        return positions
