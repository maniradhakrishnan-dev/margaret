"""
Robot Factory
Dynamically selects and instantiates robot form factors.
"""

from typing import Dict, Type
from robots.base_robot import BaseRobot
from robots.unitree_g1 import UnitreeG1Robot
from robots.default_humanoid import DefaultHumanoidRobot
from robots.aloha import AlohaRobot
from robots.limx_tron1 import Tron1Robot

_ROBOT_REGISTRY: Dict[str, Type[BaseRobot]] = {
    "g1": UnitreeG1Robot,
    "unitree_g1": UnitreeG1Robot,
    "default": DefaultHumanoidRobot,
    "humanoid": DefaultHumanoidRobot,
    "aloha": AlohaRobot,
    "aloha_bimanual": AlohaRobot,
    "desktop_arm": AlohaRobot,
    "limx_tron1": Tron1Robot,
    "tron1": Tron1Robot,
    "wheeled_biped": Tron1Robot,
}

def get_robot(robot_type: str = "unitree_g1") -> BaseRobot:
    """Returns an instance of the requested robot adapter."""
    key = robot_type.lower().strip()
    if key not in _ROBOT_REGISTRY:
        raise ValueError(
            f"Unknown robot type: '{robot_type}'. Available options: {list(_ROBOT_REGISTRY.keys())}"
        )
    return _ROBOT_REGISTRY[key]()
