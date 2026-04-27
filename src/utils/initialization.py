import logging
import os
from typing import Dict, Optional

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from tasks.task_configs import TASK_CONFIG
from utils.config_loader import load_config


async def initialize_robot_controller(
    task: str,
    logger: logging.Logger,
    controller_config_override: Optional[str] = None,
) -> Dict:
    """Initialize robot controller configurations.

    Args:
        task: Task name registered in TASK_CONFIG.
        logger: Logger instance.
        controller_config_override: Optional path to a controller config YAML
            that overrides the default TASK_CONFIG entry. Used by eval sweeps.
    """
    if task not in TASK_CONFIG:
        logger.error(f"Unknown task: {task}")
        raise ValueError(f"Unknown task: {task}")

    task_configs = TASK_CONFIG[task]
    controller_cfg_path = controller_config_override or task_configs["controller"]
    logger.info(
        "Configurations loaded successfully (controller: %s)", controller_cfg_path
    )

    return {
        "controller_config": load_config(controller_cfg_path),
        "robot_interface_config": load_config(task_configs["robot_interface"]),
    }


async def initialize_channel(task: str, robot_interface_config: Dict, logger: logging.Logger):
    """Initialize communication channel based on task type."""
    if "sim" in task or "eval" in task:  # eval tasks always run in sim
        logger.info(
            f"Initializing channel with Domain ID: {robot_interface_config['DOMAIN_ID']} "
            f"and network interface: {robot_interface_config['INTERFACE']}"
        )
        ChannelFactoryInitialize(robot_interface_config["DOMAIN_ID"], robot_interface_config["INTERFACE"])
    else:
        network_interface = os.environ.get("NETWORK_INTERFACE")
        logger.info(f"Initializing channel with Domain ID: 0 and network interface: {network_interface}")
        ChannelFactoryInitialize(0, network_interface)
