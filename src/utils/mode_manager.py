import threading
import time
from typing import Any, Dict, Optional

from commands.command_manager import CommandManager
from controllers.stand_controller import ControllerBase
from state_manager.obs_manager import ObservationManager


class ModeManager:
    """
    A flexible mode management system that allows dynamic registration of modes (controllers).
    Modes are defined as a dictionary of controllers, where the key is the mode name and the value is a dictionary of controllers.

    Thread-safety: set_mode() and get_active_controller() are guarded by a re-entrant lock.
    A command cooldown (default 0.2 s) prevents rapid successive mode switches
    from the UI thread that could cause the control-loop thread to see a stale controller.
    """

    def __init__(self, logger=None, device="cuda:0"):
        self._modes: Dict[str, Dict[str, ControllerBase]] = {}
        self._current_mode: Optional[str] = None
        self._current_submode: Optional[str] = None
        self._mode_obs_managers: Dict[str, ObservationManager] = {}
        self._submode_cmd_managers: Dict[str, CommandManager] = {}
        self.logger = logger
        self.device = device

        # Thread-safety
        self._lock = threading.RLock()
        self._last_command_time = 0.0
        self._command_cooldown = 0.2  # seconds — matches joystick cooldown

    def register_mode(self, mode_name: str, controllers: Dict[str, ControllerBase]):
        """
        Register a new mode with individual observation manager and command manager for each submode.

        :param mode_name: Name of the mode
        :param controllers: Dictionary of controllers for this mode
        """
        # Create an observation manager for each submode
        obs_managers = {
            submode_name: ObservationManager(logger=self.logger, device=self.device)
            for submode_name in controllers.keys()
        }

        self._modes[mode_name] = controllers
        self._mode_obs_managers[mode_name] = obs_managers

        # Pass corresponding obs manager to each controller
        for submode_name, controller in controllers.items():
            if hasattr(controller, "set_obs_manager"):
                controller.set_obs_manager(obs_managers[submode_name])

            if hasattr(controller, "register_commands"):
                self._submode_cmd_managers[submode_name] = CommandManager(logger=self.logger)
                controller.set_cmd_manager(self._submode_cmd_managers[submode_name])

    def set_mode(self, mode_name: str, submode: Optional[str] = None, force: bool = False):
        """
        Set the current mode and optional submode.

        :param mode_name: Name of the mode to set
        :param submode: Optional submode within the mode
        :param force: Bypass cooldown (used for safety-critical modes like DAMPING on shutdown)
        :raises ValueError: If mode or submode is not registered
        """
        with self._lock:
            # Enforce command cooldown to prevent rapid mode switching
            current_time = time.time()
            if not force and current_time - self._last_command_time < self._command_cooldown:
                if self.logger:
                    self.logger.debug(
                        f"Ignoring mode switch to {mode_name}/{submode} — "
                        f"cooldown remaining: {self._command_cooldown - (current_time - self._last_command_time):.3f}s"
                    )
                return

            if mode_name not in self._modes:
                raise ValueError(f"Mode {mode_name} not registered")

            if submode is not None and submode not in self._modes[mode_name]:
                raise ValueError(f"Submode {submode} not registered for mode {mode_name}")

            # Deactivate existing controller
            if self._current_submode and self._current_mode:
                controller = self._modes[self._current_mode][self._current_submode]
                if hasattr(controller, "active"):
                    controller.active = False

            # Log the transition
            previous_mode = self._current_mode
            previous_submode = self._current_submode

            self._current_mode = mode_name
            self._current_submode = submode
            self._last_command_time = current_time

            if submode is not None:
                self.logger.debug(f"Mode set to: {self._current_mode} - {self._current_submode}")

            # Get the new controller
            controller = None
            if self._current_submode:
                controller = self._modes[self._current_mode][self._current_submode]
            else:
                controller = self._modes[self._current_mode].get("default", None)

            # Log stale-controller potential: if a mode change was requested and
            # the previous controller was an RL controller with active threads,
            # the control-loop thread may still hold a reference to it for one iteration.
            if self.logger and previous_submode and previous_mode:
                try:
                    prev_ctrl = self._modes[previous_mode][previous_submode]
                    if hasattr(prev_ctrl, "use_threading") and prev_ctrl.use_threading:
                        self.logger.debug(
                            f"Deactivated threaded controller {prev_ctrl.__class__.__name__} — "
                            f"control loop may use stale reference for one iteration"
                        )
                except (KeyError, AttributeError):
                    pass

            # Run set_mode function if it exists in the controller
            if controller and hasattr(controller, "set_mode"):
                controller.set_mode()
            else:
                self.logger.debug(f"Mode set to: {self._current_mode}")

    def get_active_controller(self) -> ControllerBase:
        """
        Get the active controller based on current mode and submode.
        Thread-safe: acquires the mode lock so the returned controller
        is guaranteed to correspond to the current mode/submode pair.

        :return: Active controller
        :raises ValueError: If no mode is set
        """
        with self._lock:
            if self._current_mode is None:
                raise ValueError("No mode is currently set")

            if self._current_submode:
                return self._modes[self._current_mode][self._current_submode]

            return self._modes[self._current_mode].get("default", None)

    def get_active_obs_manager(self) -> ObservationManager:
        """
        Get the observation manager for the current submode.

        :return: Active observation manager
        :raises ValueError: If no mode is set
        """
        if self._current_mode is None:
            raise ValueError("No mode is currently set")

        # If no submode, use 'default' for the mode
        if self._current_submode is None:
            return self._mode_obs_managers[self._current_mode]["default"]

        return self._mode_obs_managers[self._current_mode][self._current_submode]

    def get_current_mode_info(self) -> Dict[str, Optional[str]]:
        """
        Get current mode and submode information.
        Thread-safe.

        :return: Dictionary with current mode and submode
        """
        with self._lock:
            return {"mode": self._current_mode, "submode": self._current_submode}

    def get_mode_info(self, mode_name: str) -> Dict[str, Any]:
        """
        Get mode and submode information for a given mode.

        :param mode_name: Name of the mode
        :return: Dictionary with mode and submode
        """
        return {"mode": mode_name, "submode": list(self._modes[mode_name].keys())}
