import argparse
import os

from robot_interfaces.sim_robot_interface import SimRobotInterface
from tasks.task_configs import TASK_CONFIG
from utils.config_loader import load_config
from utils.logger import get_logger


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="ATARI DOOM Mujoco Simulator")
    parser.add_argument(
        "--task",
        type=str,
        default="rl-mps-sim-go2",
        help="Task name to run (e.g., rl-sim, mpc-real).",
    )
    parser.add_argument("--log", type=str, default="test", help="Experiment name to log information")
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without graphical viewer (headless mode)",
    )
    parser.add_argument("--eval-config", type=str, default=None, help="Path to eval scenario YAML")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (numpy, torch, python random)")
    args = parser.parse_args()

    if args.seed is not None:
        import random
        import numpy as _np
        import torch as _torch
        random.seed(args.seed)
        _np.random.seed(args.seed)
        _torch.manual_seed(args.seed)

    # Load task-specific configurations
    if args.task not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {args.task}. Check tasks/task_configs.py for available tasks.")

    task_configs = TASK_CONFIG[args.task]

    # Load individual configurations
    robot_interface_config = load_config(task_configs["robot_interface"])

    # Setup logger with a more specific log file path
    log_file = os.path.join("logs", args.log, f"{args.task}_simulate.log")
    logger = get_logger(f"{args.task}_simulate", log_file)
    logger.info(f"Task Name: {args.task}")
    logger.info(f"Robot Config: {robot_interface_config}")

    # Build eval hooks if running in eval mode
    eval_hooks = None
    if args.eval_config is not None:
        import yaml
        from eval.sim_eval_hooks import SimEvalConfig, SimEvalHooks
        with open(args.eval_config) as fh:
            raw = yaml.safe_load(fh)
        eval_cfg = SimEvalConfig.from_dict(raw)
        logger.info(f"Eval mode: scenario={eval_cfg.scenario}, max_time={eval_cfg.max_time}s")
        # SimEvalHooks needs mj_model / mj_data / ball_launcher / stop_event from the
        # interface, so we construct it after the interface is created (see below).
        _eval_cfg_raw = eval_cfg  # passed through after interface construction

    if args.seed is not None:
        robot_interface_config["SEED"] = args.seed

    logger.info("Starting Simulation.")
    sim_interface = SimRobotInterface(robot_interface_config, headless=args.headless)

    if args.eval_config is not None:
        eval_hooks = SimEvalHooks(
            config=_eval_cfg_raw,
            mj_model=sim_interface.mj_model,
            mj_data=sim_interface.mj_data,
            ball_launcher=sim_interface.ball_launcher,
            stop_event=sim_interface.stop_event,
        )
        sim_interface.eval_hooks = eval_hooks

    web_viewer_url = sim_interface.get_web_viewer_url()
    if web_viewer_url is not None:
        logger.info(f"Web viewer URL: {web_viewer_url}")

    web_share_url = sim_interface.get_web_viewer_share_url()
    if web_share_url is not None:
        logger.info(f"Web viewer share URL: {web_share_url}")

    try:
        sim_interface.sim_thread.join()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping simulation.")
    finally:
        sim_interface.stop(logger)


if __name__ == "__main__":
    main()
