#!/usr/bin/env python3
"""
launch_eval.py — Launch sim + ROS eval together.

Default: both processes run in this terminal with [SIM]/[ROS] prefixed output.
         Ctrl+C here stops both cleanly.
--tmux  : create a split-pane tmux session instead (requires tmux).
         Exiting either pane kills the whole session.

Usage:
    python3 eval/launch_eval.py --scenario eval/scenarios/scenario_a.yaml
    python3 eval/launch_eval.py --scenario eval/scenarios/scenario_a.yaml --seed 42 --tmux
"""
import argparse
import os
import shlex
import signal
import subprocess
import sys
import threading
import time

RESET = "\033[0m"
BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stream(proc, label, color):
    try:
        for line in proc.stdout:
            sys.stdout.write(f"{color}[{label}]{RESET} {line}")
            sys.stdout.flush()
    except Exception:
        pass


def _build_cmds(src, args):
    """Return (sim_cmd_list, ros_cmd_list)."""
    scenario = os.path.abspath(args.scenario)
    common = ["--task", args.task, "--log", args.log, "--eval-config", scenario]

    # sim gets --seed via CLI; ROS reads seed from the eval YAML (no new CLI arg needed)
    sim_seed_args = ["--seed", str(args.seed)] if args.seed is not None else []
    sim_cmd = [sys.executable, "simulate.py", "--headless"] + common + sim_seed_args
    ros_cmd = ["ros2", "run", "master_manager", "master_node"] + common
    return sim_cmd, ros_cmd


# ---------------------------------------------------------------------------
# Mode 1: inline (default)
# ---------------------------------------------------------------------------

def launch_inline(src, args):
    sim_cmd, ros_cmd = _build_cmds(src, args)
    procs: list[subprocess.Popen] = []

    def kill_all(sig):
        """Kill each subprocess's entire process group (catches ros2→master_node grandchildren)."""
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), sig)
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        p.send_signal(sig)
                    except ProcessLookupError:
                        pass

    def cleanup(signum=None, frame=None):
        print(f"\n{YELLOW}[Launcher]{RESET} Stopping all processes...", flush=True)
        kill_all(signal.SIGTERM)
        deadline = time.time() + 6
        while any(p.poll() is None for p in procs) and time.time() < deadline:
            time.sleep(0.2)
        kill_all(signal.SIGKILL)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # preexec_fn=os.setsid gives each child its own process group so os.killpg
    # reaches all descendants (including the master_node grandchild of ros2 run).
    popen_kwargs = dict(
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        preexec_fn=os.setsid,
    )

    print(f"{YELLOW}[Launcher]{RESET} Starting sim...", flush=True)
    sim = subprocess.Popen(sim_cmd, cwd=src, **popen_kwargs)
    procs.append(sim)
    threading.Thread(target=_stream, args=(sim, "SIM", BLUE), daemon=True).start()

    print(f"{YELLOW}[Launcher]{RESET} Waiting {args.delay}s before starting ROS...", flush=True)
    for _ in range(int(args.delay * 10)):
        if sim.poll() is not None:
            print(f"{YELLOW}[Launcher]{RESET} Sim exited early (rc={sim.returncode}). Aborting.", flush=True)
            sys.exit(1)
        time.sleep(0.1)

    print(f"{YELLOW}[Launcher]{RESET} Starting ROS controller...", flush=True)
    ros = subprocess.Popen(ros_cmd, cwd=src, **popen_kwargs)
    procs.append(ros)
    threading.Thread(target=_stream, args=(ros, "ROS", GREEN), daemon=True).start()

    # Watch: when either exits naturally, kill the other
    while True:
        for i, p in enumerate(procs):
            if p.poll() is not None:
                label = "SIM" if i == 0 else "ROS"
                print(
                    f"\n{YELLOW}[Launcher]{RESET} {label} exited (rc={p.returncode}) "
                    f"— stopping remaining processes.",
                    flush=True,
                )
                cleanup()
        time.sleep(0.3)


# ---------------------------------------------------------------------------
# Mode 2: tmux split-pane
# ---------------------------------------------------------------------------

def launch_tmux(src, args):
    sim_cmd, ros_cmd = _build_cmds(src, args)
    session = f"doom_eval_{os.getpid()}"

    def qjoin(cmd):
        return " ".join(shlex.quote(c) for c in cmd)

    def tmux(*a):
        subprocess.run(["tmux"] + list(a), check=True)

    # Kill the session if Ctrl+C is pressed before we attach
    def _pre_attach_cleanup(signum=None, frame=None):
        subprocess.run(["tmux", "kill-session", "-t", session], stderr=subprocess.DEVNULL)
        sys.exit(0)

    signal.signal(signal.SIGINT, _pre_attach_cleanup)
    signal.signal(signal.SIGTERM, _pre_attach_cleanup)

    # Wrap each command: after it exits, kill the whole session
    sim_wrapped = f"cd {shlex.quote(src)} && {qjoin(sim_cmd)}; tmux kill-session -t {session}"
    ros_wrapped = (
        f"cd {shlex.quote(src)} && "
        f"sleep {args.delay} && {qjoin(ros_cmd)}; tmux kill-session -t {session}"
    )

    # Create detached session (sim in pane 0)
    tmux("new-session", "-d", "-s", session, "-x", "240", "-y", "60")
    tmux("send-keys", "-t", f"{session}:0.0", sim_wrapped, "Enter")
    # Split vertically, ros in pane 1
    tmux("split-window", "-v", "-t", f"{session}:0")
    tmux("send-keys", "-t", f"{session}:0.1", ros_wrapped, "Enter")
    # Give sim pane the larger top portion
    tmux("resize-pane", "-t", f"{session}:0.0", "-y", "40")
    # Focus top pane
    tmux("select-pane", "-t", f"{session}:0.0")

    print(f"{YELLOW}[Launcher]{RESET} Attaching to tmux session '{session}'.", flush=True)
    print(f"{YELLOW}[Launcher]{RESET} Either pane exiting will close the whole session.", flush=True)

    # Replace this process with tmux attach (Ctrl+C in panes goes to pane, not here)
    os.execlp("tmux", "tmux", "attach-session", "-t", session)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Launch Shield-Loco eval (sim + ROS)")
    parser.add_argument("--scenario", required=True, help="Path to eval scenario YAML")
    parser.add_argument("--task", default="rl-mps-eval-go2")
    parser.add_argument("--log", default="test")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--delay", type=float, default=5.0, help="Seconds before launching ROS (default 5)")
    parser.add_argument("--tmux", action="store_true", help="Launch in a split tmux session")
    args = parser.parse_args()

    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.tmux:
        launch_tmux(src, args)
    else:
        launch_inline(src, args)


if __name__ == "__main__":
    main()
