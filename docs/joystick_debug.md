# Joystick Debugging Guide

> **Scope:** End-to-end debugging of joystick detection from Ubuntu kernel level through Docker into the DOOM project, with a fallback virtual joystick procedure for remote/no-hardware scenarios.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Background: How the Project Detects Joysticks](#2-background-how-the-project-detects-joysticks)
3. [Part A — Physical Joystick: Ubuntu Host-Level Debug](#3-part-a--physical-joystick-ubuntu-host-level-debug)
4. [Part B — Physical Joystick: Docker Container Verification](#4-part-b--physical-joystick-docker-container-verification)
5. [Part C — Physical Joystick: In-Project Runtime Verification](#5-part-c--physical-joystick-in-project-runtime-verification)
6. [Part D — Virtual Joystick (Remote / No Hardware Fallback)](#6-part-d--virtual-joystick-remote--no-hardware-fallback)
7. [Part E — Common Pitfalls & Quick Fixes](#7-part-e--common-pitfalls--quick-fixes)

---

## 1. Prerequisites

- Ubuntu 20.04 or later
- `sudo` privileges on the host
- Docker installed and `nvidia-container-toolkit` configured
- (Optional, for physical testing) Xbox 360 / Xbox One / Xbox Series X controller with a working USB cable

---

## 2. Background: How the Project Detects Joysticks

The DOOM project has **two independent joystick initialization paths** that can produce a "no joystick" message:

| Path | File | Severity | When It Runs |
|------|------|----------|--------------|
| **`JoystickManager`** | `src/utils/joystick_interface.py:53` | ⚠️ Warning (non-fatal) | Every time you launch `ros2 run master_manager master_node` |
| **`UnitreeSdk2Bridge.SetupJoystick`** | `src/utils/unitree_sdk2py_bridge.py:272` | 🔴 Fatal (`sys.exit()`) | Only if `USE_JOYSTICK=true` when running the old `unitree_mujoco.py` simulator path |

Both use **pygame** to enumerate joysticks (`pygame.joystick.get_count()`). Pygame reads from `/dev/input/event*` and `/dev/input/js*` device nodes. The device nodes are created by the Ubuntu kernel when it detects a USB joystick, typically via the `xpad` kernel module.

**Critical rule (from README line 160):** The joystick must be physically connected **before** launching the Docker container. If you connect it after the container starts, the `/dev/input/` device node won't be visible inside the container because Docker only snapshots the device list at container creation time when using `--device /dev/input`.

**The debugging order that always works:** Host kernel → Pygame on host → Docker → Project.

---

## 3. Part A — Physical Joystick: Ubuntu Host-Level Debug

> **Run all commands on the Ubuntu host (outside Docker), with the Xbox controller connected via USB.**

### A.1 — Is the USB controller recognized at the hardware level?

```bash
lsusb | grep -iE "xbox|microsoft|controller|045e"
```

> *`045e` = Microsoft's USB vendor ID. Xbox controllers appear as `045e:...`.*

#### ✅ SUCCESS: Output shows a device (e.g., `045e:02ea Microsoft Corp. Xbox One Controller`)
→ Proceed to [A.2](#a2--is-the-xpad-kernel-module-loaded).

#### ❌ FAILURE: Empty output (no Xbox/Microsoft device)
1. **Try a different USB port** — prefer USB 3.0 (blue) ports on the back of the desktop. Front-panel ports or USB hubs may be under-powered.
2. **Try a different USB cable** — Xbox controller cables are detachable and commonly fail. Some cables are charge-only (no data lines). Test with a known data-capable cable (e.g., one that works for a phone).
3. **Check if the controller vibrates** when plugged in — vibration confirms it's receiving power. No vibration at all suggests a dead controller or bad cable.
4. **Check kernel logs live** while re-plugging:
   ```bash
   sudo dmesg -w
   ```
   Now unplug and replug the controller. You should see lines like:
   ```
   [  +0.123456] usb 1-2: new full-speed USB device ...
   [  +0.456789] usb 1-2: New USB device found, idVendor=045e, ...
   ```
   - **If you see new USB messages:** The hardware is detected, but the kernel may not have the right driver loaded. Proceed to [A.2](#a2--is-the-xpad-kernel-module-loaded).
   - **If no new messages appear at all:** The USB port or cable is not transmitting. Try a different desktop PC entirely to rule out a faulty controller.
   - **If you see USB errors** (e.g., `-71`, `-32`, `cannot enumerate`): The controller may be drawing too much power (try a powered USB hub) or is faulty. Try another PC.

---

### A.2 — Is the xpad kernel module loaded?

```bash
lsmod | grep xpad
```

#### ✅ SUCCESS: Output shows `xpad` loaded (e.g., `xpad  57344  0`)
→ Proceed to [A.3](#a3--is-a-joystick-device-node-created).

#### ❌ FAILURE: Empty output (xpad not loaded)
1. **Load the module manually:**
   ```bash
   sudo modprobe xpad
   ```
2. **Re-check:**
   ```bash
   lsmod | grep xpad
   ```
3. **If still empty after `modprobe xpad`:**
   ```bash
   dmesg | tail -20 | grep -i xpad
   ```
   - **If you see `xpad: Unknown symbol ...` or similar:** Your kernel headers may be mismatched. Install the correct headers:
     ```bash
     sudo apt update
     sudo apt install linux-headers-$(uname -r)
     sudo modprobe xpad
     ```
   - **If you see nothing at all:** The `xpad` module may not exist. Install `xboxdrv` as an alternative:
     ```bash
     sudo apt update
     sudo apt install xboxdrv
     ```
     Then run it:
     ```bash
     sudo xboxdrv --silent --mimic-xpad --type xbox360
     ```
     This creates a virtual joystick device from the physical controller. If `xboxdrv` fails with `--detach-kernel-driver`, first unload xpad:
     ```bash
     sudo rmmod xpad
     sudo xboxdrv --silent --mimic-xpad --type xbox360
     ```

---

### A.3 — Is a joystick device node created?

```bash
ls -la /dev/input/ | grep -E "js|event"
```

You should see `/dev/input/js0` (a joystick device node) or event nodes labeled as joysticks.

```bash
ls -la /dev/input/js*
```

#### ✅ SUCCESS: `/dev/input/js0` exists
→ Proceed to [A.4](#a4--can-jstest-read-the-controller).

#### ❌ FAILURE: No `/dev/input/js0`

1. **Check the full input device registry:**
   ```bash
   cat /proc/bus/input/devices | grep -A 10 -i "xbox\|xpad\|gamepad\|045e"
   ```
   - **If you find an Xbox entry here but no `js0`:** The controller is detected as a HID device, not a legacy joystick. This is fine — pygame can read it via `/dev/input/event*`. The event node is listed in the output (e.g., `event3`). Skip to [A.4](#a4--can-evtest-read-the-controller).
   - **If no Xbox entry at all:** The kernel driver didn't attach. Return to [A.2](#a2--is-the-xpad-kernel-module-loaded) and try `xboxdrv` as alternative.

2. **Manually create a js0 symlink to the event device:**
   ```bash
   cat /proc/bus/input/devices | grep -B 5 -A 10 "Xbox"
   # Look for the line: "H: Handlers=sysrq kbd eventX"
   # Note X (e.g., event3)
   sudo ln -s /dev/input/eventX /dev/input/js0
   ```

3. **Verify permissions:** The `/dev/input/event*` nodes should be readable by the `input` group. Add your user:
   ```bash
   sudo usermod -aG input $USER
   # Log out and back in for this to take effect
   ```

---

### A.4 — Can `jstest` or `evtest` read the controller?

#### Option 1: jstest (only works if `/dev/input/js0` exists)

```bash
sudo apt install joystick -y
jstest /dev/input/js0
```

Move the sticks and press buttons. You should see live updates to axis/button values.

#### ✅ SUCCESS: Values update when you move sticks/press buttons
→ Host-level detection is complete. Proceed to [Part B](#4-part-b--physical-joystick-docker-container-verification).

#### ❌ FAILURE: `No such file or directory`

Try `evtest` instead (works with event devices even without a js0 node).

#### Option 2: evtest (universal — works with any event device)

```bash
sudo apt install evtest -y
```

1. **Find the correct event device:**
   ```bash
   cat /proc/bus/input/devices | grep -B 5 -A 10 "Xbox\|xpad\|045e"
   ```
   Note the `eventX` number from the `Handlers` line (e.g., `event3`).

2. **Run evtest on it:**
   ```bash
   sudo evtest /dev/input/event3   # replace 3 with your event number
   ```
   Move sticks / press buttons. You should see `EV_ABS`, `EV_KEY` events.

#### ✅ SUCCESS: evtest shows events
→ Pygame should detect this. Proceed to [A.5](#a5--can-pygame-detect-the-joystick).

#### ❌ FAILURE: No events from evtest
→ The kernel driver is not sending events. Try:
1. `sudo rmmod xpad && sudo modprobe xpad` then replug.
2. Install and run `xboxdrv --mimic-xpad` (see [A.2](#a2--is-the-xpad-kernel-module-loaded) failure path).
3. Test on another Linux PC.
4. As absolute last resort, try the virtual joystick fallback in [Part D](#6-part-d--virtual-joystick-remote--no-hardware-fallback).

---

### A.5 — Can pygame detect the joystick?

```bash
python3 -c "
import pygame
pygame.init()
pygame.joystick.init()
count = pygame.joystick.get_count()
print(f'Joystick count: {count}')
for i in range(count):
    js = pygame.joystick.Joystick(i)
    js.init()
    print(f'  Joystick {i}: {js.get_name()}')
    print(f'  Axes: {js.get_numaxes()}, Buttons: {js.get_numbuttons()}, Hats: {js.get_numhats()}')
"
```

#### ✅ SUCCESS: `Joystick count: 1` (or more) with name and capabilities
→ Pygame works. Proceed to [Part B](#4-part-b--physical-joystick-docker-container-verification).

#### ❌ FAILURE: `Joystick count: 0`

1. **Check pygame sees the joystick on the event system (not js node):**
   ```bash
   python3 -c "
import pygame
pygame.init()
print(f'Pygame version: {pygame.version.ver}')
print(f'SDL version: {pygame.get_sdl_version()}')
print(f'SDL video drivers: {pygame.display.get_driver()}')
"
   ```
2. **Pypermission fallback:** If your user is in the `input` group, pygame may need the `joystick` module to be explicitly initialized before calling `get_count`:
   ```bash
   python3 -c "
import pygame
pygame.init()
pygame.joystick.init()
# Sometimes needs a small delay
import time; time.sleep(0.5)
print(f'Count after delay: {pygame.joystick.get_count()}')
"
   ```
3. **Check if pygame.joystick.init() raises any error that's being swallowed.**
   ```bash
   python3 -c "
import pygame, sys
try:
    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    print(f'Joysticks: {count}')
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
"
   ```
4. **Install joystick support explicitly (some headless systems omit SDL joystick):**
   ```bash
   sudo apt install libsdl2-2.0-0 libsdl2-dev -y
   pip install --force-reinstall pygame
   ```
5. **Run the pygame test as root (bypasses all permission issues):**
   ```bash
   sudo python3 -c "import pygame; pygame.init(); pygame.joystick.init(); print(f'Root count: {pygame.joystick.get_count()}')"
   ```
   - **If root works but user doesn't:** It's a permissions issue. Add user to `input` group and log out/in:
     ```bash
     sudo usermod -aG input $USER
     # Then log out fully and back in
     ```
   - **If root also returns 0:** The device isn't detected at the kernel level. Return to [A.3](#a3--is-a-joystick-device-node-created).

---

## 4. Part B — Physical Joystick: Docker Container Verification

> **⚠️ CRITICAL: Connect the joystick to the host BEFORE running `./doom.sh -e`.** Docker snapshots `/dev/input` at container creation time. If you plug it in later, it won't be visible inside.

### B.1 — Launch the Docker container

```bash
./doom.sh -e           # standard image
# OR
./doom.sh -e --safety  # safety filter image
```

#### ❌ FAILURE: Container fails to start
1. `systemctl status docker` — Docker might not be running.
2. `./doom.sh -b` to rebuild (or `./doom.sh -s` for safety image).
3. Check for GPU errors: `nvidia-smi` — confirm nvidia-container-toolkit is installed.

---

### B.2 — Verify `/dev/input` devices are visible inside the container

Inside the container, run:

```bash
ls -la /dev/input/
ls -la /dev/input/js* 2>/dev/null || echo "No js* devices"
```

#### ✅ SUCCESS: You see the same device nodes as on the host (event devices and/or js0)
→ Proceed to [B.3](#b3--verify-pygame-inside-docker).

#### ❌ FAILURE: `/dev/input/` is empty or missing devices present on the host

1. **Exit the container** (`exit`).
2. **Verify on the host** that the joystick IS connected and `/dev/input/js0` or event nodes exist (see [A.3](#a3--is-a-joystick-device-node-created)).
3. **Delete the container** so it re-creates with the current device list:
   ```bash
   ./doom.sh -d
   ```
4. **Try a more aggressive device passthrough.** Edit `doom.sh` line 127. Change:
   ```bash
   --device /dev/input \
   ```
   to:
   ```bash
   -v /dev/input:/dev/input \
   ```
   This mounts the entire `/dev/input` directory instead of granting single-device access.
5. **Re-enter with the modified script:**
   ```bash
   ./doom.sh -e
   ```
6. **Check permissions inside the container:**
   ```bash
   groups  # should include input (107)
   ls -la /dev/input/
   ```

---

### B.3 — Verify pygame inside Docker

Inside the container:

```bash
python3 -c "
import pygame
pygame.init()
pygame.joystick.init()
count = pygame.joystick.get_count()
print(f'Joystick count: {count}')
for i in range(count):
    js = pygame.joystick.Joystick(i)
    js.init()
    print(f'  Joystick {i}: {js.get_name()}')
    print(f'  Axes: {js.get_numaxes()}, Buttons: {js.get_numbuttons()}, Hats: {js.get_numhats()}')
"
```

#### ✅ SUCCESS: 1+ joysticks detected
→ Container can see the joystick. Proceed to [Part C](#5-part-c--physical-joystick-in-project-runtime-verification).

#### ❌ FAILURE: 0 joysticks

1. **Check the `input` group membership inside the container:**
   ```bash
   id
   ```
   You should see `107(input)` in the group list. The `doom.sh` passes `--group-add 107` which should handle this.

2. **If `input` group is missing:**
   ```bash
   # On the host, check the input group GID:
   getent group input
   # Note the number (usually 107). If different, update doom.sh line 128.
   ```

3. **Test pygame as root inside the container** (if possible):
   ```bash
   sudo python3 -c "import pygame; pygame.init(); pygame.joystick.init(); print(pygame.joystick.get_count())"
   ```
   - **If root works but user doesn't:** Add `-u 0:0` to the Docker run command in `doom.sh` (temporarily, for testing only — not for production). Or fix the group mapping.
   - **If root also returns 0:** The host's `/dev/input` wasn't properly forwarded. Exit container, verify host devices exist, delete container, re-enter.

4. **Check `/dev/input` permissions inside the container:**
   ```bash
   ls -la /dev/input/event*
   ```
   They should be `crw-rw---- root input`. If they're `crw-rw---- root root`, the user's group is wrong.

---

### B.4 — Verify the JoystickManager log output

When you launch the master node, you should see ONE of these two log lines:

**With joystick detected:**
```
Joystick initialized successfully
```

**Without joystick detected:**
```
No joystick detected
```

Source: `src/utils/joystick_interface.py:53,57`

If you see the warning, and you've confirmed pygame detects the joystick inside Docker (B.3), then the issue is in the project code — proceed to [Part C](#5-part-c--physical-joystick-in-project-runtime-verification).

---

## 5. Part C — Physical Joystick: In-Project Runtime Verification

> **Run the full DOOM pipeline and verify joystick control works end-to-end.**

### C.1 — Launch the simulator (Terminal 1)

```bash
./doom.sh -e     # or ./doom.sh -e --safety
cd src/
python3 simulate.py --task=rl-velocity-sim-go2
```

This starts the MuJoCo simulation. It should not crash with a "No gamepad detected" message — the newer `SimRobotInterface` path does NOT call `SetupJoystick` (see [Background](#2-background-how-the-project-detects-joysticks)).

#### ❌ FAILURE: Simulator exits with "No gamepad detected."
This means you're hitting the old `unitree_mujoco.py` path which calls `UnitreeSdk2Bridge.SetupJoystick()` with `USE_JOYSTICK=true`. The `SetupJoystick` method in `src/utils/unitree_sdk2py_bridge.py:272` does a hard `sys.exit()`.

**Fix:**
- If you need to run the old `unitree_mujoco.py` path directly, ensure the joystick is connected BEFORE running it.
- If you're hitting this via `simulate.py`, it means someone added a `SetupJoystick` call where there shouldn't be one — check `sim_robot_interface.py` or `simulate.py`.

---

### C.2 — Launch the master node (Terminal 2)

```bash
./doom.sh -a     # attach to existing container
source setup_local.sh
ros2 run master_manager master_node --task rl-velocity-sim-go2 --debug
```

Adding `--debug` enables debug-level logging including the joystick thread status.

#### Watch the log output:

| Log message | Meaning |
|-------------|---------|
| `Joystick initialized successfully` | ✅ Joystick detected by `JoystickManager` |
| `No joystick detected` | ⚠️ Joystick not found (but master continues) |
| `Joystick thread started (daemon=True)` | Thread started in non-debug mode |
| `Joystick thread started (daemon=False)` | Thread started in debug mode |
| `Joystick state: XXXXX` | Buttons/axes being read (only in debug mode) |

---

### C.3 — Test mode switching with the joystick

With the master node running and the simulator showing the robot:

| Button | Expected Result |
|--------|----------------|
| `Select` | Switches to `ZERO` mode (robot goes limp) |
| `Start` | Switches from ZERO to `STAY_DOWN` |
| `Up` (D-pad) | `STAY_DOWN` → `STAND_UP` |
| `Down` (D-pad) | `STAND_UP` → `STAND_DOWN` |
| `Start` | From any other mode → `DAMPING` (safe stop) |
| `L1 + R1` | `STAND_UP` → Cycles through main locomotion modes |

Mappings are defined in `src/utils/joystick_interface.py:158-267`.

#### ✅ SUCCESS: Mode changes when you press buttons
→ Joystick fully functional. Debug complete.

#### ❌ FAILURE: No response to buttons

1. **Enable debug logging** and check the joystick thread:
   ```bash
   ros2 run master_manager master_node --task rl-velocity-sim-go2 --debug
   ```
   Look for:
   - `"Joystick thread started"` — thread is running
   - `"Joystick state: XXXXX"` — raw state bitmap is being updated
   - If you DON'T see state updates, the joystick thread isn't reading the controller.

2. **Check your task config file.** Some tasks might have `USE_JOYSTICK=false` or different `JOYSTICK_TYPE`. Check the relevant YAML config (e.g., `src/robots/go2/config/go2_velocity.yaml`).

3. **Add a manual verification in the joystick thread.** Edit `src/utils/joystick_interface.py` around line 297-298, add:
   ```python
   self.logger.info(f"Joystick state raw: {key_value:016b}")
   ```
   Re-run and check the output. If you see `0000000000000000` even when pressing buttons, pygame is not reading from the device.

4. **Verify controller-specific mappings.** If you're in the RL-VELOCITY mode and mode-switching works but velocity commands don't, the controller's `get_joystick_mappings()` may need to be checked. The joystick thread reads these mappings at `joystick_interface.py:270-271`. Try a simpler controller that doesn't override mappings to test.

5. **Check the controller instance.** Print the active controller name:
   ```python
   self.logger.info(f"Active controller: {self.active_controller.__class__.__name__}")
   ```

---

## 6. Part D — Virtual Joystick (Remote / No Hardware Fallback)

> **Use this if you have no physical joystick, or physical debugging in Parts A–C has completely failed. Tested on remote SSH sessions with no GUI.**

### D.1 — Install prerequisites on the host

```bash
sudo apt update
sudo apt install -y python3-pip linux-modules-extra-$(uname -r)
```

#### ❌ FAILURE: Package not found
1. Enable universe repository:
   ```bash
   sudo add-apt-repository universe
   sudo apt update
   ```

---

### D.2 — Load the `uinput` kernel module

The `uinput` kernel module is required to create virtual input devices from userspace.

```bash
sudo modprobe uinput
lsmod | grep uinput
```

#### ✅ SUCCESS: uinput is loaded
→ Proceed to [D.3](#d3--option-a---xboxdrv-recommended).

#### ❌ FAILURE: uinput not found
1. Install extra kernel modules:
   ```bash
   sudo apt install linux-modules-extra-$(uname -r)
   ```
2. Reload:
   ```bash
   sudo modprobe uinput
   ```
3. **Still fails?** Update kernel and reboot:
   ```bash
   sudo apt upgrade -y linux-image-generic
   sudo reboot
   ```

---

### D.3 — Option A — xboxdrv (Recommended, matches Xbox mappings exactly)

xboxdrv creates a `/dev/input/js0` device that behaves identically to a physical Xbox 360 controller.

```bash
sudo apt install -y xboxdrv
sudo xboxdrv --silent --mimic-xpad --type xbox360 --detach-kernel-driver &
```

Wait 2 seconds, then verify:

```bash
ls -la /dev/input/js0
```

#### ✅ SUCCESS: `/dev/input/js0` exists
→ xboxdrv is running. The virtual joystick is ready. Skip to [D.5](#d5--verify-virtual-joystick-with-pygame).

#### ❌ FAILURE: xboxdrv crashes or no js0
1. **Unload xpad first** (if xpad is conflicting):
   ```bash
   sudo rmmod xpad
   sudo xboxdrv --silent --mimic-xpad --type xbox360 &
   ```
2. **Check for permission errors:**
   ```bash
   sudo xboxdrv --mimic-xpad --type xbox360   # without --silent to see errors
   ```
3. **If xboxdrv not available:** Use Option B below.

---

### D.4 — Option B — vgamepad (Python-based, more scriptable)

vgamepad creates a virtual Xbox 360 gamepad using the `uinput` interface.

```bash
pip3 install vgamepad
```

Verify installation:

```bash
python3 -c "import vgamepad; print('vgamepad installed successfully')"
```

#### ❌ FAILURE: pip install fails
1. Install pip: `sudo apt install python3-pip`
2. Install build deps: `sudo apt install libudev-dev python3-dev`
3. Retry: `pip3 install vgamepad`

---

### D.5 — Verify virtual joystick with pygame

Once xboxdrv is running OR you have a vgamepad test script active, verify:

```bash
python3 -c "
import pygame
pygame.init()
pygame.joystick.init()
count = pygame.joystick.get_count()
print(f'Joystick count: {count}')
for i in range(count):
    js = pygame.joystick.Joystick(i)
    js.init()
    print(f'  Joystick {i}: {js.get_name()}')
    print(f'  Axes: {js.get_numaxes()}, Buttons: {js.get_numbuttons()}, Hats: {js.get_numhats()}')
"
```

#### ✅ SUCCESS: 1+ joysticks detected
→ Proceed to [D.6](#d6--docker-setup-with-virtual-joystick).

#### ❌ FAILURE: 0 joysticks
1. **Restart the virtual joystick** — kill xboxdrv and restart it.
2. **Check `/dev/input/js0` exists:** `ls -la /dev/input/js0`
3. **Test jstest:** `jstest /dev/input/js0` — should show all axes at 0.
4. **If using vgamepad:** Make sure the vgamepad test script is still running (it needs to keep the device alive). Write this to `virtual_joystick.py` and run it:
   ```python
   import vgamepad as vg
   import time
   
   gamepad = vg.VX360Gamepad()
   print("Virtual Xbox 360 controller active. Idling...")
   try:
       while True:
           time.sleep(1)
   except KeyboardInterrupt:
       print("Stopping virtual controller.")
       gamepad.reset()
   ```
   In a separate terminal, run:
   ```bash
   python3 virtual_joystick.py
   ```
   Now in the original terminal, re-run the pygame test. The vgamepad instance must be running for pygame to see it.

---

### D.6 — Docker setup with virtual joystick

**Critical:** Start xboxdrv / vgamepad **before** launching Docker.

```bash
# Terminal 1: Start virtual joystick
sudo xboxdrv --silent --mimic-xpad --type xbox360 &

# Verify
ls -la /dev/input/js0

# Terminal 2: Launch Docker
./doom.sh -e
```

Inside Docker:
```bash
python3 -c "import pygame; pygame.init(); pygame.joystick.init(); print(f'Container count: {pygame.joystick.get_count()}')"
```

#### ✅ SUCCESS: Joystick detected inside Docker
→ Launch the master node as in [C.2](#c2--launch-the-master-node-terminal-2). The virtual joystick behaves identically to a physical one for the project's purposes.

#### ❌ FAILURE: 0 inside Docker but detected on host
→ Return to [B.2](#b2--verify-devinput-devices-are-visible-inside-the-container) — the device passthrough isn't working. Use `-v /dev/input:/dev/input` instead of `--device /dev/input`.

---

### D.7 — Sending virtual button presses (for testing mode switching)

With xboxdrv, you can send simulated button presses via command-line tools:

```bash
# Install xboxdvr tools
sudo apt install xboxdrv
```

With vgamepad, create a test script that sends specific button presses:

```python
import vgamepad as vg
import time

gamepad = vg.VX360Gamepad()

# Press Start button (mode: STAY_DOWN)
print("Pressing START...")
gamepad.press_button(vg.XUSB_BUTTON.START)
gamepad.update()
time.sleep(0.3)
gamepad.release_button(vg.XUSB_BUTTON.START)
gamepad.update()
time.sleep(1.0)

# Press D-pad Up (mode: STAND_UP)
print("Pressing D-pad UP...")
gamepad.press_button(vg.XUSB_BUTTON.DPAD_UP)
gamepad.update()
time.sleep(0.3)
gamepad.release_button(vg.XUSB_BUTTON.DPAD_UP)
gamepad.update()
time.sleep(1.0)

# Press L1 + R1 (enter locomotion mode)
print("Pressing L1 + R1...")
gamepad.press_button(vg.XUSB_BUTTON.LEFT_SHOULDER)
gamepad.press_button(vg.XUSB_BUTTON.RIGHT_SHOULDER)
gamepad.update()
time.sleep(0.3)
gamepad.release_button(vg.XUSB_BUTTON.LEFT_SHOULDER)
gamepad.release_button(vg.XUSB_BUTTON.RIGHT_SHOULDER)
gamepad.update()

print("Done. Keep running for continued joystick presence.")
time.sleep(300)
```

> **Note:** You need to keep the vgamepad instance running while the DOOM master node is active. The master node's `JoystickManager` thread reads the device state at 50Hz. If the vgamepad script exits, the virtual device disappears and pygame's joystick count drops to 0.

---

## 7. Part E — Common Pitfalls & Quick Fixes

### Pitfall 1: "No joystick detected" warning but master node works fine
**You're fine.** This is just `JoystickManager` warning you. Use the terminal UI (`--ui`) to control the robot instead. Or use the virtual joystick (Part D).

### Pitfall 2: Controller works on host, not in Docker
**Fix:** Delete the container (`./doom.sh -d`), connect the controller, re-enter (`./doom.sh -e`). Docker snapshots `/dev/input` at container creation time.

### Pitfall 3: Controller connects wirelessly (Bluetooth) instead of USB
The project is only tested with USB-wired Xbox controllers. Bluetooth may work but button mapping IDs can differ. If using Bluetooth:
1. Pair the controller via Ubuntu's Bluetooth settings.
2. Check `pygame` detects it: `python3 -c "import pygame; pygame.init(); pygame.joystick.init(); [print(pygame.joystick.Joystick(i).get_name()) for i in range(pygame.joystick.get_count())]"`
3. If the button mappings are wrong (e.g., B/A swapped), you may need to modify the `button_id` mapping in `src/utils/joystick_interface.py:71-80`.

### Pitfall 4: "libSDL2-2.0.so.0: cannot open shared object" when running pygame tests
```bash
sudo apt install libsdl2-2.0-0
```

### Pitfall 5: Container `--device /dev/input` only captures devices present at container creation
**Fix in `doom.sh`:** Replace line 127 `--device /dev/input` with `-v /dev/input:/dev/input`. This mounts the live directory so hot-plugged devices appear. Trade-off: slightly less secure Docker isolation.

### Pitfall 6: Two Xbox controllers connected — which one is `device_id=0`?
The project always uses `device_id=0` (first joystick reported by SDL). To verify which is first:
```bash
python3 -c "import pygame; pygame.init(); pygame.joystick.init(); [print(f'{i}: {pygame.joystick.Joystick(i).get_name()}') for i in range(pygame.joystick.get_count())]"
```
Disconnect the one you don't want to be device 0, or swap USB ports (lower port number = lower device ID typically).

### Pitfall 7: Everything works but joystick thread is "daemon" and exits silently
In debug mode (`--debug`), the joystick thread runs as `daemon=False`, meaning it won't exit silently. If running without `--debug`, thread exits are not logged. Always test with `--debug` first.

---

## Summary: Decision Tree

```
Connect Xbox controller to Ubuntu host via USB
│
├─ lsusb | grep 045e → Found?
│  ├─ YES → lsmod | grep xpad → Loaded?
│  │  ├─ YES → /dev/input/js0 exists?
│  │  │  ├─ YES → jstest works? → pygame works? → Start Docker → Start project ✅
│  │  │  └─ NO  → check /proc/bus/input/devices → find eventX → evtest → pygame
│  │  └─ NO  → modprobe xpad → retry → or xboxdrv fallback
│  └─ NO  → Try different cable/port → Try different PC → Controller faulty?
│
└─ All physical attempts failed
   └─ Virtual joystick (Part D)
      ├─ xboxdrv --mimic-xpad → /dev/input/js0 → pygame → Docker → Project ✅
      └─ vgamepad (VX360Gamepad) → pygame → Docker → Project ✅
```
