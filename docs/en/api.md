(api)=
# Python API: `G1BarcodeEnv`

The whole interface is the `G1BarcodeEnv` class in `g1_barcode_sim/env.py` (it can also be imported lazily with `from g1_barcode_sim import G1BarcodeEnv`). A process can create only one environment, and `close()` terminates the whole process.

(api-quick)=
## Quick start

```python
import math
import numpy as np
from g1_barcode_sim import G1BarcodeEnv
from g1_barcode_sim.camera import inv_T

# Built-in scenes must be passed explicitly; the default is "scenario" (task-suite mode)
env = G1BarcodeEnv(scene="s1_table", headless=True, seed=0)
obs = env.reset()
print(obs["rgb"].shape, obs["depth"].shape)       # (480, 640, 3) (480, 640)

r = env.step(0.3, 0.0, 0.0, duration=1.0)         # 0.3 m/s forward in the root frame for 1 s, then zero velocity
print(r["status"], r["reason"])                   # ok / fell / collision

r = env.move_relative(0.3, 0.0, 0.0, timeout=20.0)  # odometry-based closed loop: another 0.3 m forward
print(r["status"], r["gt_error"])

r = env.set_body_pose(height=0.50)                # squat to a 0.50 m height command (GR00T controller)
print(r["pelvis_z"], r["tilt_deg"])
env.set_body_pose(height=0.74)                    # stand up

# Scan (this example uses the GT standing pose directly; a real algorithm should estimate the label pose from the head camera)
bc = env.get_ground_truth()["barcodes"][0]
c, n = np.array(bc["center"]), np.array(bc["normal"])
env.teleport_root(c[0] + 0.55 * n[0], c[1] + 0.55 * n[1] - 0.20, math.atan2(-n[1], -n[0]))
env.stand_still(0.5)
T = inv_T(env.get_obs()["T_world_root_gt"])       # world -> root
env.aim_scanner_at(T[:3, :3] @ c + T[:3, 3], T[:3, :3] @ n, distance=0.20)
env.stand_still(3.0)                              # wait for the aim to converge
res = env.scan()
print(res["status"], res["content"])
env.close()                                       # terminates the whole process
```

Task-suite mode:

```python
from g1_barcode_sim import G1BarcodeEnv

env = G1BarcodeEnv(headless=True)                 # scene defaults to "scenario"
obs = env.reset(scenario="t1_L0_0000")            # a scenario id, JSON path or dict
gt = env.get_ground_truth()                       # also contains target_box_id, task1_gt_obs_pose, etc. (evaluation only)
obs = env.reset()                                 # no argument: restart the current scenario
obs = env.reset(scenario="stack4f_L0_0001")       # switch directly to another scenario, no new process needed
```

(api-methods)=
## Methods

```{list-table}
:header-rows: 1
:widths: 45 55

* - Method (signatures as in `env.py`)
  - Description and return value
* - `G1BarcodeEnv(scene: str = "scenario", headless: bool = False, seed: int = 0, config_overrides: dict | None = None)`
  - `scene`: `s1_table` / `s2_shelf` / `s3_stack` or `"scenario"`. The top-level keys of `config_overrides` are `scene` / `barcode` / `robot` / `camera` / `scanner`; they are deep-merged in the order `_common.yaml` ← `<scene>.yaml` ← overrides
* - `reset(seed: int | None = None, scenario: dict | str | None = None) -> dict`
  - Returns the first observation. Built-in scenes: the robot returns to its start pose, joints return to their defaults, labels are re-placed according to the seed (when randomization is on), and odometry and simulation time are reset to zero; passing `scenario` raises an error. Task-suite mode: rebuilds the table, boxes, labels and background and places the robot at the scenario's start pose; `scenario` is required on the first call
* - `get_obs() -> dict`
  - Observation dict, see {ref}`api-obs`
* - `step(vx: float, vy: float, wz: float, duration: float) -> dict`
  - Velocity command in the root frame, held for `duration` seconds and then zeroed. Returns `{status: ok|fell|collision, reason, obs, command}`
* - `stand_still(duration: float) -> dict`
  - Stand with zero velocity; same return format as `step`
* - `move_relative(dx: float, dy: float, dyaw: float, timeout: float = 10.0, pos_tol: float | None = None, yaw_tol: float | None = None, speed_scale: float = 1.0) -> dict`
  - Odometry-based closed-loop relative motion (in the starting root frame). Returns `{status: ok|timeout|fell|collision, reason, elapsed_s, requested, odom_displacement, gt_displacement, gt_error{pos_m, yaw_rad}, odom_error_vs_gt}`
* - `set_body_pose(height: float | None = None, torso_rpy=None, duration: float = 1.0, settle: float = 1.0) -> dict`
  - Squat / tilt the torso in place (GR00T controller only). Returns `{status, reason, height_cmd, rpy_cmd, pelvis_z, tilt_deg, obs}`
* - `set_gripper(side: str, opening: float) -> dict`
  - `side`: `left` / `right` / `both`; `opening` from 0 (closed) to 1 (open). Only sets the target; the gripper moves during subsequent steps. Returns `{status, side, applied_to, opening, joint_target, reason}`
* - `set_wrist_target(side: str, pos_root) -> dict`
  - Wrist position target (root frame), tracked by IK at every step. Returns `{status, target_root}`
* - `move_wrist(side: str, dx: float, dy: float, dz: float) -> dict`
  - Offsets the current target (or the default wrist position)
* - `reset_arms() -> dict`
  - Returns both arms to the default posture and clears wrist and aim targets
* - `aim_scanner(tip_pos_root, dir_root, anchor: str = "odom") -> dict`
  - Places the scanner tip at `tip_pos_root` with the scan direction along `dir_root` (root frame at call time). Returns `{status, side, anchor, tip_pos_root, dir_root}`
* - `aim_scanner_at(point_root, normal_root, distance: float = 0.20, anchor: str = "odom") -> dict`
  - Places the scanner tip `distance` away from the label center along its outward normal, aimed perpendicular to the label
* - `scan() -> dict`
  - Pulls the trigger, see {ref}`scan`
* - `get_ground_truth(d: float = 0.20) -> dict`
  - GT (evaluation only), see {ref}`gt`
* - `teleport_root(x: float, y: float, yaw: float) -> None`
  - Teleport for debugging: joints return to their defaults (the scanner hand stays closed), arm targets are cleared and odometry is reset to the true pose
* - `render_aux(T_world_cam_ros, hfov_deg: float | None = None) -> dict`
  - Renders the auxiliary camera at an arbitrary pose and returns `{rgb, depth, K}` without stepping physics. The pose can be built with `g1_barcode_sim.camera.look_at_ros(eye, target)`
* - `topdown() -> dict` / `oblique_views() -> dict`
  - Top-down view (image right is +x, image up is +y) / oblique views defined under `views` in the scene YAML
* - `save_snapshot(path: str) -> None`
  - Writes `head_rgb.png`, `head_depth.png` (false color), `head_depth.npy`, `wrist_rgb.png`, `wrist_depth.png`, `topdown.png`, `view_*.png`
* - `camera_rgb(name: str)`
  - RGB image of a single camera: `"head"`, `"wrist"` or the name of a recording camera
* - `close() -> None`
  - Closes the environment and terminates the process
```

Useful attributes: `env.policy_dt` (0.02 s), `env.layout` (current scene layout), `env.cfg` (merged configuration), `env.scanner_side`, `env.arm` (arm IK controller), `env.step_hooks` (list of callbacks invoked after each policy step, e.g. for recording).

(api-obs)=
## Observation dict

| Key | Description |
| --- | --- |
| `rgb` | Head camera, 480×640×3 uint8 |
| `depth` | Head depth, 480×640 float32 in m; 0 outside 0.2–3.0 m |
| `K` | 3×3 head camera intrinsics |
| `T_root_cam` | Pose of the head camera optical frame in root (computed from the current joint angles) |
| `T_world_root_odom` | Odometry pose (with drift) |
| `T_world_root_gt` | True pose, **evaluation only** |
| `T_root_scanner_mount` | Pose of the scanner G frame in root |
| `gripper_state` | `{left, right}`, opening 0–1 |
| `wrist_pos_root` / `wrist_target_root` | Wrist position / IK target (`None` if there is no target) |
| `timestamp` | Simulation time (s) |
| `wrist_rgb`, `wrist_depth`, `wrist_K` | Scanner camera image, depth (along the optical axis, m) and intrinsics (only when the scanner is mounted; same below) |
| `T_root_wrist_cam`, `T_root_scanner_tip` | Poses of the scanner camera optical frame and of the scanner tip (x axis = scan direction) in root |
| `scanner_aim_target`, `scanner_aim_error` | Current aim target `{pos, dir}` (expressed in the current root frame) and error `{pos_m, angle_deg}`; `None` when not aiming |

(api-teleop)=
## Keyboard control

`scripts/teleop.py` is for manually checking the base, wrists, gripper and scanner, with optional recording. With a display, press keys directly in the OpenCV window; without a display (or with `--headless`), type a letter in the terminal and press Enter.

```bash
python scripts/teleop.py --scene s1_table                                  # interactive control
python scripts/teleop.py --scene s1_table --record outputs/my_run.mp4      # interactive control + recording
python scripts/teleop.py --scene s1_table --demo --record outputs/demo.mp4 # built-in control test sequence
python scripts/teleop.py --scene s1_table --demo_scanner --record outputs/scanner_demo.mp4  # scanner test sequence
python scripts/teleop.py --scene s1_table --policy policy1.onnx            # use the upstream unitree locomotion policy instead
```

| Key | Action |
| --- | --- |
| {kbd}`w` / {kbd}`s`, {kbd}`a` / {kbd}`d`, {kbd}`q` / {kbd}`e`, space | Forward / backward, left / right, turn left / turn right, stop. Each key press lasts `--hold` seconds (default 0.5 s) at the limit velocity |
| {kbd}`1` / {kbd}`2` / {kbd}`3` | Subsequent wrist keys act on the left arm / right arm / both arms (default: both) |
| {kbd}`i` / {kbd}`k`, {kbd}`j` / {kbd}`l`, {kbd}`u` / {kbd}`o` | Move the wrist along ±x (forward/back), ±y (left/right), ±z (up/down) in the root frame by `--wrist_step` per press (default 3 cm) |
| {kbd}`r` | Return the arms to the default posture |
| {kbd}`g` / {kbd}`h` | Open / close the gripper (only the right hand, which does not hold the scanner) |
| {kbd}`f` | Toggle aim mode. When on, the scanner holds its current pose; with {kbd}`1` or {kbd}`3` selected, {kbd}`i` {kbd}`k` {kbd}`j` {kbd}`l` {kbd}`u` {kbd}`o` move the scanner tip |
| {kbd}`t` / {kbd}`v`, {kbd}`c` / {kbd}`b` | Aim direction up / down, left / right by `--aim_step_deg` per press (default 5°) |
| {kbd}`n` | Scan (the result is printed in the terminal and shown on the image) |
| {kbd}`p` / {kbd}`x` | Save a snapshot to `outputs/teleop/<scene>/` / quit |

`scripts/run_scene.py` is a simpler scene viewer: {kbd}`W`/{kbd}`S` forward/backward, {kbd}`A`/{kbd}`D` strafe, {kbd}`Q`/{kbd}`E` turn, space stops, {kbd}`O`/{kbd}`C` open / close the gripper (the scanner hand does not move), {kbd}`P` saves a snapshot, {kbd}`X` quits. In headless mode it writes the head image to `outputs/run_scene/<scene>/head_latest.png` every `--save_every` seconds (simulation time), and resets automatically on a fall or collision.

(api-scripts)=
## Scripts

```{list-table}
:header-rows: 1
:widths: 40 45 15

* - Script
  - Purpose
  - Needs Isaac Sim
* - `scripts/run_scene.py --scene S [--seed N] [--randomize] [--max_sim_s T]`
  - Open a built-in scene with keyboard control
  - Yes
* - `scripts/teleop.py`
  - Manual control of base + wrists + gripper + scanner, test sequences, recording
  - Yes
* - `scripts/check_env.py [--scenes ...]`
  - End-to-end self-check; report written to `outputs/check_report.md`
  - Yes
* - `scripts/render_views.py [--scenes ...]`
  - Render top-down, oblique and head camera images to `outputs/<scene>/`
  - Yes
* - `scripts/record_walk.py --scene S [--out F] [--fps 25]`
  - Walk a fixed action sequence and record it (third-person view + head camera picture-in-picture)
  - Yes
* - `scripts/generate_barcodes.py [--type code128|qr] [--scenes ...]`
  - Generate the label textures of the built-in scenes and verify them by decoding
  - No
* - `scripts/testsuite/list.py` / `show.py` / `generate.py`
  - List, inspect and generate task suites (see {ref}`suite-tools`)
  - No
* - `scripts/testsuite/preview.py`
  - Render previews of task-suite scenarios
  - Yes
* - `scripts/testsuite/measure_head_camera.py`
  - Re-measure the nominal head camera parameters (only needed after changing the camera or robot)
  - Yes
* - `examples/goto_barcode.py`, `examples/scanner_camera.py`
  - Minimal examples: walk to a barcode's reference standing pose with `move_relative`; aim and scan
  - Yes
```
