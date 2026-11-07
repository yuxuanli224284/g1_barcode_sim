# Locomotion and arm control

(wbc)=
## Whole-body control and walking

### GR00T Decoupled WBC (default)

The legs and waist are controlled by the two ONNX policies of NVIDIA GR00T Decoupled WBC: Balance (standing) and Walk (walking). The files are in `assets/gear_wbc/`, taken unmodified from [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl). The policies run on the CPU with onnxruntime.

- **Switching**: Balance is used when the norm of the velocity command ‖(vx, vy, wz)‖ < 0.05, Walk otherwise; switching is automatic. Both networks share the same observation history and previous action.
- **Rates**: physics step 0.005 s (200 Hz); with decimation 4, the policy, arm IK and camera rendering all run at 50 Hz (`env.policy_dt = 0.02`). At startup the physics step is checked against the GR00T reference configuration.
- **Output**: `[1, 15]`, position targets for 12 leg joints + 3 waist joints: target = output × 0.25 + default angle. The default angles are hip pitch −0.1, knee 0.3, ankle pitch −0.2, and 0 rad for the rest.
- **Actuators**: these 15 joints use an explicit PD actuator group (IdealPD) with kp/kd from the GR00T reference configuration: hip 150/2, knee 200/4, ankle 40/2, waist 250/5. Torque limits (N·m): hip 88, knee 139, ankle 50, waist yaw 88 / roll 50 / pitch 50.

**The input** is `[1, 516]`: the last 6 observation frames of 86 dimensions each, ordered from oldest to newest and padded with zero frames after a reset. Each frame consists of:

| Component | Dims | Scale |
| --- | --- | --- |
| Velocity command (vx, vy, wz) | 3 | × [2, 2, 0.5] |
| Body height command | 1 | raw |
| Torso roll / pitch / yaw command | 3 | raw |
| Pelvis angular velocity | 3 | × 0.5 |
| Projected gravity | 3 | raw |
| 29 joint angles − default angles | 29 | raw |
| 29 joint velocities | 29 | × 0.05 |
| Previous raw policy output | 15 | raw |

(body-pose)=
### Squatting and torso posture

`env.set_body_pose(height=None, torso_rpy=None, duration=1.0, settle=1.0)` makes the robot squat or tilt its torso in place: the height and torso commands are ramped linearly to their targets over `duration` seconds and then held for `settle` seconds, with fall and collision checks at every policy step. The height command range is 0.20–0.74 m (the default 0.74 m is standing); the torso command limits are |roll| ≤ 0.4, |pitch| ≤ 0.6, |yaw| ≤ 0.8 rad. Out-of-range values raise an error instead of being silently clipped. The robot returns to standing after a reset or teleport.

| Height command (m) | 0.74 | 0.60 | 0.50 | 0.40 | 0.30 | 0.25 |
| --- | --- | --- | --- | --- | --- | --- |
| Measured pelvis height (m) | 0.747 | 0.563 | 0.471 | 0.379 | 0.281 | 0.249 |

Measurement conditions: in place, 1 s ramp + 2 s hold per level, one trial per level. No falls and no foot slip at any level; the deeper the squat, the more the pelvis pitches forward (up to about 26°). The 0.20 m level has not been tested. With torso pitch commands of 0.2 / 0.4 / 0.6 rad, the measured forward torso lean is about 12° / 27° / 43° (at 0.6 the waist pitch reaches its joint limit).

(base-motion)=
### Base motion

- `step(vx, vy, wz, duration)`: walks with a velocity given in the root (pelvis) frame for `duration` seconds, then zeroes the velocity command. The command is first clipped to |vx| ≤ 0.4 m/s, |vy| ≤ 0.5 m/s, |wz| ≤ 1.0 rad/s. The state is checked at every policy step, and the call ends early on a fall or collision. The command is held within the process, unlike the upstream DDS interface, which clears it after one read.
- `stand_still(duration)`: equivalent to `step(0, 0, 0, duration)`.
- `move_relative(dx, dy, dyaw, timeout=10.0, pos_tol=None, yaw_tol=None, speed_scale=1.0)`: odometry-based closed-loop motion with the goal given in the starting root frame. The control law is a saturated P controller: full limit velocity while the error exceeds the braking distance (0.15 m / 0.30 rad), linear deceleration inside it. While an axis is still outside its tolerance, its command does not drop below a minimum (vx 0.15, vy 0.3, wz 0.6) so the robot does not stall at low speeds. After each round the robot first settles (speed < 0.03 m/s for 0.3 s, waiting at most 3 s) before the error is re-checked, for at most 3 rounds. Default tolerances are 3 cm / 0.03 rad.
- **Odometry**: `T_world_root_odom` is not the output of a state estimator but the true pose plus deterministic drift: ±2% displacement error per meter and ±1° heading error per meter, with signs determined by the reset seed; roll / pitch are taken from the true pose.
- **Status**: a torso tilt above 45° is reported as `fell`; any vertex of the robot's collision shapes (including the scanner outline) entering furniture or a box is reported as `collision`. `collision` is only a status report and does not prevent physical contact; clearance is computed only from vertices and is an upper bound.

(unitree-policy)=
### Alternative: upstream unitree locomotion policy

Set `configs/robot.yaml: locomotion: unitree` (or use `teleop.py --policy policy1.onnx`) to use the Wholebody locomotion policy that ships with unitree_sim_isaaclab (`assets/model/policy1.onnx` or `policy.onnx` in the upstream repository). It controls only the 12 leg joints with the waist fixed and has no height or torso commands, so `set_body_pose` is unavailable. It is implemented by subclassing upstream's `DDSRLActionProvider` and only replacing DDS communication with an in-process command object; observation construction, inference and substepping are upstream's original code. In this project's tests, GR00T drifted less while standing, reached `move_relative` goals more often, could walk backward and executed low-speed commands, so it is the default.

| Parameter | Default | Configured in |
| --- | --- | --- |
| Locomotion controller | `gear_wbc` (alternative `unitree`) | `robot.yaml: locomotion` |
| Balance / Walk switching threshold | ‖(vx, vy, wz)‖ = 0.05 | `robot.yaml: gear_wbc.switch_norm` |
| Velocity limits | vx 0.4 m/s, vy 0.5 m/s, wz 1.0 rad/s | `robot.yaml: velocity_limits` |
| Height command range | [0.20, 0.74] m | `robot.yaml: gear_wbc.height_cmd_range` |
| Torso command limits | roll 0.4, pitch 0.6, yaw 0.8 rad | `robot.yaml: gear_wbc.rpy_cmd_limit_rad` |
| `move_relative` tolerances / max rounds | 3 cm / 0.03 rad / 3 rounds | `robot.yaml: move_relative` |
| Fall detection | tilt > 45° | `robot.yaml: fall.tilt_deg` |
| Odometry drift | 2% / m, 1° / m | `robot.yaml: odom` |
| Physics device | CPU | `robot.yaml: physics_device` |

(arm)=
## Arm control

The arms are not controlled by the locomotion policy. Each arm has 7 joints whose position targets come directly from IK or joint commands and are tracked by the upstream implicit PD actuators. At every policy step (50 Hz) the IK is first updated from the measured joint state and then the locomotion policy runs once, so the IK is closed-loop step by step.

1. **Wrist position**: `set_wrist_target(side, pos_root)`, `move_wrist(side, dx, dy, dz)`. Damped least-squares differential IK that controls only the position of `{side}_hand_base_link` (the gripper base) in the root frame, not its orientation. Each step computes dq = gain·Jᵀ(JJᵀ + λ²I)⁻¹e with gain 0.1 and λ 0.05, using the Jacobian from PhysX. Targets are restricted to a box of ±0.25 m around the default wrist position; unreachable targets stop at the nearest reachable point.
2. **Scanner aiming (left arm)**: `aim_scanner(tip_pos_root, dir_root, anchor="odom")`, `aim_scanner_at(point_root, normal_root, distance=0.20, anchor="odom")`. A 5-D task: 3-D tip position + 2-D scan direction, without constraining the roll about the scan axis; the direction error is multiplied by 0.20 m/rad (the nominal scan distance) to weight it against the position error. With `anchor="odom"` the target is fixed in the odometry frame, so base drift is gradually compensated; with `"root"` the target moves with the body. Aim mode is not limited by the wrist box; the residual is reported in `obs["scanner_aim_error"]`.
3. **Joint space**: `env.arm.set_joint_targets(side, q, n_steps)` interpolates linearly from the current command to the 7-D joint target `q` over `n_steps` policy steps. There is no corresponding wrapper method on `env`.

Common limits: joint changes are at most 0.05 rad per step (i.e. 2.5 rad/s) and are clamped to the soft joint limits. `reset_arms()` clears the targets and returns both arms to the default posture (shoulder pitch 0.35, shoulder roll ±0.18, elbow 0.87 rad, hands in front of the hips, wrists at about (0.0, ±0.22, −0.09) m in the root frame).
