"""NVIDIA GR00T Decoupled WBC lower-body controller (Balance / Walk ONNX), in-process.

Reference: GR00T-WholeBodyControl sim2mujoco (`run_mujoco_gear_wbc.py`, `g1_gear_wbc.yaml`), cross-checked against
IsaacLab-Arena's Isaac Lab 2.3 port and the rhoyn PhysX benchmark (spec in assets/gear_wbc/SOURCE.md).

Policy I/O
  input  [1, 516] = 6 frames x 86, oldest -> newest, zero frames left-padded after a reset
  frame  cmd(vx,vy,wz)*cmd_scale | height_cmd | rpy_cmd(3) | ang_vel_b*ang_vel_scale | projected gravity |
         q29 - default29 | dq29*dof_vel_scale | last raw action (15)
  output [1, 15] raw action for 12 leg + 3 waist joints; target = raw*action_scale + default15 (no clipping)
  Balance net iff ||cmd|| < switch_norm, Walk net otherwise (both share history / last action).

The 14 arm joints and the 4 Dex1 gripper joints are driven from outside (IK / gripper commands), exactly like the
unitree walker: `robot_dds.positions[15:29]` holds the arm targets in Unitree motor order and `gripper_dds.left/right`
the gripper joint targets. `step_policy()` advances physics by control_decimation substeps.

Needs a live env (built after the Isaac Sim app is running).
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np
import onnxruntime as ort
import torch
import yaml

# GR00T body order = Unitree motor order (g1_supplemental_info.py, robot_model.py). The first 15 are the policy outputs.
G29 = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
G15 = G29[:15]
ARM_JOINT_MAPPING = {n: i for i, n in enumerate(G29[15:])}          # name -> 0..13 (motor index - 15), as upstream
GRIPPER_JOINTS = {"left": ["left_hand_Joint1_1", "left_hand_Joint2_1"],
                  "right": ["right_hand_Joint1_1", "right_hand_Joint2_1"]}

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_gear_cfg(gcfg: dict) -> dict:
    """robot.yaml `gear_wbc` block + the GR00T reference yaml it points to (scales / defaults / gains come from there)."""
    ref = yaml.safe_load(open(os.path.join(PKG_ROOT, gcfg["reference_yaml"])))
    if ref["num_actions"] != 15 or ref["obs_history_len"] * 86 != ref["num_obs"]:
        raise ValueError(f"unexpected GR00T reference yaml layout: {ref['num_actions']} actions, {ref['num_obs']} obs")
    return ref


class _ArmCommand:
    def __init__(self):
        self.positions = [0.0] * 29


class _GripperCommand:
    def __init__(self):
        self.left = None
        self.right = None


class GearWbcWalker:
    def __init__(self, env, gcfg: dict):
        self.env = env
        self.robot = env.scene["robot"]
        self.ref = ref = load_gear_cfg(gcfg)
        if abs(ref["simulation_dt"] - env.physics_dt) > 1e-9:
            raise ValueError(f"GR00T was run at dt {ref['simulation_dt']}, env physics_dt is {env.physics_dt}")
        self.decimation = int(ref["control_decimation"])
        names = list(self.robot.data.joint_names)
        self.joint_to_index = {n: i for i, n in enumerate(names)}
        self.idx29 = np.array([names.index(n) for n in G29])
        self.idx15 = self.idx29[:15]
        self.arm_joint_mapping = dict(ARM_JOINT_MAPPING)
        self._grip_idx = {s: [names.index(n) for n in js] for s, js in GRIPPER_JOINTS.items()}
        covered = set(self.idx29.tolist()) | {i for v in self._grip_idx.values() for i in v}
        if covered != set(range(len(names))):
            missing = [names[i] for i in range(len(names)) if i not in covered]
            raise ValueError(f"GR00T walker leaves articulation joints without a target: {missing}")

        self.default15 = np.asarray(ref["default_angles"], dtype=np.float32)
        self.default29 = np.concatenate([self.default15, np.zeros(14, dtype=np.float32)])
        self.cmd_scale = np.asarray(ref["cmd_scale"], dtype=np.float32)
        self.height_cmd = float(gcfg["height_cmd"] if gcfg.get("height_cmd") is not None else ref["height_cmd"])
        self.rpy_cmd = np.asarray(ref["rpy_cmd"], dtype=np.float32)
        # body commands (base height, torso roll/pitch/yaw): defaults restored by reset(); bounds in robot.yaml
        self.height_default, self.rpy_default = self.height_cmd, self.rpy_cmd.copy()
        self.height_range = tuple(float(v) for v in gcfg["height_cmd_range"])
        self.rpy_limit = np.asarray(gcfg["rpy_cmd_limit_rad"], dtype=np.float32)
        self.ang_vel_scale = float(ref["ang_vel_scale"])
        self.dof_vel_scale = float(ref["dof_vel_scale"])
        self.dof_pos_scale = float(ref["dof_pos_scale"])
        self.action_scale = float(ref["action_scale"])
        self.history_len = int(ref["obs_history_len"])
        self.switch_norm = float(gcfg["switch_norm"])

        d = os.path.join(PKG_ROOT, gcfg["model_dir"])
        self._sess = {"balance": ort.InferenceSession(os.path.join(d, gcfg["balance_onnx"]), providers=["CPUExecutionProvider"]),
                      "walk": ort.InferenceSession(os.path.join(d, gcfg["walk_onnx"]), providers=["CPUExecutionProvider"])}
        for k, s in self._sess.items():
            i, o = s.get_inputs()[0], s.get_outputs()[0]
            if i.shape[-1] != 86 * self.history_len or o.shape[-1] != 15:
                raise ValueError(f"{k} onnx has io {i.shape} -> {o.shape}, expected [.,{86 * self.history_len}] -> [.,15]")
        self._in_name = {k: s.get_inputs()[0].name for k, s in self._sess.items()}

        # interface shared with InProcessWholebodyWalker (arm.py / env.py use these)
        self.robot_dds = _ArmCommand()
        self.gripper_dds = _GripperCommand()
        default = self.robot.data.default_joint_pos[0].cpu().numpy()
        self.robot_dds.positions = [0.0] * 15 + [float(default[self.joint_to_index[n]]) for n in G29[15:]]
        self.cmd = np.zeros(3, dtype=np.float32)
        self.active = None                      # "balance" | "walk", the net used at the last step (for logs)
        self.last_raw = None                    # last raw 15-dim output (for logs / tests)
        self._target = self.robot.data.default_joint_pos.clone()
        self.reset()

    # ------------------------------------------------------------------ control surface
    def reset(self):
        self.history = deque([np.zeros(86, dtype=np.float32) for _ in range(self.history_len)], maxlen=self.history_len)
        self.last_action = np.zeros(15, dtype=np.float32)
        self.height_cmd, self.rpy_cmd = self.height_default, self.rpy_default.copy()   # a reset / teleport stands up

    def set_body_command(self, height: float | None = None, rpy=None):
        """GR00T policy inputs 3 (base height, m) and 4-6 (torso roll / pitch / yaw relative to the yaw-only pelvis
        frame, rad); the policy realises them with the legs and the waist. Out-of-range values raise (no clamping)."""
        if height is not None:
            lo, hi = self.height_range
            if not lo - 1e-9 <= float(height) <= hi + 1e-9:
                raise ValueError(f"height_cmd {height} outside {self.height_range} (robot.yaml gear_wbc.height_cmd_range)")
            self.height_cmd = float(height)
        if rpy is not None:
            r = np.asarray(rpy, dtype=np.float32).reshape(3)
            if np.any(np.abs(r) > self.rpy_limit + 1e-6):
                raise ValueError(f"rpy_cmd {r.tolist()} outside +-{self.rpy_limit.tolist()} (robot.yaml gear_wbc.rpy_cmd_limit_rad)")
            self.rpy_cmd = r

    def set_velocity(self, vx: float, vy: float, wz: float):
        self.cmd[:] = (vx, vy, wz)

    def set_gripper_joint_targets(self, left: float, right: float):
        self.gripper_dds.left, self.gripper_dds.right = float(left), float(right)

    # ------------------------------------------------------------------ policy
    def build_frame(self) -> np.ndarray:
        data = self.robot.data
        q = data.joint_pos[0].cpu().numpy()[self.idx29]
        dq = data.joint_vel[0].cpu().numpy()[self.idx29]
        return np.concatenate([
            self.cmd * self.cmd_scale,
            [self.height_cmd],
            self.rpy_cmd,
            data.root_link_ang_vel_b[0].cpu().numpy() * self.ang_vel_scale,
            data.projected_gravity_b[0].cpu().numpy(),
            (q - self.default29) * self.dof_pos_scale,
            dq * self.dof_vel_scale,
            self.last_action,
        ]).astype(np.float32)

    def infer(self) -> np.ndarray:
        self.history.append(self.build_frame())
        obs = np.concatenate(list(self.history))[None, :]
        self.active = "balance" if float(np.linalg.norm(self.cmd)) < self.switch_norm else "walk"
        raw = self._sess[self.active].run(None, {self._in_name[self.active]: obs})[0][0].astype(np.float32)
        self.last_action = raw
        self.last_raw = raw
        return raw

    def step_policy(self):
        raw = self.infer()
        tgt = self._target
        tgt[0, self.idx15] = torch.as_tensor(raw * self.action_scale + self.default15, dtype=tgt.dtype, device=tgt.device)
        tgt[0, self.idx29[15:]] = torch.as_tensor(np.asarray(self.robot_dds.positions[15:29], dtype=np.float32),
                                                  dtype=tgt.dtype, device=tgt.device)
        if self.gripper_dds.left is None:
            raise RuntimeError("gripper targets not set (call set_gripper_joint_targets before stepping)")
        for side, v in (("left", self.gripper_dds.left), ("right", self.gripper_dds.right)):
            tgt[0, self._grip_idx[side]] = v
        env = self.env
        for _ in range(self.decimation):
            self.robot.set_joint_position_target(tgt)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
        env.sim.render()
        env.observation_manager.compute()
