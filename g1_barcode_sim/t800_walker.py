"""EngineAI T800 Pro lower-body walker: the vendor RL walk policy (t800/walk_policy.py, a port of the SDK's
rl_walking_example runner validated in MuJoCo) driving the Isaac Lab articulation.

Interface shared with GearWbcWalker / InProcessWholebodyWalker (env.py / arm.py use these):
  set_velocity(vx, vy, wz), step_policy(), reset(), decimation,
  robot_dds.positions[15:29] = arm joint targets (left 7, right 7, order of robot_t800.yaml arm_ik.joints),
  arm_joint_mapping {joint name: index 0..13} (motor index = index + 15, as the Unitree convention arm.py expects).
No body command (height / torso): there is no set_body_command, so env.set_body_pose raises as for the Unitree walker.
No gripper.

Every policy step (control_dt = 10 ms, walk.yaml):
  obs  : q, dq of the 43 joints (model.yaml order), IMU = root link orientation + body angular velocity (the IMU site
         sits on LINK_BASE with identity rotation, imu_install_bias 0). With arm_obs = spoof the 10 policy arm joints
         are replaced by the virtual arm (previous policy arm targets, finite-difference velocity).
  act  : T800WalkPolicy.act -> q_des (43). Legs (and the held waist) take q_des; with arm_owner = external the 14 arm
         joints take the arm command; the head takes head_hold; the fingers take q_des (= default).
  PD   : the actuator model (robot_t800.yaml actuator) with the walk.yaml gains; the external arm joints (arm_ff) and
         the held head joints (head_ff) get the PhysX gravity (+ Coriolis) compensation torque as feed-forward,
         refreshed every physics step. Without head_ff the head pitch joint sags under gravity (Isaac smoke: +0.35 deg
         standing, up to 0.78 deg walking), which tilts the head camera by the same angle.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import torch

from .t800_robot import _sdk, joint_tables

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_walk_policy_module():
    """t800/walk_policy.py (single source, also used by t800/mujoco_check.py)."""
    path = os.path.join(PKG_ROOT, "t800", "walk_policy.py")
    spec = importlib.util.spec_from_file_location("t800_walk_policy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _ArmCommand:
    def __init__(self):
        self.positions = [0.0] * 29


class T800Walker:
    ARM_OWNERS = ("external", "policy")
    ARM_OBS = ("spoof", "real")
    FF_MODES = ("gravity_coriolis", "gravity", "none")

    def __init__(self, env, rcfg: dict):
        self.env = env
        self.robot = env.scene["robot"]
        w = rcfg["t800_walk"]
        for key, allowed in (("arm_owner", self.ARM_OWNERS), ("arm_obs", self.ARM_OBS), ("arm_ff", self.FF_MODES),
                             ("head_ff", self.FF_MODES)):
            if w[key] not in allowed:
                raise ValueError(f"t800_walk.{key} must be one of {allowed}, got {w[key]!r}")
        self.arm_owner, self.arm_obs, self.arm_ff, self.head_ff = w["arm_owner"], w["arm_obs"], w["arm_ff"], w["head_ff"]
        WP = load_walk_policy_module()
        self.pol = WP.T800WalkPolicy(policy_path=w["policy"], cfg_yaml=_sdk(rcfg, w["walk_yaml"]),
                                     model_yaml=_sdk(rcfg, w["model_yaml"]),
                                     enable_command_lpf=bool(w["enable_command_lpf"]))
        pol = self.pol
        ratio = pol.control_dt / env.physics_dt
        self.decimation = int(round(ratio))
        if abs(self.decimation - ratio) > 1e-9:
            raise ValueError(f"policy control_dt {pol.control_dt} is not a multiple of physics_dt {env.physics_dt}")

        names = list(self.robot.data.joint_names)
        missing = [n for n in pol.joint_names if n not in names]
        if missing or len(names) != pol.num_total_joints:
            raise ValueError(f"articulation joints do not match model.yaml: missing {missing}, {len(names)} joints")
        self.joint_to_index = {n: i for i, n in enumerate(names)}
        self.idx43 = np.array([names.index(n) for n in pol.joint_names])      # model order -> Isaac index

        arm = rcfg["arm_ik"]["joints"]
        self.arm14 = list(arm["left"]) + list(arm["right"])
        self.arm_joint_mapping = {n: i for i, n in enumerate(self.arm14)}
        jm = {n: i for i, n in enumerate(pol.joint_names)}                   # name -> model-order index
        self.arm14_m = np.array([jm[n] for n in self.arm14])
        active = set(pol.active_joint_names)
        self.pol_arm_m = np.array([jm[n] for n in pol.active_joint_names if n in set(self.arm14)])   # 10 joints
        if len(self.pol_arm_m) != 10 or not set(pol.joint_names[i] for i in self.pol_arm_m) <= active:
            raise ValueError("expected the 10 policy arm joints (shoulder pitch/roll/yaw, elbow pitch/yaw x 2)")
        T = joint_tables(rcfg)
        self.head_m = np.array([jm[n] for n in w["head_hold"]])
        self.head_q = np.array([float(v) for v in w["head_hold"].values()])
        self.kp_cfg = np.array([T["kp"][n] for n in pol.joint_names])
        self.kd_cfg = np.array([T["kd"][n] for n in pol.joint_names])
        self.ff_idx_arm = torch.as_tensor(self.idx43[self.arm14_m], dtype=torch.long)    # Isaac indices, arm joints
        self.ff_idx_head = torch.as_tensor(self.idx43[self.head_m], dtype=torch.long)    # Isaac indices, held head

        # interface shared with the G1 walkers (arm.py): arm targets at positions[15:29], defaults = walk.yaml pose
        self.robot_dds = _ArmCommand()
        self.robot_dds.positions = [0.0] * 15 + [float(pol.default_q[jm[n]]) for n in self.arm14]
        self.gripper_dds = None
        self._target = self.robot.data.default_joint_pos.clone()
        self._effort = torch.zeros_like(self._target)
        self.last_out = None          # last PolicyOutput (logs / tests)
        self.last_ff = None           # last feed-forward effort target (all joints, Isaac order; zeros elsewhere)
        self.reset()

    # ------------------------------------------------------------------ control surface
    def reset(self):
        """Runner Enter(): history cleared, startup blend from the default pose with the policy gains, command 0.
        The virtual (spoofed) arm restarts at the default pose."""
        self.pol.reset()
        self.virt_q = self.pol.default_q[self.pol_arm_m].copy()
        self.virt_dq = np.zeros(len(self.pol_arm_m))

    def set_velocity(self, vx: float, vy: float, wz: float):
        """Physical command (m/s, m/s, rad/s); raises outside the operator stick range (walk.yaml command_scale)."""
        self.pol.set_command(float(vx), float(vy), float(wz))

    @property
    def cmd(self) -> np.ndarray:
        return self.pol.command_raw.copy()

    # ------------------------------------------------------------------ policy
    def observe(self):
        """(q, dq, ang_vel_b, gravity_b) as the policy sees them (model order; arm spoofed if configured)."""
        data = self.robot.data
        q = data.joint_pos[0].cpu().numpy().astype(np.float64)[self.idx43]
        dq = data.joint_vel[0].cpu().numpy().astype(np.float64)[self.idx43]
        w, g = self.pol.imu_to_obs(data.root_link_quat_w[0].cpu().numpy(), data.root_link_ang_vel_b[0].cpu().numpy())
        if self.arm_owner == "external" and self.arm_obs == "spoof":
            q[self.pol_arm_m], dq[self.pol_arm_m] = self.virt_q, self.virt_dq
        return q, dq, w, g

    def infer(self) -> np.ndarray:
        """One policy step; returns the 43 joint targets in model order."""
        pol = self.pol
        out = pol.act(*self.observe())
        # reset() starts the blend from the policy gains, so kp / kd equal walk.yaml gains = the actuator gains
        if np.abs(out.kp - self.kp_cfg).max() > 1e-6 or np.abs(out.kd - self.kd_cfg).max() > 1e-6:
            raise RuntimeError("policy gains differ from the actuator gains (startup blend from other gains?)")
        self.last_out = out
        q_des = out.q_des.copy()
        pol_arm = out.q_des[self.pol_arm_m]
        self.virt_dq = (pol_arm - self.virt_q) / pol.control_dt
        self.virt_q = pol_arm.copy()
        if self.arm_owner == "external":
            q_des[self.arm14_m] = np.asarray(self.robot_dds.positions[15:29], dtype=np.float64)
        q_des[self.head_m] = self.head_q
        return q_des

    def _ff_groups(self):
        """(Isaac joint indices, mode) of the joints that get a feed-forward torque now (arm_ff / head_ff may be changed
        at run time, e.g. the smoke A/B of arm_ff)."""
        groups = [(self.ff_idx_head, self.head_ff)]
        if self.arm_owner == "external":
            groups.append((self.ff_idx_arm, self.arm_ff))
        return [(i, m) for i, m in groups if m != "none"]

    def _feedforward(self) -> torch.Tensor | None:
        groups = self._ff_groups()
        if not groups:
            return None
        view = self.robot.root_physx_view
        # floating base: generalized forces are (6 root + n_dof); joint j is column j + 6 (as the jacobian)
        g = torch.as_tensor(view.get_gravity_compensation_forces()[0, 6:], dtype=self._effort.dtype,
                            device=self._effort.device)
        c = None
        if any(m == "gravity_coriolis" for _, m in groups):
            c = torch.as_tensor(view.get_coriolis_and_centrifugal_compensation_forces()[0, 6:],
                                dtype=self._effort.dtype, device=self._effort.device)
        eff = torch.zeros_like(self._effort[0])
        for idx, mode in groups:
            eff[idx] = g[idx] + c[idx] if mode == "gravity_coriolis" else g[idx]
        return eff

    def step_policy(self):
        q_des = self.infer()
        tgt = self._target
        tgt[0, torch.as_tensor(self.idx43)] = torch.as_tensor(q_des, dtype=tgt.dtype, device=tgt.device)
        env = self.env
        for _ in range(self.decimation):
            self.robot.set_joint_position_target(tgt)
            ff = self._feedforward()
            if ff is not None:
                self._effort[0] = ff
                self.robot.set_joint_effort_target(self._effort)
                self.last_ff = ff.cpu().numpy()
            elif self.last_ff is not None:
                # feed-forward switched off at run time: the articulation keeps its last effort target, clear it once
                self._effort.zero_()
                self.robot.set_joint_effort_target(self._effort)
                self.last_ff = None
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
        env.sim.render()
        env.observation_manager.compute()
