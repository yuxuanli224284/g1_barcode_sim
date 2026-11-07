"""Wrist position control for the G1 arms (damped-least-squares differential IK, position only).

Each arm has 7 joints (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw). The controlled point is
the `<side>_hand_base_link` origin (the gripper base, i.e. the wrist), expressed in the root (pelvis)
frame, so the target stays attached to the robot while it walks. Orientation is not constrained
(3-D task, 7 DoF), the redundancy is resolved by the minimum-norm DLS step.

Every policy step: e = p_target - p_now (root frame), J = d p / d q_arm (root frame, from PhysX),
dq = gain * J^T (J J^T + lambda^2 I)^-1 e, clipped per joint, added to the arm joint targets, then clamped to
the joint limits.

Scanner aim mode (`set_aim`, the hand holding the scanner): the task is the scanner tip position (3) plus the
scan direction (2: the tip x axis must point along a target direction; roll about it is left free), 5 of 7 DoF.
Rows: J_tip = J_v - [r]x J_w (tip = link origin + r), J_dir = (I - a a^T) J_w (rotation about the scan axis a
projected out), error e_dir = axis * angle of the rotation taking a onto the target direction. The direction rows
and error are scaled by `aim_weight` (m/rad) so that an angle error counts like the aim-point displacement it
causes at the nominal scan distance. The arm targets reach the robot through the walker's arm-command stand-in (the same
path upstream uses for DDS lowcmd arm commands).
"""
from __future__ import annotations

import numpy as np

ARM_JOINTS = {
    side: [f"{side}_shoulder_pitch_joint", f"{side}_shoulder_roll_joint", f"{side}_shoulder_yaw_joint",
           f"{side}_elbow_joint", f"{side}_wrist_roll_joint", f"{side}_wrist_pitch_joint", f"{side}_wrist_yaw_joint"]
    for side in ("left", "right")
}


class WristIK:
    def __init__(self, env, cfg: dict):
        """env: G1BarcodeEnv; cfg: robot.yaml `arm_ik`."""
        self.env, self.cfg = env, cfg
        robot = env.robot
        names = list(robot.data.joint_names)
        self.joint_idx = {s: [names.index(j) for j in ARM_JOINTS[s]] for s in ARM_JOINTS}
        self.body_idx = {s: env.body_names.index(cfg["ee_link"].format(side=s)) for s in ARM_JOINTS}
        if robot.is_fixed_base:
            raise RuntimeError("expected a floating-base G1")
        # floating base: PhysX jacobian columns 0..5 are the base, joint j is column j + 6; body index unchanged
        self.jac_cols = {s: [j + 6 for j in self.joint_idx[s]] for s in ARM_JOINTS}
        lim = robot.data.soft_joint_pos_limits[0].cpu().numpy()
        self.limits = {s: lim[self.joint_idx[s]] for s in ARM_JOINTS}
        # motor index (0..28) of each arm joint in the walker's arm command (upstream: arm mapping idx + 15)
        amap = env.walker.arm_joint_mapping
        self.motor_idx = {s: [amap[j] + 15 for j in ARM_JOINTS[s]] for s in ARM_JOINTS}
        self.default_q = {s: np.array([env.walker.robot_dds.positions[m] for m in self.motor_idx[s]]) for s in ARM_JOINTS}
        self.home = None   # measured by measure_home() once the arms have settled at the default pose
        self.reset()

    def reset(self):
        """Arms back to the default joint pose, IK targets cleared."""
        for s in ARM_JOINTS:
            self._write_q(s, self.default_q[s])
        self.q_cmd = {s: self.default_q[s].copy() for s in ARM_JOINTS}
        self.target = {s: None for s in ARM_JOINTS}      # wrist position target; None = no wrist IK
        self.aim = {s: None for s in ARM_JOINTS}         # scanner aim target {"pos", "dir"} (root frame); None = off
        self.traj = {s: None for s in ARM_JOINTS}        # joint-space trajectory (set_joint_targets); None = off

    def _write_q(self, side, q):
        pos = self.env.walker.robot_dds.positions
        for m, v in zip(self.motor_idx[side], q):
            pos[m] = float(v)

    def wrist_pos_root(self, side) -> np.ndarray:
        from .camera import inv_T, make_T
        p = self.env.robot.data.body_link_pose_w[0, self.body_idx[side]].cpu().numpy()
        T_rw = inv_T(self.env._T_world_root())
        return (T_rw @ make_T(p[:3], p[3:]))[:3, 3]

    def measure_home(self, step_fn, max_s: float):
        """Wrist position of the default arm pose (root frame). The robot spawns in a different (transient) arm
        pose, so hold the default joint targets until every arm joint is within settle_tol of them, then
        measure. Constant afterwards (waist is held at default). Raises if the arms do not settle."""
        tol = self.cfg["settle_tol"]
        n = int(round(max_s / self.env.policy_dt))
        for _ in range(n):
            q = self.env.robot.data.joint_pos[0].cpu().numpy()
            if all(np.max(np.abs(q[self.joint_idx[s]] - self.default_q[s])) < tol for s in ARM_JOINTS):
                self.home = {s: self.wrist_pos_root(s) for s in ARM_JOINTS}
                return self.home
            step_fn()
        raise RuntimeError(f"arms did not settle to the default pose within {max_s} s (tol {tol} rad)")

    def home_pos(self, side) -> np.ndarray:
        """Wrist position (root frame) of the default arm pose."""
        return self.home[side].copy()

    def set_target(self, side, pos_root):
        """Absolute wrist target in the root frame, clamped to `home +- workspace` (robot.yaml)."""
        self._take_over(side)
        self.aim[side] = None
        h = self.home_pos(side)
        w = np.asarray(self.cfg["workspace_half_extent"], dtype=float)
        self.target[side] = np.clip(np.asarray(pos_root, dtype=float), h - w, h + w)
        return self.target[side].copy()

    def set_joint_targets(self, side, q, n_steps: int):
        """Joint-space command (RobotInterface.set_arm_joints): ramp the joint targets linearly from the current
        command to q (clamped to the soft limits) over n_steps policy steps, then hold q. Ends any IK target."""
        q = np.asarray(q, dtype=float)
        if q.shape != (len(ARM_JOINTS[side]),):
            raise ValueError(f"{side} arm needs {len(ARM_JOINTS[side])} joint values, got shape {q.shape}")
        lo, hi = self.limits[side][:, 0], self.limits[side][:, 1]
        self.target[side], self.aim[side] = None, None
        self.traj[side] = {"q0": self.q_cmd[side].copy(), "q1": np.clip(q, lo, hi), "n": max(1, int(n_steps)), "k": 0}
        return self.traj[side]["q1"].copy()

    def _take_over(self, side):
        self.traj[side] = None
        if self.target[side] is None and self.aim[side] is None:
            # IK takes over: integrate from where the arm actually is, not from the last joint command. After
            # reset_arms() the command is the default pose while the arm may still be on its way there; starting
            # from that command makes every step integrate the in-transit error (windup) and the arm can end
            # up 10-30 cm off target after 2 s (measured).
            self.q_cmd[side] = self.env.robot.data.joint_pos[0, self.joint_idx[side]].cpu().numpy().copy()

    def set_aim(self, side, tip_pos_root, dir_root, T_hand_tip, anchor: str = "odom"):
        """Scanner tip target position + scan direction, given in the current root frame. T_hand_tip: tip pose in
        the hand link. anchor="odom": the target is fixed in the odometry frame (converted with the current odom
        pose every step), so it stays on the label when the base drifts; anchor="root": it moves with the body.
        Not clamped to the wrist workspace box: an unreachable target leaves the arm at the closest reachable
        pose (damped least squares); the residual is reported by aim_error()."""
        d = np.asarray(dir_root, dtype=float)
        if np.linalg.norm(d) < 1e-9:
            raise ValueError("scan direction must be non-zero")
        if anchor not in ("odom", "root"):
            raise ValueError(f"anchor must be odom|root, got {anchor!r}")
        self._take_over(side)
        self.target[side] = None
        p, d = np.asarray(tip_pos_root, dtype=float), d / np.linalg.norm(d)
        if anchor == "odom":
            T = self.env._odom_T                       # T_odom_root at the time of the call
            p, d = T[:3, :3] @ p + T[:3, 3], T[:3, :3] @ d
        self.aim[side] = {"pos": p, "dir": d, "anchor": anchor, "T_hand_tip": np.asarray(T_hand_tip, dtype=float)}
        return self.aim_target_root(side)

    def aim_target_root(self, side) -> dict | None:
        """Aim target {"pos", "dir"} in the current root frame (odom-anchored targets re-expressed every call)."""
        a = self.aim[side]
        if a is None:
            return None
        if a["anchor"] == "root":
            return {"pos": a["pos"].copy(), "dir": a["dir"].copy()}
        from .camera import inv_T
        T = inv_T(self.env._odom_T)                    # T_root_odom now
        return {"pos": T[:3, :3] @ a["pos"] + T[:3, 3], "dir": T[:3, :3] @ a["dir"]}

    def tip_pose_root(self, side, T_hand_tip) -> np.ndarray:
        from .camera import inv_T, make_T
        p = self.env.robot.data.body_link_pose_w[0, self.body_idx[side]].cpu().numpy()
        return inv_T(self.env._T_world_root()) @ make_T(p[:3], p[3:]) @ T_hand_tip

    def update(self):
        """One DLS step for every arm that has a target. Called once per policy step."""
        for s in ARM_JOINTS:                               # joint-space trajectories
            t = self.traj[s]
            if t is not None:
                t["k"] = min(t["k"] + 1, t["n"])
                self.q_cmd[s] = t["q0"] + (t["q1"] - t["q0"]) * (t["k"] / t["n"])
                self._write_q(s, self.q_cmd[s])
        active = [s for s in ARM_JOINTS if self.target[s] is not None or self.aim[s] is not None]
        if not active:
            return
        R_wr = self.env._T_world_root()[:3, :3]
        J_all = self.env.robot.root_physx_view.get_jacobians()
        lam2 = self.cfg["dls_lambda"] ** 2
        for s in active:
            Jfull = J_all[0, self.body_idx[s], :, self.jac_cols[s]].cpu().numpy()   # world frame, rows [lin; ang]
            Jv, Jw = R_wr.T @ Jfull[:3], R_wr.T @ Jfull[3:]                           # -> root frame
            if self.aim[s] is None:
                e, J = self.target[s] - self.wrist_pos_root(s), Jv
            else:
                e, J = self._aim_task(s, Jv, Jw)
            # gain < 1: correct only part of the error per policy step; the arm PD needs several steps to follow
            # a new target, so a full correction every step keeps integrating ahead of the arm and diverges
            dq = self.cfg["gain"] * (J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(len(e)), e))
            dq = np.clip(dq, -self.cfg["max_dq_per_step"], self.cfg["max_dq_per_step"])
            lo, hi = self.limits[s][:, 0], self.limits[s][:, 1]
            self.q_cmd[s] = np.clip(self.q_cmd[s] + dq, lo, hi)
            self._write_q(s, self.q_cmd[s])

    def _aim_task(self, s, Jv, Jw):
        a_cfg = self.aim[s]
        tgt = self.aim_target_root(s)
        T = self.tip_pose_root(s, a_cfg["T_hand_tip"])
        p_hand = self.wrist_pos_root(s)
        r = T[:3, 3] - p_hand
        a = T[:3, 0]                                                    # current scan direction
        skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        J_tip = Jv - skew @ Jw                                          # v_tip = v_hand + w x r
        P = np.eye(3) - np.outer(a, a)                                  # drop roll about the scan axis
        e_rot = rotation_vector(a, tgt["dir"])
        w = self.cfg["aim_weight"]
        J = np.vstack([J_tip, w * (P @ Jw)])
        e = np.concatenate([tgt["pos"] - T[:3, 3], w * (P @ e_rot)])
        return e, J

    def aim_error(self, side) -> dict | None:
        """{"pos_m", "angle_deg"} of the scanner tip w.r.t. its aim target, or None when aim mode is off."""
        if self.aim[side] is None:
            return None
        tgt = self.aim_target_root(side)
        T = self.tip_pose_root(side, self.aim[side]["T_hand_tip"])
        ang = np.degrees(np.arctan2(np.linalg.norm(np.cross(T[:3, 0], tgt["dir"])), T[:3, 0] @ tgt["dir"]))
        return {"pos_m": float(np.linalg.norm(tgt["pos"] - T[:3, 3])), "angle_deg": float(ang)}

    def error(self, side) -> float | None:
        if self.target[side] is None:
            return None
        return float(np.linalg.norm(self.target[side] - self.wrist_pos_root(side)))


def rotation_vector(a, b) -> np.ndarray:
    """Axis * angle of the shortest rotation taking unit vector a onto unit vector b (any perpendicular axis if
    they are opposite)."""
    c = np.cross(a, b)
    sn, cs = np.linalg.norm(c), float(a @ b)
    ang = np.arctan2(sn, cs)
    if sn < 1e-9:
        if cs > 0:
            return np.zeros(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        return axis / np.linalg.norm(axis) * ang
    return c / sn * ang
