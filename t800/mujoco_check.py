"""Sim2sim check of the EngineAI T800 Pro walk policy (t800/walk_policy.py) in the official MJCF.

Usage (CPU only):
  MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
  /data/yicheng/HumanoidBarcodeScan/venvs/t800/bin/python -I t800/mujoco_check.py [--video] [--probe]
      [--lpf {on,off,both}] [--jobs N]

Command low-pass filter (--lpf): the runner smooths the REMOTE (gamepad) command with a first-order LPF
(runner.cc:315-317, enable_remote_command_lpf walk.yaml:80, listed under "# sim to real fine tune
parameters" walk.yaml:79). Its C++ header is not shipped, so its discretisation is our assumption (see
walk_policy.FirstOrderLPF). Every navigation-relevant number (start threshold for vx, minimum step,
coasting after stop) depends on it, so by default every scenario is run twice, with the LPF on (yaml
default) and off (what a navigation agent that sends velocity commands directly would use), and both
sets are written to the JSON (field "command_lpf") plus a side-by-side table (mujoco_check_lpf_compare.txt).

Control: policy at control_dt = 0.01 s (walk.yaml:77); joint PD torque
  tau = kp (q_des - q) - kd dq   (runner.cc:515-516, gains walk.yaml:52-70)
recomputed at every MuJoCo step (timestep 0.002 s from serial_t800pro.xml:3), clamped by the
MJCF actuator ctrlrange (serial_actuators.xml). Joints outside the 22 policy joints (waist, wrists,
fingers, head) get q_des = default_q with the same yaml gains -- exactly what the runner sends.

Arm modes (task 3, our IK will own the arms):
  policy      : the runner as shipped (policy drives the 10 arm joints)
  ext_real    : the 10 arm joints are driven by an external PD to an external target; the policy sees
                the TRUE arm joint state; its arm actions are discarded
  ext_spoof   : as ext_real, but the policy's arm observation is replaced by a virtual arm that
                follows the policy's own arm command (q_obs = previous arm q_des of the policy,
                dq_obs = its finite difference), i.e. the policy believes it still owns the arms
External arm targets: right arm "scan" pose (see SCAN_POSE; wrist ~0.39 m in front of the pelvis at
chest height by FK), left arm at default; optionally the right arm moves sinusoidally around it.
External arm gains = the yaml arm gains (walk.yaml:56,58), the same PD the runner would use.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from walk_policy import SDK_ROOT, T800WalkPolicy  # noqa: E402
from scenarios import (FALL_HEIGHT_FRAC, FALL_TILT_RAD, MAIN_SCENARIOS, PROBES, SCENARIOS,  # noqa: E402,F401
                       TRACK_SETTLE_S, metrics)

MJCF = os.path.join(SDK_ROOT, "assets/resource/t800pro.xml")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Test scenarios, fall criterion and tracking window: t800/scenarios.py (shared with the Isaac sim2sim run).
# Scanning-like right-arm pose (rad), chosen by FK: upper arm ~34 deg forward, elbow bent ~69 deg.
SCAN_POSE = {"J27_SHOULDER_PITCH_R": -0.6, "J28_SHOULDER_ROLL_R": -0.15, "J29_SHOULDER_YAW_R": 0.0,
             "J30_ELBOW_PITCH_R": -1.2, "J31_ELBOW_YAW_R": 0.0}
# "moving" variant: +-0.3 rad on shoulder pitch and elbow at 0.5 Hz (a slow IK-like reaching motion).
MOVE_AMP, MOVE_HZ = 0.3, 0.5
SEED = 0
VIDEO_FPS = 25          # one frame every 4 control steps
VIDEO_W, VIDEO_H = 640, 480

ARM_JOINTS = ["J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L", "J16_ELBOW_PITCH_L",
              "J17_ELBOW_YAW_L", "J27_SHOULDER_PITCH_R", "J28_SHOULDER_ROLL_R", "J29_SHOULDER_YAW_R",
              "J30_ELBOW_PITCH_R", "J31_ELBOW_YAW_R"]


class Sim:
    def __init__(self, pol: T800WalkPolicy):
        self.m = mujoco.MjModel.from_xml_path(MJCF)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.n_sub = int(round(pol.control_dt / m.opt.timestep))
        if abs(self.n_sub * m.opt.timestep - pol.control_dt) > 1e-9:
            raise ValueError("control_dt is not a multiple of the MJCF timestep")
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in pol.joint_names]
        if min(jid) < 0:
            raise ValueError("joint missing in MJCF")
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.vadr = np.array([m.jnt_dofadr[j] for j in jid])
        self.aid = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_" + n)
                             for n in pol.joint_names])
        if min(self.aid) < 0:
            raise ValueError("actuator missing in MJCF")
        self.s_quat = self._sensor("imu_quaternion")
        self.s_gyro = self._sensor("imu_angular_velocity")
        self.base_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "LINK_BASE")
        foot_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("LINK_FOOT_L", "LINK_FOOT_R")}
        self.foot_geoms = [g for g in range(m.ngeom)
                           if m.geom_bodyid[g] in foot_bodies and m.geom_contype[g] != 0]

    def _sensor(self, name):
        sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr, dim = self.m.sensor_adr[sid], self.m.sensor_dim[sid]
        return slice(adr, adr + dim)

    def reset(self, q0):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[self.qadr] = q0
        d.qpos[2] = 0.0
        mujoco.mj_forward(m, d)
        # put the lowest foot-box corner on the ground (z=0)
        zmin = np.inf
        for g in self.foot_geoms:
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError("foot collision geoms expected to be boxes")
            hs = m.geom_size[g]
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * hs
            z = d.geom_xpos[g][2] + corners @ d.geom_xmat[g].reshape(3, 3)[2]
            zmin = min(zmin, z.min())
        d.qpos[2] = -zmin
        mujoco.mj_forward(m, d)

    def q(self):
        return self.d.qpos[self.qadr].copy()

    def dq(self):
        return self.d.qvel[self.vadr].copy()

    def pd_step(self, q_des, kp, kd, ff_idx=None):
        """ff_idx: joints (indices into the 43) that additionally get the MuJoCo bias force
        (gravity + Coriolis, qfrc_bias) as feed-forward -- used only for the external arm controller."""
        d = self.d
        for _ in range(self.n_sub):
            tau = kp * (q_des - d.qpos[self.qadr]) - kd * d.qvel[self.vadr]
            if ff_idx is not None:
                tau[ff_idx] += d.qfrc_bias[self.vadr[ff_idx]]
            d.ctrl[self.aid] = tau
            mujoco.mj_step(self.m, d)


def yaw_of(quat):
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def arm_target(pol, t, moving):
    tgt = pol.default_q.copy()
    for n, v in SCAN_POSE.items():
        tgt[pol.joint_names.index(n)] = v
    if moving:
        s = MOVE_AMP * np.sin(2 * np.pi * MOVE_HZ * t)
        tgt[pol.joint_names.index("J27_SHOULDER_PITCH_R")] += s
        tgt[pol.joint_names.index("J30_ELBOW_PITCH_R")] -= s
    return tgt


def run(scenario: str, arm_mode: str, moving: bool = False, video: bool = False, arm_gc: bool = False,
        lpf: bool = True):
    np.random.seed(SEED)  # the run is deterministic (no noise injected); seed fixed for completeness
    dur, cmd_fn, track_win = SCENARIOS[scenario]
    pol = T800WalkPolicy(enable_command_lpf=lpf)
    sim = Sim(pol)
    sim.reset(pol.default_q)
    pol.reset()
    arm_idx = np.array([pol.joint_names.index(n) for n in ARM_JOINTS])
    virt_q = pol.default_q[arm_idx].copy()
    virt_dq = np.zeros(len(arm_idx))
    dt = pol.control_dt
    h0 = sim.d.qpos[2]
    p0 = sim.d.qpos[0:3].copy()
    yaw0 = yaw_of(sim.d.qpos[3:7])

    renderer = frames = cam = None
    if video:
        renderer = mujoco.Renderer(sim.m, VIDEO_H, VIDEO_W)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = sim.base_bid
        cam.distance, cam.azimuth, cam.elevation = 3.0, 135.0, -15.0
        frames = []
        frame_every = int(round(1.0 / (VIDEO_FPS * dt)))

    log = {k: [] for k in ("t", "pos", "yaw", "v_head", "wz", "cmd_raw", "cmd_filt", "tilt", "h",
                           "arm_err")}
    fell_at = None
    n_ctrl = int(round(dur / dt))
    for k in range(n_ctrl):
        t = k * dt
        pol.set_command(*cmd_fn(t))
        q, dq = sim.q(), sim.dq()
        w, g = pol.imu_to_obs(sim.d.sensordata[sim.s_quat], sim.d.sensordata[sim.s_gyro])
        q_obs, dq_obs = q.copy(), dq.copy()
        if arm_mode == "ext_spoof":
            q_obs[arm_idx], dq_obs[arm_idx] = virt_q, virt_dq
        out = pol.act(q_obs, dq_obs, w, g)
        q_des, kp, kd = out.q_des.copy(), out.kp, out.kd
        if arm_mode != "policy":
            pol_arm = out.q_des[arm_idx]
            virt_dq = (pol_arm - virt_q) / dt
            virt_q = pol_arm.copy()
            q_des[arm_idx] = arm_target(pol, t, moving)[arm_idx]
        sim.pd_step(q_des, kp, kd, ff_idx=arm_idx if (arm_gc and arm_mode != "policy") else None)

        d = sim.d
        yaw = yaw_of(d.qpos[3:7])
        c, s = np.cos(yaw), np.sin(yaw)
        vw = d.qvel[0:3]
        g_now = pol.imu_to_obs(d.sensordata[sim.s_quat], d.sensordata[sim.s_gyro])[1]
        tilt = float(np.arccos(np.clip(-g_now[2], -1, 1)))
        log["t"].append(t + dt)
        log["pos"].append(d.qpos[0:3].copy())
        log["yaw"].append(yaw)
        log["v_head"].append([c * vw[0] + s * vw[1], -s * vw[0] + c * vw[1]])
        # world-z yaw rate: rotate local angular velocity (freejoint qvel[3:6] is body frame)
        R = d.xmat[sim.base_bid].reshape(3, 3)
        log["wz"].append((R @ d.qvel[3:6])[2])
        log["cmd_raw"].append(pol.command_raw.copy())
        log["cmd_filt"].append(pol.command.copy())
        log["tilt"].append(tilt)
        log["h"].append(d.qpos[2])
        log["arm_err"].append(np.abs(sim.q()[arm_idx] - q_des[arm_idx]).max())
        if video and k % frame_every == 0:
            renderer.update_scene(d, cam)
            frames.append(renderer.render().copy())
        if d.qpos[2] < FALL_HEIGHT_FRAC * h0 or tilt > FALL_TILT_RAD:
            fell_at = t + dt
            break

    L = {k: np.asarray(v) for k, v in log.items()}
    tt = L["t"]
    res = {"arm_mode": arm_mode, "moving_arm": moving, "arm_gc": arm_gc, "command_lpf": bool(lpf)}
    res.update(metrics(scenario, L, p0, yaw0, h0, dt, fell_at))
    if arm_mode != "policy":
        res["arm_track_err_max_rad"] = float(L["arm_err"][int(1.0 / dt):].max()) if len(tt) > 100 else None
    if video:
        import imageio
        os.makedirs(OUT_DIR, exist_ok=True)
        tag = f"{scenario}_{arm_mode}{'_moving' if moving else ''}{'_gc' if arm_gc else ''}{'' if lpf else '_nolpf'}"
        path = os.path.join(OUT_DIR, f"{tag}.mp4")
        imageio.mimwrite(path, frames, fps=VIDEO_FPS, codec="libx264", quality=7)
        res["video"] = path
    return res


def _job(a):
    return run(*a)


# metrics shown side by side (LPF on / off) in mujoco_check_lpf_compare.txt
COMPARE_KEYS = ("mean_meas_vx_vy_wz", "stand_drift_xy_m", "dist_after_stop_m", "mean_speed_last2s",
                "step_disp_forward_m", "final_yaw_change_rad")


def _fmt_line(r):
    line = f"{r['scenario']:22s} {r['arm_mode']:9s} {'mov' if r['moving_arm'] else '   '} " \
           f"{'gc' if r['arm_gc'] else '  '} lpf={'on ' if r['command_lpf'] else 'off'} " \
           f"fell={r['fell']!s:5s}"
    if r["fell"]:
        line += f" at {r['fell_at_s']:.2f}s"
    line += f" minH={r['min_height']:.3f}/{r['h0']:.3f} maxTilt={r['max_tilt_deg']:.1f}deg"
    if "mean_meas_vx_vy_wz" in r:
        line += " meas=" + ",".join(f"{v:+.3f}" for v in r["mean_meas_vx_vy_wz"])
        line += " |err|=" + ",".join(f"{v:.3f}" for v in r["mean_abs_err_vx_vy_wz"])
    for k in ("stand_drift_xy_m", "stand_mean_speed_last5s", "dist_after_stop_m", "mean_speed_last2s",
              "step_disp_forward_m", "step_disp_lateral_m", "mean_speed_last1s", "arm_track_err_max_rad"):
        if k in r and r[k] is not None:
            line += f" {k}={r[k]:.3f}"
    line += f" yaw={r['final_yaw_change_rad']:+.3f}"
    return line


def _fmt_val(v):
    if v is None:
        return "-"
    if isinstance(v, list):
        return "(" + ",".join(f"{x:+.3f}" for x in v) + ")"
    return f"{v:+.3f}"


def _compare_lines(results):
    by = {}
    for r in results:
        if r.get("video"):
            continue
        key = (r["scenario"], r["arm_mode"], r["moving_arm"], r["arm_gc"])
        by.setdefault(key, {})[r["command_lpf"]] = r
    lines = []
    for key, d in by.items():
        if set(d) != {True, False}:
            continue
        on, off = d[True], d[False]
        cells = []
        for k in COMPARE_KEYS:
            if k in on or k in off:
                cells.append(f"{k}: on={_fmt_val(on.get(k))} off={_fmt_val(off.get(k))}")
        fell = f"fell on={on['fell']} off={off['fell']}"
        lines.append(f"{key[0]:22s} {key[1]:9s} {'mov' if key[2] else '   '} {'gc' if key[3] else '  '} "
                     f"{fell} | " + " | ".join(cells))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true",
                    help="render 5 mp4s (vx=0.5 walk policy/ext_spoof/ext_real, stop ext_spoof moving, "
                         "walk ext_spoof moving gc), with the yaml-default command LPF (on)")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--probe", action="store_true", help="also run the command-threshold probes (PROBES)")
    ap.add_argument("--lpf", choices=("on", "off", "both"), default="both",
                    help="command low-pass filter setting(s) to run (default both, see module docstring)")
    args = ap.parse_args()
    lpf_settings = {"on": [True], "off": [False], "both": [True, False]}[args.lpf]
    jobs = []
    for lpf in lpf_settings:
        for sc in MAIN_SCENARIOS:
            for mode in ("policy", "ext_real", "ext_spoof"):
                jobs.append((sc, mode, False, False, False, lpf))
            for mode in ("ext_real", "ext_spoof"):
                jobs.append((sc, mode, True, False, False, lpf))
            # external arm PD + gravity/Coriolis feed-forward (what an IK arm controller would add)
            jobs.append((sc, "ext_spoof", False, False, True, lpf))
            jobs.append((sc, "ext_spoof", True, False, True, lpf))
        if args.probe:
            for name in PROBES:
                jobs.append((name, "policy", False, False, False, lpf))
    if args.video:
        jobs += [("walk_vx0.5", "policy", False, True), ("walk_vx0.5", "ext_spoof", False, True),
                 ("walk_vx0.5", "ext_real", False, True), ("stop_and_stand", "ext_spoof", True, True),
                 ("walk_vx0.5", "ext_spoof", True, True, True)]
    with ProcessPoolExecutor(args.jobs) as ex:
        results = list(ex.map(_job, jobs))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "mujoco_check_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    n_main = sum(1 for r in results if not r.get("video") and r["scenario"] in MAIN_SCENARIOS)
    n_probe = sum(1 for r in results if not r.get("video") and r["scenario"] in PROBES)
    n_video = sum(1 for r in results if r.get("video"))
    header = (f"# runs: {n_main} main + {n_probe} probe + {n_video} video; LPF settings: "
              f"{['on' if x else 'off' for x in lpf_settings]}")
    lines = [header] + [_fmt_line(r) for r in results if not r.get("video")]
    with open(os.path.join(OUT_DIR, "mujoco_check_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if len(lpf_settings) == 2:
        cmp_lines = ["# same scenario with the command LPF on (yaml default) vs off"] + _compare_lines(results)
        with open(os.path.join(OUT_DIR, "mujoco_check_lpf_compare.txt"), "w") as f:
            f.write("\n".join(cmp_lines) + "\n")
        print("\n".join(cmp_lines))


if __name__ == "__main__":
    main()
