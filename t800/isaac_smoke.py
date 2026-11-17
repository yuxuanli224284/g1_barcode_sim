"""Isaac smoke run of a robot in a barcode scene: stand, walk / turn / stop, scanner arm to a scan-like pose and back,
head RGB-D + scanner images, metrics JSON, optional video. The same command sequence runs for both robots so a G1 run
of this script on the unmodified code and on the t800 branch can be compared step by step (--traj).

  gpuq run -n 1 --note "HumanScan T800 isaac smoke" --eta 40m -- \
      bash t800/run_isaac.sh -u t800/isaac_smoke.py --robot t800 --video --tag t800
  # G1 identity check: the same script against another code root (e.g. a `git archive` of the branch base)
  ... t800/isaac_smoke.py --robot g1 --code_root /tmp/g1_base --tag g1_base --traj

Outputs (t800/out/): isaac_<tag>_metrics.json, isaac_<tag>_{head_rgb,head_depth,scanner_rgb}.png,
isaac_<tag>.mp4 (--video), isaac_<tag>_traj.npz (--traj: root pose + joint positions every policy step).

Stated test choices (not robot parameters):
  PLAN            phase durations / commands below. Commands lie inside both robots' velocity limits (G1 vx 0.4,
                  wz 1.0; T800 vx 0.5, wz 1.0) and above the T800 start thresholds measured in MuJoCo (vx 0.25,
                  wz 0.5). The route keeps clear of the s1_table table (front face x = 1.30 m).
  SETTLE_S        averages of the short walk phases skip their first 1 s. The phases are 2-3 s long, so these numbers
                  are START-UP TRANSIENTS (MuJoCo, same procedure: ~3 s from standing to the steady speed), not
                  steady-state tracking, and they are reported as metrics["walk_transient"]. Steady-state tracking and
                  start thresholds come from t800/isaac_sim2sim.py vs t800/mujoco_mirror.py (t800/scenarios.py).
  SCAN_POSE       right-arm (T800) / left-arm (G1, scanner side) joint pose; T800 = mujoco_check.SCAN_POSE
                  (wrist ~0.39 m in front of the pelvis at chest height by FK), wrists 0.
  ARM_RAMP_S      joint-space ramp duration (arm.set_joint_targets).
  VIDEO_FPS, VIEW the fixed third-person view (aux camera), behind-left of the start pose.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--robot", choices=("g1", "t800"), required=True)
ap.add_argument("--code_root", default=os.path.dirname(HERE), help="g1_barcode_sim checkout to import")
ap.add_argument("--scene", default="s1_table")
ap.add_argument("--tag", required=True)
ap.add_argument("--video", action="store_true")
ap.add_argument("--traj", action="store_true")
ap.add_argument("--no_override", action="store_true", help="pass no config_overrides (code roots without `model`)")
args = ap.parse_args()
sys.path.insert(0, args.code_root)
sys.path.insert(0, os.path.expanduser("~/random/switchyard/clients"))

import numpy as np  # noqa: E402
from switchyard_progress import Run  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
SETTLE_S = 1.0
ARM_RAMP_S = 1.0
VIDEO_FPS = 25
VIEW_EYE, VIEW_TARGET, VIEW_HFOV = (-1.6, -2.2, 2.3), (0.7, 0.3, 0.8), 70.0
# (label, vx, vy, wz, seconds)
PLAN = [("stand", 0.0, 0.0, 0.0, 10.0), ("forward", 0.4, 0.0, 0.0, 2.0), ("stop", 0.0, 0.0, 0.0, 2.0),
        ("turn_left", 0.0, 0.0, 0.6, 3.0), ("forward2", 0.4, 0.0, 0.0, 2.0), ("stop2", 0.0, 0.0, 0.0, 3.0)]
SCAN_POSE = {"t800": [-0.6, -0.15, 0.0, -1.2, 0.0, 0.0, 0.0],
             "g1": [-0.6, 0.15, 0.0, 0.6, 0.0, 0.0, 0.0]}
ARM_HOLD_S = 4.0
ARM_NOFF_S = 2.0          # T800 only: hold continued with the arm feed-forward switched off (A/B of arm_ff)
ARM_BACK_S = 2.0
PROGRESS_EVERY_S = 0.5

gpus = [int(g) for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
run = Run(f"hbs-t800-smoke-{args.tag}", label=f"HumanoidBarcodeScan T800 · Isaac smoke {args.tag}", gpus=gpus)
run.status("Starting Isaac Sim")
total_s = sum(p[4] for p in PLAN) + ARM_RAMP_S + ARM_HOLD_S + ARM_BACK_S

from g1_barcode_sim.env import G1BarcodeEnv  # noqa: E402

t_wall0 = time.time()
over = None if args.no_override else {"robot": {"model": args.robot}}
env = G1BarcodeEnv(args.scene, headless=True, seed=0, config_overrides=over)
t_build = time.time() - t_wall0
try:
    import cv2
    from g1_barcode_sim.camera import look_at_ros, yaw_of
    from g1_barcode_sim.env import depth_colormap

    robot = env.robot
    side = env.scanner_side
    arm = env.arm
    from g1_barcode_sim.arm import ARM_JOINTS
    arm_names = getattr(arm, "arm_joints", ARM_JOINTS)[side]     # code roots before the T800 change: G1 constant
    log = {k: [] for k in ("t", "pos", "yaw", "v_head", "wz", "tilt", "cmd", "arm_q", "arm_cmd", "q", "phase")}
    state = {"phase": "init", "cmd": (0.0, 0.0, 0.0)}

    def hook():
        d = robot.data
        T = env._T_world_root()
        yaw = yaw_of(T)
        vw = d.root_lin_vel_w[0].cpu().numpy()
        c, s = math.cos(yaw), math.sin(yaw)
        g = d.projected_gravity_b[0].cpu().numpy()
        q = d.joint_pos[0].cpu().numpy()
        log["t"].append(env._sim_time); log["pos"].append(T[:3, 3].copy()); log["yaw"].append(yaw)
        log["v_head"].append([c * vw[0] + s * vw[1], -s * vw[0] + c * vw[1]])
        log["wz"].append(float(d.root_ang_vel_w[0, 2]))
        log["tilt"].append(float(np.degrees(np.arccos(np.clip(-g[2], -1, 1)))))
        log["cmd"].append(state["cmd"]); log["phase"].append(state["phase"])
        log["arm_q"].append(q[arm.joint_idx[side]].copy()); log["arm_cmd"].append(arm.q_cmd[side].copy())
        log["q"].append(q.copy())
        if env._sim_time >= state.get("next_report", 0.0):          # progress file every PROGRESS_EVERY_S sim seconds
            run.progress(round(min(env._sim_time, total_s), 2), total_s, unit="sim s", status=f"Running: {state['phase']}")
            state["next_report"] = env._sim_time + PROGRESS_EVERY_S
    env.step_hooks.append(hook)

    vw = None
    frames = {"n": 0}
    if args.video:
        cam_T = look_at_ros(np.array(VIEW_EYE), np.array(VIEW_TARGET))
        steps_per_frame = max(1, round(1.0 / VIDEO_FPS / env.policy_dt))

    def write_frame(r):
        global vw
        aux = env.render_aux(cam_T, hfov_deg=VIEW_HFOV)["rgb"]
        img = cv2.cvtColor(aux, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        insets = [("head camera", r["obs"]["rgb"])]
        if "wrist_rgb" in r["obs"]:
            insets.append(("scanner camera", r["obs"]["wrist_rgb"]))
        hh, hw = h // 3, int(h // 3 * 4 / 3)
        for i, (name, im) in enumerate(insets):
            x0 = w - (i + 1) * (hw + 10)
            img[h - hh - 10:h - 10, x0:x0 + hw] = cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), (hw, hh))
            cv2.putText(img, name, (x0 + 5, h - hh - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        txt = (f"{args.robot}  t={r['obs']['timestamp']:5.2f}s  {state['phase']}  cmd vx={state['cmd'][0]:+.2f} "
               f"wz={state['cmd'][2]:+.2f}  [{r['status']}]")
        for col, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
            cv2.putText(img, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, th, cv2.LINE_AA)
        if vw is None:
            vw = cv2.VideoWriter(os.path.join(OUT, f"isaac_{args.tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 VIDEO_FPS, (w, h))
        vw.write(img)
        frames["n"] += 1

    statuses = []

    def run_for(label, vx, vy, wz, secs):
        state["phase"], state["cmd"] = label, (vx, vy, wz)
        if args.video:
            n = int(round(secs / (steps_per_frame * env.policy_dt)))
            for _ in range(n):
                r = env.step(vx, vy, wz, duration=steps_per_frame * env.policy_dt)
                write_frame(r)
                statuses.append((label, r["status"], r["reason"]))
                if r["status"] != "ok":
                    return r
            return r
        r = env.step(vx, vy, wz, duration=secs)
        statuses.append((label, r["status"], r["reason"]))
        return r

    t_run0 = time.time()
    run.status("Running")
    pose0 = env._T_world_root()
    phase_bounds = {}
    for label, vx, vy, wz, secs in PLAN:
        t0 = env._sim_time
        r = run_for(label, vx, vy, wz, secs)
        phase_bounds[label] = (t0, env._sim_time)
        print(f"[smoke] {label}: {r['status']} {r['reason']} pose={env._T_world_root()[:3, 3].round(3)}", flush=True)
        if r["status"] == "fell":
            break

    # scanner arm: ramp to the scan pose, hold, ramp back to the default pose
    arm_res = {}
    if statuses[-1][1] != "fell":
        q_def = arm.default_q[side].copy()
        n_ramp = int(round(ARM_RAMP_S / env.policy_dt))
        t0 = env._sim_time
        arm.set_joint_targets(side, np.array(SCAN_POSE[args.robot]), n_ramp)
        run_for("arm_to_scan", 0.0, 0.0, 0.0, ARM_RAMP_S)
        t1 = env._sim_time
        run_for("arm_hold", 0.0, 0.0, 0.0, ARM_HOLD_S)
        t2 = env._sim_time
        obs = env.get_obs()
        if args.robot == "t800":
            ff_saved = env.walker.arm_ff
            env.walker.arm_ff = "none"
            run_for("arm_hold_noff", 0.0, 0.0, 0.0, ARM_NOFF_S)
            env.walker.arm_ff = ff_saved
            phase_bounds["arm_hold_noff"] = (t2, env._sim_time)
            t2b = env._sim_time
        arm.set_joint_targets(side, q_def, n_ramp)
        run_for("arm_back", 0.0, 0.0, 0.0, ARM_BACK_S)
        phase_bounds.update({"arm_to_scan": (t0, t1), "arm_hold": (t1, t2),
                             "arm_back": (t2b if args.robot == "t800" else t2, env._sim_time)})
        tt = np.array(log["t"])
        aq, ac = np.array(log["arm_q"]), np.array(log["arm_cmd"])
        hold = (tt > t1 + 1.0) & (tt <= t2)          # skip the first second after the ramp (settling)
        err = np.abs(aq - ac)
        arm_res = {"joints": list(arm_names), "scan_pose": SCAN_POSE[args.robot],
                   "hold_err_max_rad_per_joint": err[hold].max(0).round(4).tolist(),
                   "hold_err_mean_rad_per_joint": err[hold].mean(0).round(4).tolist(),
                   "hold_err_max_rad": float(err[hold].max()),
                   "ramp_err_max_rad": float(err[(tt > t0) & (tt <= t1)].max()),
                   "back_final_err_max_rad": float(np.abs(aq[-1] - q_def).max()),
                   "body_drift_during_arm_m": float(np.linalg.norm(np.array(log["pos"])[-1][:2] - np.array(log["pos"])[np.searchsorted(tt, t0)][:2])),
                   "wrist_pos_root_at_scan": obs["wrist_pos_root"][side].round(4).tolist()}
        if "arm_hold_noff" in phase_bounds:
            a_, b_ = phase_bounds["arm_hold_noff"]
            nf = (tt > a_ + 1.0) & (tt <= b_)
            arm_res["hold_noff_err_max_rad_per_joint"] = err[nf].max(0).round(4).tolist()
            arm_res["hold_noff_err_max_rad"] = float(err[nf].max())
        if env.scanner_on:
            arm_res["T_root_scanner_tip_at_scan"] = np.round(obs["T_root_scanner_tip"], 4).tolist()
        # images at the scan pose
        cv2.imwrite(os.path.join(OUT, f"isaac_{args.tag}_head_rgb.png"), cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUT, f"isaac_{args.tag}_head_depth.png"),
                    depth_colormap(obs["depth"], env.cfg["camera"]["depth"]["valid_range"]))
        if "wrist_rgb" in obs:
            cv2.imwrite(os.path.join(OUT, f"isaac_{args.tag}_scanner_rgb.png"), cv2.cvtColor(obs["wrist_rgb"], cv2.COLOR_RGB2BGR))
        dep = obs["depth"]
        cam = {"K": np.round(obs["K"], 3).tolist(), "rgb_shape": list(obs["rgb"].shape),
               "depth_valid_frac": float((dep > 0).mean()), "depth_min_max_valid": [float(dep[dep > 0].min()), float(dep[dep > 0].max())] if (dep > 0).any() else None,
               "T_root_cam": np.round(obs["T_root_cam"], 4).tolist(),
               "cam_height_world_m": float((obs["T_world_root_gt"] @ obs["T_root_cam"])[2, 3]),
               "cam_pitch_down_deg": float(np.degrees(np.arcsin(-(obs["T_world_root_gt"] @ obs["T_root_cam"])[2, 2])))}
        if "wrist_rgb" in obs:
            cam["scanner_rgb_shape"] = list(obs["wrist_rgb"].shape)
            cam["scanner_K"] = np.round(obs["wrist_K"], 3).tolist()
            cam["scanner_rgb_mean"] = float(obs["wrist_rgb"].mean())
    else:
        cam = None

    if vw is not None:
        vw.release()
    t_run = time.time() - t_run0

    # ------------------------------------------------------------------ metrics
    tt = np.array(log["t"]); P = np.array(log["pos"]); Y = np.unwrap(np.array(log["yaw"]))
    V = np.array(log["v_head"]); WZ = np.array(log["wz"]); TL = np.array(log["tilt"])

    def win(label, skip=0.0):
        a, b = phase_bounds[label]
        return (tt > a + skip) & (tt <= b)

    m = {"robot": args.robot, "tag": args.tag, "code_root": args.code_root, "scene": args.scene,
         "policy_dt": env.policy_dt, "physics_dt": env.env.physics_dt, "statuses": sorted(set(statuses)),
         "fell": any(s[1] == "fell" for s in statuses), "any_non_ok": [s for s in statuses if s[1] != "ok"][:5],
         "init_root_z": float(P[0, 2]), "min_root_z": float(P[:, 2].min()), "max_tilt_deg": float(TL.max()),
         "wall_build_s": t_build, "wall_run_s": t_run, "sim_s": float(tt[-1]), "realtime_factor": float(tt[-1] / t_run)}
    if "stand" in phase_bounds:
        s = win("stand")
        i0, i1 = np.where(s)[0][[0, -1]]
        m["stand"] = {"drift_xy_m": float(np.linalg.norm(P[i1, :2] - P[i0, :2])),
                      "yaw_change_rad": float(Y[i1] - Y[i0]),
                      "mean_speed_last5s": float(np.linalg.norm(V[s & (tt > phase_bounds['stand'][1] - 5.0)], axis=1).mean()),
                      "root_z_mean": float(P[s, 2].mean()), "max_tilt_deg": float(TL[s].max())}
    m["walk_transient"] = {}
    for label, vx, vy, wz, secs in PLAN:
        if label not in phase_bounds or (vx, vy, wz) == (0.0, 0.0, 0.0):
            continue
        s = win(label, SETTLE_S)
        m["walk_transient"][label] = {"cmd": [vx, vy, wz], "mean_vx_vy_wz": [float(V[s, 0].mean()), float(V[s, 1].mean()), float(WZ[s].mean())],
                            "max_tilt_deg": float(TL[s].max())}
    for label in ("stop", "stop2"):
        if label in phase_bounds:
            s = win(label)
            i0, i1 = np.where(s)[0][[0, -1]]
            m.setdefault("stops", {})[label] = {"dist_after_stop_m": float(np.linalg.norm(P[i1, :2] - P[i0, :2])),
                                                "speed_last_0.5s": float(np.linalg.norm(V[s & (tt > phase_bounds[label][1] - 0.5)], axis=1).mean())}
    if "turn_left" in phase_bounds:
        a, b = phase_bounds["turn_left"]
        m["turn_yaw_change_rad"] = float(Y[np.searchsorted(tt, b) - 1] - Y[np.searchsorted(tt, a)])
    m["final_pose_xy_yaw"] = [float(P[-1, 0]), float(P[-1, 1]), float(Y[-1])]
    if args.robot == "t800":       # held head pitch joint: deviation from its hold angle (tilts the head camera)
        hc = env.cfg["robot"]["head_camera"]
        hold = float(env.cfg["robot"]["t800_walk"]["head_hold"][hc["pitch_joint"]])
        he = np.degrees(np.array(log["q"])[:, list(robot.data.joint_names).index(hc["pitch_joint"])] - hold)
        st = win("stand") & (tt > phase_bounds["stand"][1] - 5.0)
        walking = np.zeros_like(tt, dtype=bool)
        for label, vx, vy, wz, secs in PLAN:
            if label in phase_bounds and (vx, vy, wz) != (0.0, 0.0, 0.0):
                walking |= win(label)
        m["head_pitch_err_deg"] = {"stand_last5s_mean": float(he[st].mean()), "walk_max_abs": float(np.abs(he[walking]).max()),
                                   "run_max_abs": float(np.abs(he).max())}
    m["arm"] = arm_res
    m["camera"] = cam
    if args.robot == "t800" and hasattr(env.walker, "pol"):
        w = env.walker
        m["t800_walk"] = {"arm_owner": w.arm_owner, "arm_obs": w.arm_obs, "arm_ff": w.arm_ff,
                          "command_lpf": bool(w.pol.enable_command_lpf), "decimation": w.decimation,
                          "camera_mount": env.cfg["camera"].get("mount"),
                          "camera_mount_check": env.head_camera_check}
    json.dump(m, open(os.path.join(OUT, f"isaac_{args.tag}_metrics.json"), "w"), indent=1)
    if args.traj:
        np.savez_compressed(os.path.join(OUT, f"isaac_{args.tag}_traj.npz"), t=tt, pos=P, yaw=np.array(log["yaw"]),
                            q=np.array(log["q"]), joint_names=np.array(list(robot.data.joint_names)))
    print("RESULT_JSON " + json.dumps(m), flush=True)
    run.done()
except BaseException as e:
    import traceback
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
    run.failed(f"{type(e).__name__}: {e}")
    os._exit(1)
sys.stdout.flush()
env.close()
