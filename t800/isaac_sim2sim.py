"""Isaac side of the T800 sim2sim comparison: the scenarios and metrics of t800/scenarios.py (the same procedure as
t800/mujoco_mirror.py, whose control law mirrors this walker) run on the real Isaac path, G1BarcodeEnv(model t800) and
its T800Walker with the robot_t800.yaml switches, arms held at the default pose by arm.py.

  gpuq run -n 1 --note "HumanScan T800 isaac sim2sim" --eta 40m -- bash t800/run_isaac.sh -u t800/isaac_sim2sim.py
Output: t800/out/isaac_sim2sim.json; compare with python3 t800/sim2sim_compare.py.

Per scenario the robot is teleported (env.teleport_root: default joints, zero velocity, walker / policy history and
arm reset, i.e. the runner Enter() state) to (START_X, 0) at scenarios.start_yaw, then every control step:
walker.set_velocity(command(t)) and env._step_once() (arm update + policy step + 5 physics steps). The command goes to
the walker directly, as mujoco_check / mujoco_mirror call policy.set_command (no env velocity clipping, no env status
check: the fall criterion is scenarios.FALL_*).

Stated test choice:
  START_X = -4.0 m   s1_table has its only obstacle (table + boxes) at x 1.30..1.90, |y| <= 0.6; the start yaw sends
                     every scenario's translation towards -x (<= 0.7 m/s x 12 s = 8.4 m -> x >= -12.4), on the 60 x 60 m
                     ground collider (scene_builder; the warehouse art has physics disabled).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.expanduser("~/random/switchyard/clients"))
ap = argparse.ArgumentParser()
ap.add_argument("--only", nargs="*", default=None, help="scenario names (default: scenarios.SIM2SIM_SCENARIOS); the "
                "results are merged into an existing out/isaac_sim2sim.json")
args = ap.parse_args()

import numpy as np  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("t800_scenarios", os.path.join(HERE, "scenarios.py"))  # by path: no
S = importlib.util.module_from_spec(_spec)                                           # sys.path entry for t800/
_spec.loader.exec_module(S)
ALL, FALL_HEIGHT_FRAC, FALL_TILT_RAD, SIM2SIM_SCENARIOS, metrics, start_yaw = (
    S.ALL, S.FALL_HEIGHT_FRAC, S.FALL_TILT_RAD, S.SIM2SIM_SCENARIOS, S.metrics, S.start_yaw)
from switchyard_progress import Run  # noqa: E402

START_X = -4.0
names = args.only or SIM2SIM_SCENARIOS
total_s = sum(ALL[n][0] for n in names)
gpus = [int(g) for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
run = Run("hbs-t800-sim2sim", label="HumanoidBarcodeScan T800 · Isaac sim2sim scenarios", gpus=gpus)
run.status("Starting Isaac Sim")

from g1_barcode_sim.env import G1BarcodeEnv  # noqa: E402

env = G1BarcodeEnv("s1_table", headless=True, seed=0, config_overrides={"robot": {"model": "t800"}})
try:
    robot, walker = env.robot, env.walker
    pol = walker.pol
    rcfg = env.cfg["robot"]
    pitch_name = rcfg["head_camera"]["pitch_joint"]
    pitch_j = list(robot.data.joint_names).index(pitch_name)
    pitch_hold = float(rcfg["t800_walk"]["head_hold"][pitch_name])
    dt = env.policy_dt
    if abs(dt - pol.control_dt) > 1e-12:
        raise RuntimeError(f"env policy_dt {dt} != walk policy control_dt {pol.control_dt}")
    from g1_barcode_sim.camera import yaw_of

    def state():
        d = robot.data
        p = d.root_link_pos_w[0].cpu().numpy().astype(np.float64)
        T = env._T_world_root()
        yaw = yaw_of(T)
        vw = d.root_link_lin_vel_w[0].cpu().numpy().astype(np.float64)
        c, s = math.cos(yaw), math.sin(yaw)
        g = d.projected_gravity_b[0].cpu().numpy().astype(np.float64)
        return {"pos": p, "yaw": yaw, "v_head": [c * vw[0] + s * vw[1], -s * vw[0] + c * vw[1]],
                "wz": float(d.root_link_ang_vel_w[0, 2]), "tilt": float(np.arccos(np.clip(-g[2], -1, 1))),
                "h": float(p[2]), "head_err": float(d.joint_pos[0, pitch_j]) - pitch_hold}

    results = []
    done_s = 0.0
    t_all = time.time()
    run.status("Running")
    for name in names:
        dur, cmd_fn, _ = ALL[name]
        y0 = start_yaw(name)
        env.teleport_root(START_X, 0.0, y0)
        s0 = state()
        p0, yaw0, h0 = s0["pos"].copy(), s0["yaw"], s0["h"]
        log = {k: [] for k in ("t", "pos", "yaw", "v_head", "wz", "cmd_raw", "cmd_filt", "tilt", "h", "head_err")}
        fell_at = None
        t_wall = time.time()
        n = int(round(dur / dt))
        for k in range(n):
            t = k * dt
            walker.set_velocity(*cmd_fn(t))
            env._step_once()
            st = state()
            log["t"].append(t + dt)
            for key in ("pos", "yaw", "v_head", "wz", "tilt", "h", "head_err"):
                log[key].append(st[key])
            log["cmd_raw"].append(pol.command_raw.copy())
            log["cmd_filt"].append(pol.command.copy())
            if st["h"] < FALL_HEIGHT_FRAC * h0 or st["tilt"] > FALL_TILT_RAD:
                fell_at = t + dt
                break
            if k % 100 == 0:
                run.progress(round(done_s + t, 1), total_s, unit="sim s", status=f"Running: {name}")
        L = {k: np.asarray(v) for k, v in log.items()}
        res = {"sim": "isaac", "start_xy_yaw": [START_X, 0.0, y0]}
        res.update(metrics(name, L, p0, yaw0, h0, dt, fell_at))
        res["wall_s"] = time.time() - t_wall
        results.append(res)
        done_s += dur
        print(f"[sim2sim] {name}: {'FELL' if fell_at else 'ok'} meas={res.get('mean_meas_vx_vy_wz')} "
              f"drift={res.get('stand_drift_xy_m')} stop={res.get('dist_after_stop_m')} "
              f"head={res.get('head_pitch_err_deg_mean_max')} wall={res['wall_s']:.0f}s", flush=True)
    w = rcfg["t800_walk"]
    out = {"config": {k: w[k] for k in ("policy", "enable_command_lpf", "arm_owner", "arm_obs", "arm_ff", "head_ff",
                                        "head_hold", "self_collision")},
           "physics_dt": env.env.physics_dt, "policy_dt": dt, "wall_total_s": time.time() - t_all, "results": results}
    path = os.path.join(HERE, "out", "isaac_sim2sim.json")
    if args.only and os.path.exists(path):
        # partial run: replace / add these scenarios in the existing file, which must have the same switches
        with open(path) as f:
            prev = json.load(f)
        if prev["config"] != out["config"]:
            raise RuntimeError(f"--only run config {out['config']} differs from {path} config {prev['config']}")
        new = {r["scenario"] for r in results}
        out["results"] = [r for r in prev["results"] if r["scenario"] not in new] + results
        out["wall_total_s"] = prev["wall_total_s"] + out["wall_total_s"]
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("RESULT_JSON " + json.dumps({"n": len(results), "fell": [r["scenario"] for r in results if r["fell"]],
                                       "wall_total_s": out["wall_total_s"]}), flush=True)
    run.done()
except BaseException as e:
    import traceback
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
    run.failed(f"{type(e).__name__}: {e}")
    os._exit(1)
sys.stdout.flush()
env.close()
