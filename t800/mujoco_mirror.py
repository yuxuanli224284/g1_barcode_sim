"""MuJoCo counterpart of the Isaac T800 walker for the sim2sim comparison: the SAME scenarios and metrics
(t800/scenarios.py) and the SAME control law as g1_barcode_sim/t800_walker.T800Walker, configured from the same
configs/robot_t800.yaml t800_walk switches, in the vendor MJCF (t800/mujoco_check.Sim):

  policy      robot_t800.yaml t800_walk.policy (the .npz the Isaac walker loads), enable_command_lpf
  arms        arm_owner external: the 14 arm joints (arm_ik.joints, incl. wrists) take the arm command = the policy
              default pose (what arm.py holds after a reset); arm_obs spoof: the 10 policy arm joints are observed as
              the policy's own last arm targets and their finite difference
  head        head_hold targets
  PD          walk.yaml gains (policy output kp / kd), every physics step; feed-forward = MuJoCo qfrc_bias (gravity +
              Coriolis) on the external arm joints (arm_ff) and the held head joints (head_ff)
  start       feet on the ground (mujoco_check.Sim.reset), yaw 0 (metrics are in the start heading frame)

Differences that remain by construction (they are what the comparison measures): contact geometry (MJCF: 4 boxes
per foot; Isaac: convex hull of the URDF LINK_ANKLE_ROLL mesh), solver, torque limits (MJCF ctrlrange vs URDF effort).

  MUJOCO_GL=egl /data/yicheng/HumanoidBarcodeScan/venvs/t800/bin/python -I t800/mujoco_mirror.py [--jobs N]
Output: t800/out/mujoco_mirror.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

# one BLAS thread per worker process: the scenarios run in parallel processes (without this each process spawns a
# thread per core and the pool oversubscribes the CPU ~100x); results do not depend on it
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mujoco_check import ARM_JOINTS, Sim, yaw_of  # noqa: E402
from scenarios import ALL, FALL_HEIGHT_FRAC, FALL_TILT_RAD, SIM2SIM_SCENARIOS, metrics  # noqa: E402
from walk_policy import T800WalkPolicy  # noqa: E402

RCFG_PATH = os.path.join(os.path.dirname(HERE), "configs", "robot_t800.yaml")


def load_cfg():
    with open(RCFG_PATH) as f:
        rcfg = yaml.safe_load(f)
    w = rcfg["t800_walk"]
    for key in ("arm_ff", "head_ff"):
        if w[key] not in ("gravity_coriolis", "none"):
            # MuJoCo's qfrc_bias is gravity + Coriolis together; a gravity-only term would need its own computation
            raise NotImplementedError(f"mirror supports {key} gravity_coriolis | none, config has {w[key]!r}")
    return rcfg


def run(scenario: str) -> dict:
    rcfg = load_cfg()
    w = rcfg["t800_walk"]
    sdk = rcfg["asset"]["sdk_root"]
    pol = T800WalkPolicy(policy_path=w["policy"], cfg_yaml=os.path.join(sdk, w["walk_yaml"]),
                         model_yaml=os.path.join(sdk, w["model_yaml"]), enable_command_lpf=bool(w["enable_command_lpf"]))
    jm = {n: i for i, n in enumerate(pol.joint_names)}
    arm14 = np.array([jm[n] for n in list(rcfg["arm_ik"]["joints"]["left"]) + list(rcfg["arm_ik"]["joints"]["right"])])
    pol_arm = np.array([jm[n] for n in pol.active_joint_names if n in set(pol.joint_names[i] for i in arm14)])
    if sorted(pol.joint_names[i] for i in pol_arm) != sorted(ARM_JOINTS):
        raise ValueError("policy arm joints differ from mujoco_check.ARM_JOINTS")
    head = np.array([jm[n] for n in w["head_hold"]])
    head_q = np.array([float(v) for v in w["head_hold"].values()])
    pitch_j = jm[rcfg["head_camera"]["pitch_joint"]]
    pitch_hold = float(w["head_hold"][rcfg["head_camera"]["pitch_joint"]])
    external = w["arm_owner"] == "external"
    if w["arm_owner"] not in ("external", "policy") or w["arm_obs"] not in ("spoof", "real"):
        raise ValueError(f"arm_owner / arm_obs {w['arm_owner']!r} / {w['arm_obs']!r}")
    ff = []
    if external and w["arm_ff"] != "none":
        ff += list(arm14)
    if w["head_ff"] != "none":
        ff += list(head)
    ff_idx = np.array(ff, dtype=int) if ff else None

    dur, cmd_fn, _ = ALL[scenario]
    sim = Sim(pol)
    sim.reset(pol.default_q)
    pol.reset()
    virt_q, virt_dq = pol.default_q[pol_arm].copy(), np.zeros(len(pol_arm))
    arm_cmd = pol.default_q[arm14].copy()
    dt = pol.control_dt
    h0, p0, yaw0 = sim.d.qpos[2], sim.d.qpos[0:3].copy(), yaw_of(sim.d.qpos[3:7])
    log = {k: [] for k in ("t", "pos", "yaw", "v_head", "wz", "cmd_raw", "cmd_filt", "tilt", "h", "head_err")}
    fell_at = None
    for k in range(int(round(dur / dt))):
        t = k * dt
        pol.set_command(*cmd_fn(t))
        q, dq = sim.q(), sim.dq()
        wv, g = pol.imu_to_obs(sim.d.sensordata[sim.s_quat], sim.d.sensordata[sim.s_gyro])
        if external and w["arm_obs"] == "spoof":
            q[pol_arm], dq[pol_arm] = virt_q, virt_dq
        out = pol.act(q, dq, wv, g)
        q_des = out.q_des.copy()
        virt_dq = (out.q_des[pol_arm] - virt_q) / dt
        virt_q = out.q_des[pol_arm].copy()
        if external:
            q_des[arm14] = arm_cmd
        q_des[head] = head_q
        sim.pd_step(q_des, out.kp, out.kd, ff_idx=ff_idx)

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
        log["wz"].append((d.xmat[sim.base_bid].reshape(3, 3) @ d.qvel[3:6])[2])
        log["cmd_raw"].append(pol.command_raw.copy())
        log["cmd_filt"].append(pol.command.copy())
        log["tilt"].append(tilt)
        log["h"].append(d.qpos[2])
        log["head_err"].append(sim.q()[pitch_j] - pitch_hold)
        if d.qpos[2] < FALL_HEIGHT_FRAC * h0 or tilt > FALL_TILT_RAD:
            fell_at = t + dt
            break
    L = {k: np.asarray(v) for k, v in log.items()}
    res = {"sim": "mujoco"}
    res.update(metrics(scenario, L, p0, yaw0, h0, dt, fell_at))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--only", nargs="*", default=None, help="scenario names (default: scenarios.SIM2SIM_SCENARIOS)")
    args = ap.parse_args()
    names = args.only or SIM2SIM_SCENARIOS
    rcfg = load_cfg()
    with ProcessPoolExecutor(args.jobs) as ex:
        results = list(ex.map(run, names))
    out = {"config": {k: rcfg["t800_walk"][k] for k in ("policy", "enable_command_lpf", "arm_owner", "arm_obs", "arm_ff",
                                                         "head_ff", "head_hold")},
           "results": results}
    path = os.path.join(HERE, "out", "mujoco_mirror.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    for r in results:
        print(r["scenario"], "fell" if r["fell"] else "ok", r.get("mean_meas_vx_vy_wz"), r.get("stand_drift_xy_m"),
              r.get("dist_after_stop_m"), r.get("head_pitch_err_deg_mean_max"))
    print("wrote", path)


if __name__ == "__main__":
    main()
