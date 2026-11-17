"""Side-by-side table of the T800 sim2sim runs (same scenarios / metrics / control law, t800/scenarios.py):
t800/out/mujoco_mirror.json (MuJoCo, vendor MJCF) vs t800/out/isaac_sim2sim.json (Isaac Lab, converted URDF).
Raw numbers only (no pass / fail thresholds).  python3 t800/sim2sim_compare.py > t800/out/sim2sim_compare.txt"""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


mj, isa = load("mujoco_mirror.json"), load("isaac_sim2sim.json")
print("# config mujoco:", json.dumps(mj["config"]))
print("# config isaac :", json.dumps(isa["config"]))
if mj["config"] != {k: isa["config"][k] for k in mj["config"]}:
    print("# WARNING: the two runs used different t800_walk switches")
M = {r["scenario"]: r for r in mj["results"]}
I = {r["scenario"]: r for r in isa["results"]}


def f(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(f"{x:+.3f}" for x in v) + "]"
    return f"{v:+.3f}"


KEYS = [("mean_meas_vx_vy_wz", "meas vx,vy,wz (window)"), ("stand_drift_xy_m", "stand drift m"),
        ("stand_mean_speed_last5s", "stand speed last5s"), ("dist_after_stop_m", "dist after stop m"),
        ("mean_speed_last2s", "speed last2s"), ("step_disp_forward_m", "step disp fwd m"),
        ("final_yaw_change_rad", "yaw change rad"), ("min_height", "min base z"), ("max_tilt_deg", "max tilt deg"),
        ("head_pitch_err_deg_mean_max", "head pitch err deg mean,|max|")]
print(f"{'scenario':26s} {'metric':30s} {'mujoco':>24s} {'isaac':>24s}")
for name in [r["scenario"] for r in mj["results"]]:
    m, i = M[name], I.get(name)
    if i is None:
        print(f"{name:26s} (no isaac result)")
        continue
    win = m.get("track_window_s")
    head = f"{name:26s} {'fell / h0':30s} {f(m['fell']) + ' ' + f(m['h0']):>24s} {f(i['fell']) + ' ' + f(i['h0']):>24s}"
    print(head + (f"   cmd {m.get('cmd_target')} window {win} s" if win else ""))
    for k, label in KEYS:
        if k in m or k in i:
            print(f"{'':26s} {label:30s} {f(m.get(k)):>24s} {f(i.get(k)):>24s}")
