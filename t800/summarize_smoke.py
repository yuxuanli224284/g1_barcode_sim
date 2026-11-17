"""Short text summary of the smoke_all.sh stages (for the log and the [HumanScan] report email).
python t800/summarize_smoke.py > t800/out/smoke_summary.txt"""
import json
import os
import re

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def last_json(log, tag):
    p = os.path.join(OUT, "logs", log)
    if not os.path.exists(p):
        return None, "log missing"
    txt = open(p, errors="replace").read()
    m = re.findall(rf"^{tag} (.*)$", txt, flags=re.M)
    if m:
        return json.loads(m[-1]), None
    tb = txt[txt.rfind("Traceback"):][:1500] if "Traceback" in txt else txt[-800:]
    return None, tb


lines = []
m, err = last_json("t800_smoke.log", "RESULT_JSON")
if m is None:
    lines.append(f"T800 smoke: FAILED\n{err}")
else:
    lines.append(f"T800 smoke: fell={m['fell']} non_ok={m['any_non_ok']} init_z={m['init_root_z']:.3f} "
                 f"min_z={m['min_root_z']:.3f} max_tilt={m['max_tilt_deg']:.1f}deg realtime={m['realtime_factor']:.2f}")
    if "stand" in m:
        s = m["stand"]
        lines.append(f"  stand 10s: drift {s['drift_xy_m']:.3f} m, yaw {s['yaw_change_rad']:+.3f} rad, "
                     f"speed(last5s) {s['mean_speed_last5s']:.3f} m/s, z {s['root_z_mean']:.3f}")
    for k, v in m.get("walk_transient", {}).items():
        lines.append(f"  {k} (2-3 s phase, start-up transient): cmd {v['cmd']} meas vx/vy/wz "
                     f"{[round(x, 3) for x in v['mean_vx_vy_wz']]}")
    if "head_pitch_err_deg" in m:
        h = m["head_pitch_err_deg"]
        lines.append(f"  head pitch joint err: stand mean {h['stand_last5s_mean']:+.3f} deg, walk max |.| "
                     f"{h['walk_max_abs']:.3f} deg, run max |.| {h['run_max_abs']:.3f} deg")
    for k, v in m.get("stops", {}).items():
        lines.append(f"  {k}: dist after stop {v['dist_after_stop_m']:.3f} m, final speed {v['speed_last_0.5s']:.3f}")
    a = m.get("arm") or {}
    if a:
        lines.append(f"  arm: hold err max {a['hold_err_max_rad']:.4f} rad (ff on), "
                     f"{a.get('hold_noff_err_max_rad', float('nan')):.4f} rad (ff off), ramp err max "
                     f"{a['ramp_err_max_rad']:.3f}, back err {a['back_final_err_max_rad']:.4f}, body drift "
                     f"{a['body_drift_during_arm_m']:.3f} m")
    c = m.get("camera") or {}
    if c:
        lines.append(f"  head cam: height {c['cam_height_world_m']:.3f} m, pitch {c['cam_pitch_down_deg']:.1f} deg, "
                     f"depth valid {c['depth_valid_frac']:.2f}, scanner rgb mean {c.get('scanner_rgb_mean')}")
for tag in ("g1_base", "g1_mod"):
    g, err = last_json(f"{tag}.log", "RESULT_JSON")
    lines.append(f"{tag}: " + ("FAILED " + err[-300:] if g is None else
                               f"fell={g['fell']} stand drift {g['stand']['drift_xy_m']:.4f} final {[round(x, 4) for x in g['final_pose_xy_yaw']]}"))
c, err = last_json("g1_compare.log", "COMPARE_JSON")
lines.append("G1 identity: " + ("n/a " + (err or "")[-200:] if c is None else json.dumps(c)))
s2s, err = last_json("isaac_sim2sim.log", "RESULT_JSON")
lines.append("Isaac sim2sim: " + ("n/a " + (err or "")[-300:] if s2s is None else json.dumps(s2s)))
cmp_path = os.path.join(OUT, "sim2sim_compare.txt")
if s2s is not None and os.path.exists(cmp_path):
    lines.append(open(cmp_path).read())
print("\n".join(lines))
