"""D435i head camera check (camera_head.yaml model d435i), run in the simulator env:

1. torso pitch (GR00T body command) levels standing still: waist pitch, pelvis tilt, head camera pitch, status -> the
   lean calibration the agent plans its "look up" views with (--calib_out);
2. stereo depth on a scenario: the robot placed at several distances in front of the stack, colour / infrared images
   and the colour camera's render depth (ground truth here) saved, the agent's StereoDepth run on the infrared pair
   and compared with the render depth.

python scripts/d435i_check.py --scenario stack4f_L0_0000 --out outputs/d435i_check [--calib_out path.json]
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="stack4f_L0_0000")
ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "d435i_check"))
ap.add_argument("--agent_root", default=os.path.join(os.path.dirname(ROOT), "g1_agent_system"))
ap.add_argument("--pitches", default="0,-0.1,-0.2,-0.3,-0.4,-0.5,0.2")
ap.add_argument("--dists", default="0.6,1.0,1.4,2.0,2.5")
ap.add_argument("--calib_out", default=None)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

import numpy as np  # noqa: E402
from g1_barcode_sim.env import G1BarcodeEnv  # noqa: E402

env = G1BarcodeEnv("scenario", headless=True, seed=0)
import cv2  # noqa: E402
sys.path.insert(0, args.agent_root)
import yaml  # noqa: E402
from g1agent.perception.stereo_depth import StereoDepth  # noqa: E402
from g1_barcode_sim.camera import inv_T, make_T  # noqa: E402

report = {"scenario": args.scenario}
try:
    env.reset(scenario=args.scenario)
    env.stand_still(1.5)
    w = env.walker
    rpy0 = np.asarray(w.rpy_default, float).copy()
    jn = list(env.robot.data.joint_names)

    def cam_pitch_deg():
        T = env._link_T(env._cam_body) @ env._T_link_cam
        z = T[:3, 2]
        return math.degrees(math.asin(-z[2]))      # optical axis below the horizontal

    def pelvis_rp_deg():
        R = env._T_world_root()[:3, :3]
        return math.degrees(math.atan2(R[2, 1], R[2, 2])), math.degrees(math.asin(-R[2, 0]))

    # ---- 1. torso pitch levels
    levels = []
    for p in [float(v) for v in args.pitches.split(",")]:
        t0 = time.time()
        r = env.set_body_pose(None, rpy0 + np.array([0.0, p, 0.0]), duration=1.0, settle=2.0)
        q = env.robot.data.joint_pos[0].cpu().numpy()
        roll, pitch = pelvis_rp_deg()
        levels.append({"pitch_cmd": p, "status": r["status"], "reason": r["reason"], "tilt_deg": r.get("tilt_deg"),
                       "pelvis_z": r.get("pelvis_z"), "pelvis_roll_deg": round(roll, 3), "pelvis_pitch_deg": round(pitch, 3),
                       "waist_pitch": float(q[jn.index("waist_pitch_joint")]), "cam_pitch_below_horizontal_deg": round(cam_pitch_deg(), 2),
                       "q": {n: float(v) for n, v in zip(jn, q)}, "wall_s": round(time.time() - t0, 2)})
        print("PITCH", {k: v for k, v in levels[-1].items() if k != "q"}, flush=True)
        if r["status"] != "ok":
            env.reset(scenario=args.scenario); env.stand_still(1.5)
    env.set_body_pose(None, rpy0, duration=1.0, settle=1.0)
    report["pitch_levels"] = [{k: v for k, v in l.items() if k != "q"} for l in levels]
    if args.calib_out:
        json.dump({"source": f"g1_barcode_sim scripts/d435i_check.py ({args.scenario}, Isaac, GR00T walker): standing still, "
                             "arms at home, torso pitch command ramped 1.0 s from upright, held 2.0 s; values at the end",
                   "rpy_default": rpy0.tolist(), "levels": levels}, open(args.calib_out, "w"), indent=1)

    # ---- 2. stereo depth vs render depth
    cc = env.cfg["camera"]
    st = cc["stereo"]
    hd = yaml.safe_load(open(os.path.join(args.agent_root, "configs", "perception.yaml")))["head_depth"]
    T_link_color = make_T(cc["offset_pos"], cc["offset_rot_wxyz"])
    T_color_left = inv_T(T_link_color) @ env._T_link_ir["left"]
    sd = None
    front_x = min(b["pos"][0] - b["size"][0] / 2 for b in env.layout["boxes"])
    shots = []
    for d in [float(v) for v in args.dists.split(",")]:
        for p in (0.0, min(l["pitch_cmd"] for l in levels if l["status"] == "ok")):
            env.teleport_root(front_x - d, 0.0, 0.0)
            env.stand_still(1.5)
            env.set_body_pose(None, rpy0 + np.array([0.0, p, 0.0]), duration=1.0, settle=1.5)
            t0 = time.time()
            o = env.get_obs()
            s = env.head_stereo_obs()
            t_synth = time.time() - t0
            if sd is None:
                sd = StereoDepth(hd, s["K_ir"], s["K_ir_right"], s["T_irleft_irright"], [st["height"], st["width"]],
                                 o["K"], o["rgb"].shape[:2], T_color_left)
            depth, stats = sd.compute(s["ir_left"], s["ir_right"])
            gt = o["depth"]
            both = (gt > 0) & (depth > 0)
            err = (depth - gt)[both]
            tag = f"d{d:.1f}_p{p:+.1f}"
            cv2.imwrite(os.path.join(args.out, f"{tag}_rgb.png"), cv2.cvtColor(o["rgb"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(args.out, f"{tag}_irL.png"), s["ir_left"])
            cv2.imwrite(os.path.join(args.out, f"{tag}_irR.png"), s["ir_right"])
            vis = lambda z: cv2.applyColorMap(np.where(z > 0, 255 - np.clip(z / 3.0 * 255, 0, 255), 0).astype(np.uint8), cv2.COLORMAP_JET) * (z > 0)[..., None]
            cv2.imwrite(os.path.join(args.out, f"{tag}_depth_stereo.png"), vis(depth))
            cv2.imwrite(os.path.join(args.out, f"{tag}_depth_gt.png"), vis(gt))
            np.savez_compressed(os.path.join(args.out, f"{tag}.npz"), depth=depth, gt=gt, irL=s["ir_left"], irR=s["ir_right"], rgb=o["rgb"])
            rel = np.abs(err) / gt[both]
            shot = {"d": d, "pitch_cmd": p, "cam_pitch_deg": round(cam_pitch_deg(), 2), "synth_s": round(t_synth, 3),
                    **stats, "gt_valid_frac": float((gt > 0).mean()), "n_both": int(both.sum()),
                    "abs_err_median_m": float(np.median(np.abs(err))) if err.size else None,
                    "abs_err_p90_m": float(np.percentile(np.abs(err), 90)) if err.size else None,
                    "rel_err_median": float(np.median(rel)) if err.size else None,
                    "frac_err_gt_5cm": float((np.abs(err) > 0.05).mean()) if err.size else None,
                    "coverage_of_gt": float(both.sum() / max(1, (gt > 0).sum()))}
            shots.append(shot)
            print("SHOT", shot, flush=True)
    report["shots"] = shots
    # simulation speed with the extra cameras
    t0 = time.time(); env.stand_still(2.0); report["sim_s_per_s"] = round((time.time() - t0) / 2.0, 3)
    print("SPEED wall s per sim s:", report["sim_s_per_s"], flush=True)
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    print("DONE", flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    os._exit(1)
sys.stdout.flush()
os._exit(0)
