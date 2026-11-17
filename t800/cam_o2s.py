"""Head-camera images of the T800 (O2S left RGB mount, robot_t800.yaml head_camera) at the initial pose of a scene:
RGB, depth colormap, raw depth (.npy), RGB annotated with every GT barcode label's projected centre (green = visible
by camera.visibility, red = in the image but occluded / back-facing), metrics JSON.

  CUDA_VISIBLE_DEVICES=<card> bash t800/run_isaac.sh -u t800/cam_o2s.py --scene s1_table --tag s1_table
  CUDA_VISIBLE_DEVICES=<card> bash t800/run_isaac.sh -u t800/cam_o2s.py --scene scenario --scenario wall6x5_L0_0000 --tag wall6x5

Outputs: t800/out/cam_o2s/<tag>_{head_rgb,head_depth,head_rgb_annotated}.png, <tag>_head_depth.npy, <tag>_metrics.json

Stated choice: --settle_s 1.0 = stand this long before the images (textures stream in, the robot settles); the same
value as scripts/testsuite/render_scenario.py.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="s1_table")
ap.add_argument("--scenario", default=None, help="test-suite scenario id (with --scene scenario)")
ap.add_argument("--tag", required=True)
ap.add_argument("--settle_s", type=float, default=1.0)
args = ap.parse_args()

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.expanduser("~/random/switchyard/clients"))
from switchyard_progress import Run  # noqa: E402

OUT = os.path.join(HERE, "out", "cam_o2s")
os.makedirs(OUT, exist_ok=True)
gpus = [int(g) for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip().isdigit()]
run = Run(f"hbs-t800-cam-o2s-{args.tag}", label=f"HumanoidBarcodeScan T800 · O2S head camera {args.tag}", gpus=gpus)
run.status("Starting Isaac Sim")

from g1_barcode_sim.env import G1BarcodeEnv  # noqa: E402

t0 = time.time()
env = None


def main():
    global env
    env = G1BarcodeEnv(args.scene, headless=True, seed=0, config_overrides={"robot": {"model": "t800"}})
    import cv2
    from g1_barcode_sim.camera import visibility
    from g1_barcode_sim.env import depth_colormap

    print("MOUNT_CHECK " + json.dumps(env.head_camera_check), flush=True)
    run.progress(0, 2, unit="steps", status="Settling")
    if args.scenario is not None:
        env.reset(scenario=args.scenario)
    r = env.stand_still(args.settle_s)
    if r["status"] != "ok":
        raise RuntimeError(f"robot did not stand: {r}")
    run.progress(1, 2, unit="steps", status="Capturing")
    obs = env.get_obs()
    cc = env.cfg["camera"]
    T_wr = obs["T_world_root_gt"]
    T_wc = T_wr @ obs["T_root_cam"]
    raw = env.head_cam.data.output["distance_to_image_plane"][0].cpu().numpy().squeeze(-1)
    H, W = raw.shape
    vis = [visibility(obs["K"], T_wc, W, H, raw, b, cc["visibility_depth_tol"]) for b in env.layout["barcodes"]]
    p = lambda n: os.path.join(OUT, f"{args.tag}_{n}")  # noqa: E731
    cv2.imwrite(p("head_rgb.png"), cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(p("head_depth.png"), depth_colormap(obs["depth"], cc["depth"]["valid_range"]))
    np.save(p("head_depth.npy"), obs["depth"])
    img = cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR)
    for v in vis:
        if v["in_image"]:
            col = (0, 255, 0) if v["visible"] else (0, 0, 255)
            cv2.circle(img, (int(v["u"]), int(v["v"])), 5, col, 2)
            cv2.putText(img, v["content"], (int(v["u"]) + 6, int(v["v"]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    cv2.imwrite(p("head_rgb_annotated.png"), img)

    # pitch: optical axis below the horizontal; head / torso tilt: root and head-link x axes below the horizontal
    z = T_wc[:3, 2]
    head_T = env._link_T(env.body_names.index(env.cfg["robot"]["head_camera"]["parent_link"]))
    tilt = lambda v: float(np.degrees(math.atan2(-v[2], math.hypot(v[0], v[1]))))  # noqa: E731
    names = list(env.robot.data.joint_names)
    q = env.robot.data.joint_pos[0].cpu().numpy()
    dep = obs["depth"]
    m = {"scene": args.scene, "scenario": args.scenario, "settle_s": args.settle_s, "status": r["status"],
         "mount_check": env.head_camera_check, "mount": cc["mount"],
         "cam_height_above_floor_m": float(T_wc[2, 3]), "cam_pos_world": np.round(T_wc[:3, 3], 4).tolist(),
         "cam_pitch_down_deg": tilt(z), "root_x_pitch_down_deg": tilt(T_wr[:3, 0]),
         "head_link_x_pitch_down_deg": tilt(head_T[:3, 0]),
         "head_torso_joints_deg": {n: float(np.degrees(q[names.index(n)])) for n in names if "HEAD" in n or "TORSO" in n},
         "root_height_m": float(T_wr[2, 3]),
         "T_root_cam": np.round(obs["T_root_cam"], 5).tolist(),
         "depth_valid_frac": float((dep > 0).mean()),
         "depth_min_max_valid_m": [float(dep[dep > 0].min()), float(dep[dep > 0].max())] if (dep > 0).any() else None,
         "barcodes_total": len(vis), "barcodes_in_image": sum(v["in_image"] for v in vis),
         "barcodes_visible": sum(v["visible"] for v in vis),
         "barcodes": [{k: (round(v[k], 2) if isinstance(v[k], float) else v[k]) for k in
                       ("content", "u", "v", "z", "facing", "in_image", "visible", "px_w", "px_h")} for v in vis],
         "wall_s": time.time() - t0}
    json.dump(m, open(p("metrics.json"), "w"), indent=1)
    print("RESULT_JSON " + json.dumps({k: v for k, v in m.items() if k != "barcodes"}), flush=True)
    run.progress(2, 2, unit="steps", status="Done")


try:
    main()
except BaseException as exc:        # report the failure to the dashboard, print it, exit non-zero (Kit hangs on close)
    import traceback
    traceback.print_exc()
    run.failed(repr(exc))
    sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
run.done()
sys.stdout.flush(); os._exit(0)
