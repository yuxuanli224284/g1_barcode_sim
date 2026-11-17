"""Render third-person views of one test-suite scenario with either robot standing at its start pose.

  CUDA_VISIBLE_DEVICES=<card> bash t800/run_isaac.sh -u scripts/testsuite/render_scenario.py \
      --scenario wall6x5_L0_0000 --robot t800 --out_dir t800/out/scenes --prefix wall6x5

Views (aux camera, --width x --height, --hfov):
  left_rear      behind the robot on its left, elevated, looking at the robot and the boxes (the robot's back and the
                 robot-facing box faces); the distance is fitted so the robot's bodies and every box corner are inside
                 the image with a --fit_margin border.
  front          straight on the boxes' -x face (the face the scenario's labels sit on) at mid height, fitted the same
                 way; the robot is moved out of the way for this view only (no physics step).
  front_closeup  straight on the top-left 2 x 2 boxes of the -x face (label placement / orientation check).
  side           level camera on the robot's left (perpendicular to its start heading) at --side_hfov, fitted to the robot
                 and every box: robot height next to the boxes. Horizontal reference lines at the measured robot top,
                 the top box face and every --ruler_step m are drawn at the plane through the robot root parallel to the
                 image (where both the robot and the wall centre stand when the robot faces the wall squarely).

Stated choices: --rear_az_deg 40 / --elev_deg 30, picked by eye from a sweep (az 25 / 40 / 55 x el 30 / 45 on
wall6x5_L0_0000 with the T800): the robot's back and the whole label face in one image, the least wall hidden behind
the robot without the face turning edge-on; --settle_s 1.0 (stand before rendering: textures stream in, the robot
settles), --fit_margin 0.05 (5 % image border, as factors.yaml task2_start.fov_margin_frac).
--side_hfov 15 deg: a narrow lens puts the camera far away, so objects nearer / farther than the robot's plane are
barely magnified (the fitted distance is printed; a 6-box wall end 0.6 m off the robot plane at ~10 m is ~6 % larger);
--ruler_step 0.25 m = one box height of wall6x5_L0.
"""
import argparse
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--scenario", required=True)
ap.add_argument("--robot", choices=("g1", "t800"), required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--prefix", required=True)
ap.add_argument("--width", type=int, default=1600)
ap.add_argument("--height", type=int, default=1000)
ap.add_argument("--hfov", type=float, default=60.0)
ap.add_argument("--rear_az_deg", type=float, nargs="+", default=[40.0], help="0 = straight behind, 90 = straight left of the robot")
ap.add_argument("--elev_deg", type=float, nargs="+", default=[30.0], help="with several az / elev values every pair is rendered "
                "as left_rear_az<a>_el<e> (viewpoint sweep)")
ap.add_argument("--settle_s", type=float, default=1.0)
ap.add_argument("--fit_margin", type=float, default=0.05)
ap.add_argument("--side_hfov", type=float, default=15.0)
ap.add_argument("--ruler_step", type=float, default=0.25)
args = ap.parse_args()

import numpy as np  # noqa: E402

SY = os.path.expanduser("~/random/switchyard/clients")
sys.path.insert(0, SY)
from switchyard_progress import Run  # noqa: E402

gpus = [int(g) for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip().isdigit()]
run = Run(f"hbs-render-{args.prefix}-{args.robot}", label=f"HumanoidBarcodeScan · render {args.scenario} ({args.robot})", gpus=gpus)
run.status("Starting Isaac Sim")
os.makedirs(args.out_dir, exist_ok=True)

from g1_barcode_sim.env import G1BarcodeEnv  # noqa: E402

t0 = time.time()
env = G1BarcodeEnv("scenario", headless=True, seed=0,
                   config_overrides={"robot": {"model": args.robot}, "camera": {"aux": {"width": args.width, "height": args.height}}})


def main():
    import cv2
    from g1_barcode_sim.camera import look_at_ros, project

    W, H = args.width, args.height

    def intr(hfov):
        f = W / 2 / math.tan(math.radians(hfov) / 2)
        return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])

    K = intr(args.hfov)

    def fits(T, pts, K=K):
        uv, z = project(K, T, pts)
        mx, my = args.fit_margin * W, args.fit_margin * H
        return bool(np.all(z > 0) and np.all((uv[:, 0] >= mx) & (uv[:, 0] <= W - mx) & (uv[:, 1] >= my) & (uv[:, 1] <= H - my)))

    def fitted(target, direction, pts, K=K):
        """Camera on the ray target + s * direction, the nearest s (1 cm steps) that frames every point."""
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        for s in np.arange(0.5, 30.0, 0.01):
            T = look_at_ros(target + s * d, target)
            if fits(T, pts, K):
                return T, s
        raise RuntimeError(f"no camera distance up to 30 m frames {len(pts)} points from direction {d}")

    def save(name, T, hfov=args.hfov, draw=None):
        rgb = env.render_aux(T, hfov_deg=hfov)["rgb"]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if draw is not None:
            draw(bgr)
        p = os.path.join(args.out_dir, f"{args.prefix}_{name}.png")
        cv2.imwrite(p, bgr)
        print(f"{name}: eye {np.round(T[:3, 3], 3).tolist()} -> {p}", flush=True)

    run.status("Loading scenario")
    env.reset(scenario=args.scenario)
    r = env.stand_still(args.settle_s)
    if r["status"] != "ok":
        raise RuntimeError(f"robot did not stand: {r}")
    lay = env.layout
    corners = np.array([np.asarray(b["pos"]) + np.asarray(b["size"]) / 2 * np.array([sx, sy, sz])
                        for b in lay["boxes"] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    # robot extent: every link's local mesh bound (USD, link frame) placed at the link's physics pose (the USD link
    # transforms are not updated while CPU physics runs, so world bounds read from USD would be stale)
    from pxr import Usd, UsdGeom
    import omni.usd
    from g1_barcode_sim.camera import quat_wxyz_to_R
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bp, bq = env.robot.data.body_pos_w[0].cpu().numpy(), env.robot.data.body_quat_w[0].cpu().numpy()
    body = []
    for i, name in enumerate(env.body_names):
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{name}")
        if not prim.IsValid():
            raise RuntimeError(f"robot link prim {name} not found")
        rng = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            body.append(bp[i])
            continue
        lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
        loc = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        body.extend(bp[i] + loc @ quat_wxyz_to_R(bq[i]).T)
    body = np.array(body)
    print(f"robot extent: z {body[:, 2].min():.3f}..{body[:, 2].max():.3f} m", flush=True)
    run.progress(0, 4, unit="views", status="Rendering")

    # left_rear: robot frame at its start yaw
    ri = lay["robot_init"]
    ri0 = ri["pos"]
    yaw = float(ri["yaw"])
    fwd, left = np.array([math.cos(yaw), math.sin(yaw), 0.0]), np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    pts = np.vstack([corners, body])
    target = (pts.min(axis=0) + pts.max(axis=0)) / 2
    sweep = len(args.rear_az_deg) * len(args.elev_deg) > 1
    for az in args.rear_az_deg:
        for el in args.elev_deg:
            a, e = math.radians(az), math.radians(el)
            horiz = -math.cos(a) * fwd + math.sin(a) * left
            direction = math.cos(e) * horiz + np.array([0.0, 0.0, math.sin(e)])
            T, s = fitted(target, direction, pts)
            save(f"left_rear_az{az:g}_el{el:g}" if sweep else "left_rear", T)
    run.progress(1, 4, unit="views", status="Rendering")

    # side: level camera on the robot's left, narrow lens, height reference lines in the robot-root plane
    Ks = intr(args.side_hfov)
    T, s = fitted(target, left, pts, Ks)
    root = np.array([ri0[0], ri0[1], 0.0])
    top_robot, top_box = float(body[:, 2].max()), float(corners[:, 2].max())

    def ruler(img):
        # tick labels on the left edge, the measured box / robot tops on the right edge, so a measured top that falls
        # on a tick (box top 1.25 m = 5 x 0.25 m) never prints over that tick's label; every label sits on a white
        # backing so it stays readable over the dark floor.
        lines = [(z, f"{z:.2f} m", (60, 60, 60), 1, "L") for z in np.arange(0.0, top_robot + 1e-9, args.ruler_step)]
        lines += [(top_box, f"box top {top_box:.3f} m", (0, 140, 255), 2, "R"),
                  (top_robot, f"robot top {top_robot:.3f} m", (0, 0, 255), 2, "R")]
        for z, txt, col, th, side in lines:
            uv, _ = project(Ks, T, np.array([root + np.array([0, 0, z]) - 5 * fwd, root + np.array([0, 0, z]) + 5 * fwd]))
            p0, p1 = tuple(int(v) for v in uv[0]), tuple(int(v) for v in uv[1])
            cv2.line(img, p0, p1, col, th, cv2.LINE_AA)
            yy = int((uv[0][1] + uv[1][1]) / 2) - 6
            (tw, tht), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            xx = 10 if side == "L" else img.shape[1] - tw - 10
            cv2.rectangle(img, (xx - 3, yy - tht - 3), (xx + tw + 3, yy + base), (255, 255, 255), -1)
            cv2.putText(img, txt, (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2 if th > 1 else 1, cv2.LINE_AA)
    save("side", T, hfov=args.side_hfov, draw=ruler)
    print(f"side: camera {s:.2f} m from the scene centre, hfov {args.side_hfov} deg; robot top {top_robot:.3f} m, "
          f"box top {top_box:.3f} m", flush=True)
    run.progress(2, 4, unit="views", status="Rendering")

    # front views: the robot out of the frame (teleported far behind the camera side, no physics step)
    env.teleport_root(ri["pos"][0] - 8.0, ri["pos"][1], ri["yaw"])
    front = corners[corners[:, 0] <= corners[:, 0].min() + 1e-9]         # the -x face of the boxes
    tgt = (front.min(axis=0) + front.max(axis=0)) / 2
    T, s = fitted(tgt, np.array([-1.0, 0.0, 0.0]), front)
    save("front", T)
    run.progress(3, 4, unit="views", status="Rendering")
    # top-left 2 x 2 boxes as seen facing the -x face (left = +y)
    ys = sorted({round(float(b["pos"][1]), 6) for b in lay["boxes"]}, reverse=True)[:2]
    zs = sorted({round(float(b["pos"][2]), 6) for b in lay["boxes"]}, reverse=True)[:2]
    sel = [b for b in lay["boxes"] if round(float(b["pos"][1]), 6) in ys and round(float(b["pos"][2]), 6) in zs]
    cc = np.array([np.asarray(b["pos"]) + np.asarray(b["size"]) / 2 * np.array([-1, sy, sz]) for b in sel for sy in (-1, 1) for sz in (-1, 1)])
    T, s = fitted((cc.min(axis=0) + cc.max(axis=0)) / 2, np.array([-1.0, 0.0, 0.0]), cc)
    save("front_closeup", T)
    run.progress(4, 4, unit="views", status="Done")
    print(f"done in {time.time() - t0:.0f} s", flush=True)


try:
    main()
except BaseException as exc:        # report the failure to the dashboard, print it, exit non-zero (Kit hangs on close)
    import traceback
    traceback.print_exc()
    run.failed(repr(exc))
    sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
run.done()
sys.stdout.flush(); os._exit(0)
