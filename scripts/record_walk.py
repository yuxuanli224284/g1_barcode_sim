"""Record an mp4 of the G1 walking through a scene: fixed oblique third-person view (aux camera)
with the live head-camera image inset, and the command / sim time overlaid.

python scripts/record_walk.py --scene s2_shelf --out outputs/walk_s2_shelf.mp4
"""
import argparse, math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="s2_shelf")
ap.add_argument("--out", default=None)
ap.add_argument("--fps", type=int, default=25)
args = ap.parse_args()
out = args.out or os.path.join(ROOT, "outputs", f"walk_{args.scene}.mp4")

from g1_barcode_sim.env import G1BarcodeEnv  # noqa: E402
env = G1BarcodeEnv(args.scene, headless=True, seed=0)
try:
    import cv2
    import numpy as np
    from g1_barcode_sim.camera import look_at_ros

    # (label, vx, vy, wz, seconds): a stroll in front of the shelf that stays clear of it
    plan = [("stand", 0, 0, 0, 1.0), ("forward", 0.4, 0, 0, 3.0), ("strafe left", 0, 0.5, 0, 2.0),
            ("turn right", 0, 0, -1.0, 1.5), ("turn left", 0, 0, 1.0, 1.5), ("strafe right", 0, -0.5, 0, 3.5),
            ("backward", -0.3, 0, 0, 2.0), ("stand", 0, 0, 0, 1.0)]
    cam_T = look_at_ros(np.array([-1.6, 2.4, 2.3]), np.array([1.2, 0.0, 0.6]))   # behind-left, looking at the scene
    hfov = 70.0
    frame_dt = 1.0 / args.fps
    steps_per_frame = max(1, round(frame_dt / env.policy_dt))
    vw = None
    n_frames = 0
    for label, vx, vy, wz, secs in plan:
        n = int(round(secs / (steps_per_frame * env.policy_dt)))
        for _ in range(n):
            r = env.step(vx, vy, wz, duration=steps_per_frame * env.policy_dt)
            aux = env.render_aux(cam_T, hfov_deg=hfov)["rgb"]
            img = cv2.cvtColor(aux, cv2.COLOR_RGB2BGR)
            head = cv2.cvtColor(r["obs"]["rgb"], cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            hh, hw = h // 3, int(h // 3 * 4 / 3)
            img[h - hh - 10:h - 10, w - hw - 10:w - 10] = cv2.resize(head, (hw, hh))
            cv2.rectangle(img, (w - hw - 11, h - hh - 11), (w - 9, h - 9), (255, 255, 255), 2)
            cv2.putText(img, "head camera", (w - hw, h - hh - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            txt = f"t={r['obs']['timestamp']:5.2f}s  {label}  cmd vx={r['command'][0]:+.2f} vy={r['command'][1]:+.2f} wz={r['command'][2]:+.2f}  [{r['status']}]"
            cv2.putText(img, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            if vw is None:
                vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            vw.write(img)
            n_frames += 1
            if r["status"] == "fell":
                raise RuntimeError(f"robot fell during '{label}': {r['reason']}")
        print(f"[record] {label} done, frames={n_frames}, pose={env.get_obs()['T_world_root_gt'][:2, 3].round(3)}", flush=True)
    vw.release()
    print(f"RECORDED {out} frames={n_frames} seconds={n_frames / args.fps:.1f}", flush=True)
except BaseException:
    import traceback
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)
env.close()
