"""Manual control of the G1: base walking + wrist position (IK) + grippers, with optional video recording.

python scripts/teleop.py --scene s1_table                          # interactive
python scripts/teleop.py --scene s1_table --record out.mp4         # interactive + record
python scripts/teleop.py --scene s1_table --demo --record out.mp4  # built-in control test sequence
python scripts/teleop.py --scene s1_table --demo_scanner --record out.mp4  # scanner + scanner-camera test sequence

Keys (one key per line when headless; with a display the OpenCV window takes the keys directly):
  base    w/s forward/back   a/d left/right   q/e turn left/right   space stop  (held --hold s per press)
  arm     1 left  2 right  3 both   (selects which wrist the keys below move; default: both)
  wrist   i/k +x/-x (forward/back)   j/l +y/-y (left/right)   u/o +z/-z (up/down)   r arms home
          (root frame, --wrist_step m per press, limited to +-0.25 m around the default wrist position)
  gripper g open   h close   (the hand holding the scanner keeps its grip; configs/scanner.yaml)
  scanner f aim mode on/off (on: the scanner holds its current pose; with arm 1 or 3 selected, i/k/j/l/u/o move the
            scanner tip instead of the wrist)   t/v aim up/down   c/b aim left/right (--aim_step_deg)   n scan
  misc    p snapshot   x quit
"""
import argparse
import math
import os
import select
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--scene", default="s1_table", choices=["s1_table", "s2_shelf", "s3_stack"])
ap.add_argument("--headless", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--hold", type=float, default=0.5, help="seconds a base key press is applied")
ap.add_argument("--wrist_step", type=float, default=0.03, help="metres per wrist key press")
ap.add_argument("--record", default=None, help="write an mp4 (chase camera + head-camera and scanner-camera insets)")
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--aim_step_deg", type=float, default=5.0, help="degrees per scanner aim key press")
ap.add_argument("--demo", action="store_true", help="run the built-in control test sequence instead of keys")
ap.add_argument("--demo_scanner", action="store_true",
                help="scanner test: for every target barcode, place the robot in front of it (teleport), aim the scanner "
                     "at it with aim_scanner_at() from 0.30 / 0.20 / 0.12 m and call scan()")
ap.add_argument("--max_sim_s", type=float, default=None)
ap.add_argument("--policy", default=None, help="use the unitree walker with this policy file under unitree assets/model "
                                                "(default: robot.yaml locomotion, i.e. GR00T gear_wbc)")
args = ap.parse_args()

headless = args.headless or not os.environ.get("DISPLAY") or args.demo or args.demo_scanner
from g1_barcode_sim.env import G1BarcodeEnv  # noqa: E402

overrides = {"robot": {"locomotion": "unitree", "policy_model": f"assets/model/{args.policy}"}} if args.policy else None
env = G1BarcodeEnv(args.scene, headless=headless, seed=args.seed, config_overrides=overrides)
POLICY = "gear_wbc" if env.locomotion == "gear_wbc" else os.path.basename(env.cfg["robot"]["policy_model"])
try:
    import cv2
    import numpy as np
    from g1_barcode_sim.camera import look_at_ros

    L = env.cfg["robot"]["velocity_limits"]
    BASE = {"w": (L["vx"], 0, 0), "s": (-L["vx"], 0, 0), "a": (0, L["vy"], 0), "d": (0, -L["vy"], 0),
            "q": (0, 0, L["wz"]), "e": (0, 0, -L["wz"]), " ": (0, 0, 0)}
    st = args.wrist_step
    WRIST = {"i": (st, 0, 0), "k": (-st, 0, 0), "j": (0, st, 0), "l": (0, -st, 0), "u": (0, 0, st), "o": (0, 0, -st)}
    SIDES = {"1": "left", "2": "right", "3": "both"}

    def _rot(axis, deg):
        a = np.radians(deg); c, s_ = np.cos(a), np.sin(a)
        x, y, z = axis
        return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s_, x * z * (1 - c) + y * s_],
                         [y * x * (1 - c) + z * s_, c + y * y * (1 - c), y * z * (1 - c) - x * s_],
                         [z * x * (1 - c) - y * s_, z * y * (1 - c) + x * s_, c + z * z * (1 - c)]])
    AS = args.aim_step_deg   # scan direction keys, root frame: pitch up/down = about -y/+y, yaw left/right = about +z/-z
    AIM_ROT = {"t": _rot((0, -1, 0), AS), "v": _rot((0, 1, 0), AS), "c": _rot((0, 0, 1), AS), "b": _rot((0, 0, -1), AS)}
    out_dir = os.path.join(ROOT, "outputs", "teleop", args.scene)
    os.makedirs(out_dir, exist_ok=True)

    steps_per_frame = max(1, round(1.0 / args.fps / env.policy_dt))
    writer = None
    hud = {"mode": "", "side": "both", "scan": ""}

    def chase_view():
        """Camera in front-left of the robot (moves with it) so both arms and the head are visible."""
        T = env._T_world_root()
        yaw = math.atan2(T[1, 0], T[0, 0])
        c, s_ = math.cos(yaw), math.sin(yaw)
        def w(p):  # root-frame xy offset -> world
            return np.array([T[0, 3] + c * p[0] - s_ * p[1], T[1, 3] + s_ * p[0] + c * p[1], p[2]])
        if args.demo_scanner:   # left side, close: the scanner hand and the target box stay in view at the table
            return env.render_aux(look_at_ros(w((0.55, 1.5, 1.35)), w((0.45, 0.0, 0.95))), hfov_deg=55.0)["rgb"]
        return env.render_aux(look_at_ros(w((2.0, 1.3, 1.6)), w((0.15, 0.0, 0.85))), hfov_deg=60.0)["rgb"]

    def frame(obs, status):
        img = cv2.cvtColor(chase_view(), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        hh, hw = h // 3, int(h // 3 * 4 / 3)
        insets = [("head cam", obs["rgb"], w - hw - 10)]
        if "wrist_rgb" in obs:
            insets.append(("scanner cam", obs["wrist_rgb"], 10))
        for name, im, x0 in insets:
            img[h - hh - 10:h - 10, x0:x0 + hw] = cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), (hw, hh))
            cv2.rectangle(img, (x0 - 1, h - hh - 11), (x0 + hw + 1, h - 9), (255, 255, 255), 2)
            for col, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
                cv2.putText(img, name, (x0 + 8, h - hh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, th, cv2.LINE_AA)
        lines = [f"t={obs['timestamp']:5.2f}s  {hud['mode']}  [{status}]  policy={POLICY}",
                 f"base cmd vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f}   arm keys -> {hud['side']}"]
        for sd in ("left", "right"):
            p, tg = obs["wrist_pos_root"][sd], obs["wrist_target_root"][sd]
            err = "" if tg is None else f"  target ({tg[0]:+.2f},{tg[1]:+.2f},{tg[2]:+.2f})  err {np.linalg.norm(tg - p) * 100:4.1f} cm"
            lines.append(f"{sd:5s} wrist ({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}){err}  grip {obs['gripper_state'][sd]:.2f}")
        if "T_root_scanner_tip" in obs:
            d = obs["T_root_scanner_tip"][:3, 0]
            ae = obs["scanner_aim_error"]
            lines.append(f"scanner dir (root) elev {math.degrees(math.asin(np.clip(d[2], -1, 1))):+5.1f} deg  yaw {math.degrees(math.atan2(d[1], d[0])):+6.1f} deg"
                         + ("" if ae is None else f"   aim err {ae['pos_m'] * 100:4.1f} cm {ae['angle_deg']:4.1f} deg"))
        if hud["scan"]:
            lines.append(hud["scan"])
        for i, t in enumerate(lines):
            for col, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
                cv2.putText(img, t, (16, 34 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, th, cv2.LINE_AA)
        return img

    def advance(n_frames):
        """Advance n_frames video frames (steps_per_frame policy steps each); record / show; return last obs."""
        global writer
        obs, status = None, "ok"
        for _ in range(n_frames):
            r = env.step(*cmd, duration=steps_per_frame * env.policy_dt)
            env.walker.set_velocity(*cmd)  # step() zeroes the command at its end; keep holding it between frames
            obs, status = r["obs"], r["status"]
            if args.record or not headless:
                img = frame(obs, status)
                if args.record:
                    if writer is None:
                        writer = cv2.VideoWriter(args.record + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), args.fps, img.shape[1::-1])
                    writer.write(img)
                if not headless:
                    cv2.imshow("G1 teleop", img)
            if status == "fell":   # keep going so the run/video completes: record it, put the robot back
                print(f"[teleop] FELL during '{hud['mode']}': {r['reason']} -> reset", flush=True)
                falls.append({"segment": hud["mode"], "t": obs["timestamp"], "reason": r["reason"]})
                obs = env.reset()
                return obs, "fell"
        return obs, status

    cmd = (0.0, 0.0, 0.0)
    log, falls = [], []
    if args.demo or args.demo_scanner:
        def seg(label, n_s, base=(0, 0, 0), wrist=None, grip=None):
            """wrist: {side: (dx,dy,dz) relative to home} reached by a linear ramp over the segment."""
            global cmd
            hud["mode"] = label
            cmd = base
            if grip is not None:
                env.set_gripper("both", grip)
            n = int(round(n_s * args.fps))
            if wrist:
                start = {sd: (env.arm.target[sd] if env.arm.target[sd] is not None else env.arm.home_pos(sd)) for sd in wrist}
                goal = {sd: env.arm.home_pos(sd) + np.asarray(v, dtype=float) for sd, v in wrist.items()}
                ramp = max(1, int(n * 0.6))
                for k in range(n):
                    a = min(1.0, (k + 1) / ramp)
                    for sd in wrist:
                        env.arm.set_target(sd, (1 - a) * start[sd] + a * goal[sd])
                    obs, status = advance(1)
            else:
                obs, status = advance(n)
            T = obs["T_world_root_gt"]
            rec = {"segment": label, "status": status, "root_xy": T[:2, 3].round(3).tolist(),
                   "root_yaw_deg": round(math.degrees(math.atan2(T[1, 0], T[0, 0])), 1),
                   "wrist_err_cm": {sd: (None if env.arm.error(sd) is None else round(env.arm.error(sd) * 100, 2)) for sd in ("left", "right")},
                   "grip": {sd: round(obs["gripper_state"][sd], 2) for sd in ("left", "right")},
                   "wrist_pos_root": {sd: np.round(obs["wrist_pos_root"][sd], 3).tolist() for sd in ("left", "right")},
                   "arm_q_target_vs_actual_left": [np.round(env.arm.q_cmd["left"], 2).tolist(),
                                                   np.round(env.robot.data.joint_pos[0, env.arm.joint_idx["left"]].cpu().numpy(), 2).tolist()]}
            log.append(rec)
            print("[demo]", rec, flush=True)

    if args.demo:
        seg("stand", 1.0)
        seg("base: forward", 2.0, (L["vx"], 0, 0))
        seg("base: backward", 2.0, (-L["vx"], 0, 0))
        seg("base: strafe left", 2.0, (0, L["vy"], 0))
        seg("base: strafe right", 2.0, (0, -L["vy"], 0))
        seg("base: turn left", 1.5, (0, 0, L["wz"]))
        seg("base: turn right", 1.5, (0, 0, -L["wz"]))
        seg("stand", 1.0)
        seg("left wrist: forward +0.20", 2.5, wrist={"left": (0.20, 0, 0)})
        seg("left wrist: up +0.20", 2.5, wrist={"left": (0.20, 0, 0.20)})
        seg("left wrist: out (left) +0.10", 2.5, wrist={"left": (0.20, 0.10, 0.20)})
        seg("right wrist: forward +0.20, up +0.10", 2.5, wrist={"right": (0.20, 0, 0.10)})
        seg("right wrist: in (left) +0.10", 2.5, wrist={"right": (0.20, 0.10, 0.10)})
        seg("both wrists: back home", 2.5, wrist={"left": (0, 0, 0), "right": (0, 0, 0)})
        seg("grippers: close", 5.0, grip=0.0)    # gripper time constant ~2 s (upstream actuator friction)
        seg("grippers: open", 5.0, grip=1.0)
        seg("walk forward, both wrists forward +0.15", 2.5, (L["vx"], 0, 0), wrist={"left": (0.15, 0, 0.05), "right": (0.15, 0, 0.05)})
        seg("stand", 1.0)
    elif args.demo_scanner:
        sd = env.scanner_side
        from g1_barcode_sim.camera import inv_T
        seg("stand", 1.0)
        for off in [(0.20, 0.0, 0.05), (0.10, -0.10, 0.15)]:
            seg(f"{sd} wrist {off}: scanner camera view", 3.0, wrist={sd: off})
        env.reset_arms()
        scans = []
        STANDOFF, LATERAL = 0.55, 0.20   # root 0.55 m from the label plane, label 0.20 m to the left of the root
                                         # (in front of the left shoulder): reachable with 0 error

        def to_root(v, point=True):
            T = inv_T(env.get_obs()["T_world_root_gt"])
            return T[:3, :3] @ np.asarray(v, float) + (T[:3, 3] if point else 0.0)

        for bc in [b for b in env.get_ground_truth()["barcodes"] if not b.get("distractor")]:
            n = np.asarray(bc["normal"], float); nh = n.copy(); nh[2] = 0.0; nh /= np.linalg.norm(nh)
            yaw = math.atan2(-nh[1], -nh[0])
            left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
            p = np.asarray(bc["center"]) + STANDOFF * nh - LATERAL * left
            env.teleport_root(float(p[0]), float(p[1]), yaw)          # test placement (GT); the scan itself uses no GT
            seg(f"placed in front of {bc['content']} (teleport)", 1.0)
            for dist in (0.30, 0.20, 0.12):
                env.aim_scanner_at(to_root(bc["center"]), to_root(bc["normal"], False), dist)
                seg(f"aim_scanner_at({bc['content']}, d={dist:.2f})", 3.0)
                r = env.scan()
                o = env.get_obs()
                rec = {"barcode": bc["content"], "aim_distance": dist, "scan_status": r["status"], "content": r["content"],
                       "correct": r["content"] == bc["content"], "aim_error": o["scanner_aim_error"],
                       "gt_hit": r["gt"]["hit"]}
                scans.append(rec)
                hud["scan"] = f"scan -> {r['content'] or r['status']}  ({'correct' if rec['correct'] else 'wrong/none'})"
                print("[demo] scan", rec, flush=True)
                import cv2 as _cv
                _cv.imwrite(os.path.join(out_dir, f"scan_{bc['content']}_{dist:.2f}m.png"), _cv.cvtColor(o["wrist_rgb"], _cv.COLOR_RGB2BGR))
                seg(f"scan result: {hud['scan']}", 1.5)
                hud["scan"] = ""
            env.reset_arms()
        log.append({"segment": "scan_summary", "scans": scans})
        seg("stand", 1.0)
    else:
        print(__doc__)
        pending, eof, until, n_snap = [], False, 0.0, 0
        t_sim = 0.0
        while True:
            key = None
            if headless:
                if not eof and select.select([sys.stdin.fileno()], [], [], 0)[0]:
                    chunk = os.read(sys.stdin.fileno(), 4096)
                    eof = chunk == b""
                    pending.extend(chunk.decode(errors="ignore").splitlines())
                if pending:
                    key = (pending.pop(0).strip().lower() or " ")[0]
            else:
                k = cv2.waitKey(1) & 0xFF
                key = chr(k).lower() if k != 255 else None
            if key in BASE:
                cmd, until = BASE[key], t_sim + args.hold
                hud["mode"] = f"base key {key!r}"
            elif key in SIDES:
                hud["side"] = SIDES[key]
            elif key in WRIST:
                t = env.get_obs().get("scanner_aim_target")
                if t is not None and hud["side"] in (env.scanner_side, "both"):   # aim mode: keys move the scanner tip
                    print(env.aim_scanner(t["pos"] + np.asarray(WRIST[key]), t["dir"]))
                    if hud["side"] == "both":
                        other = "right" if env.scanner_side == "left" else "left"
                        print(env.move_wrist(other, *WRIST[key]))
                else:
                    print(env.move_wrist(hud["side"], *WRIST[key]))
                hud["mode"] = f"wrist key {key!r}"
            elif key == "f" and env.scanner_on:
                o = env.get_obs()
                if o["scanner_aim_target"] is None:      # lock the scanner where it is; aim keys then steer it
                    T = o["T_root_scanner_tip"]
                    env.aim_scanner(T[:3, 3], T[:3, 0]); hud["mode"] = "scanner aim ON"
                else:
                    env.reset_arms(); hud["mode"] = "scanner aim OFF (arms home)"
            elif key in AIM_ROT and env.scanner_on and env.get_obs()["scanner_aim_target"] is not None:
                t = env.get_obs()["scanner_aim_target"]
                env.aim_scanner(t["pos"], AIM_ROT[key] @ t["dir"]); hud["mode"] = f"aim key {key!r}"
            elif key == "n" and env.scanner_on:
                r = env.scan()
                hud["scan"] = f"scan -> {r['content'] or r['status']}"
                print("[teleop] scan", {k: r[k] for k in ("status", "content", "decoded", "gt")}, flush=True)
            elif key == "r":
                env.reset_arms(); hud["mode"] = "arms home"
            elif key == "g":
                env.set_gripper("both", 1.0)
            elif key == "h":
                env.set_gripper("both", 0.0)
            elif key == "p":
                n_snap += 1
                env.save_snapshot(os.path.join(out_dir, f"snap_{n_snap:03d}"))
                print("[teleop] snapshot", n_snap, flush=True)
            elif key == "x":
                break
            if t_sim > until:
                cmd = (0.0, 0.0, 0.0)
            obs, status = advance(1)
            t_sim = obs["timestamp"]
            if args.max_sim_s is not None and t_sim >= args.max_sim_s:
                break

    if writer is not None:
        writer.release()
        # re-encode to H.264 for players / browsers; keep the raw file if ffmpeg is missing
        import shutil, subprocess
        ff = shutil.which("ffmpeg")
        if ff is None:
            try:
                import imageio_ffmpeg
                ff = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                ff = None
        if ff:
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", args.record + ".tmp.mp4", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", args.record], check=True)
            os.remove(args.record + ".tmp.mp4")
        else:
            os.replace(args.record + ".tmp.mp4", args.record)
        print("RECORDED", args.record, flush=True)
    if log:
        import json
        with open(os.path.splitext(args.record or os.path.join(out_dir, "demo"))[0] + "_log.json", "w") as f:
            json.dump({"policy": POLICY, "segments": log, "falls": falls}, f, indent=1)
except BaseException:
    import traceback
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)
env.close()
