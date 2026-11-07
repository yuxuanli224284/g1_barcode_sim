"""Launch a scene and drive the G1 by keyboard.

python scripts/run_scene.py --scene s1_table [--headless] [--seed 0] [--randomize]

Keys: W/S forward/back, A/D strafe left/right, Q/E turn left/right, Space stop, O/C open/close
grippers, P snapshot, X quit. Each key press holds that command for `--hold` seconds.
- With a display: an OpenCV window shows the head RGB live and takes the keys.
- Headless (no display): type the same keys + Enter in the terminal; the head image is saved to
  outputs/run_scene/<scene>/head_latest.png every `--save_every` seconds.
"""
import argparse
import os
import select
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="s1_table", choices=["s1_table", "s2_shelf", "s3_stack"])
ap.add_argument("--headless", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--randomize", action="store_true")
ap.add_argument("--hold", type=float, default=0.5, help="seconds each key press is applied")
ap.add_argument("--save_every", type=float, default=2.0, help="headless: seconds (sim time) between saved head images")
ap.add_argument("--max_sim_s", type=float, default=None, help="quit after this much sim time (for scripted runs)")
args = ap.parse_args()

headless = args.headless or not os.environ.get("DISPLAY")
from g1_barcode_sim.env import G1BarcodeEnv  # noqa: E402

env = G1BarcodeEnv(args.scene, headless=headless, seed=args.seed,
                   config_overrides={"barcode": {"randomize": {"enabled": bool(args.randomize)}}})
import cv2  # noqa: E402  (after the app: Isaac Sim ships its own Qt/OpenCV paths)

L = env.cfg["robot"]["velocity_limits"]
KEYS = {"w": (L["vx"], 0, 0), "s": (-L["vx"], 0, 0), "a": (0, L["vy"], 0), "d": (0, -L["vy"], 0),
        "q": (0, 0, L["wz"]), "e": (0, 0, -L["wz"]), " ": (0, 0, 0)}
out_dir = os.path.join(ROOT, "outputs", "run_scene", args.scene)
os.makedirs(out_dir, exist_ok=True)
print(__doc__)
print(f"[run_scene] {'headless: type keys + Enter' if headless else 'focus the head-camera window and press keys'}")

cmd, cmd_until, last_save, n_snap = (0.0, 0.0, 0.0), 0.0, -1e9, 0
obs = env.get_obs()
running = True
pending, stdin_eof = [], False
while running:
    key = None
    if headless:
        # read the raw fd: sys.stdin.readline() would pull several lines into Python's buffer, after which
        # select() on the fd no longer reports them
        if not stdin_eof and select.select([sys.stdin.fileno()], [], [], 0)[0]:
            chunk = os.read(sys.stdin.fileno(), 4096)
            stdin_eof = chunk == b""
            pending.extend(chunk.decode(errors="ignore").splitlines())
        if pending:
            key = (pending.pop(0).strip().lower() or " ")[0]
    else:
        cv2.imshow("G1 head RGB", cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(1) & 0xFF
        key = chr(k).lower() if k != 255 else None
    t = obs["timestamp"]
    if key in KEYS:
        cmd, cmd_until = KEYS[key], t + args.hold
    elif key == "o":
        print(env.set_gripper("both", 1.0))
    elif key == "c":
        print(env.set_gripper("both", 0.0))
    elif key == "p":
        n_snap += 1
        env.save_snapshot(os.path.join(out_dir, f"snap_{n_snap:03d}"))
        print("[run_scene] snapshot", n_snap)
    elif key == "x":
        running = False
    if t > cmd_until:
        cmd = (0.0, 0.0, 0.0)
    r = env.step(*cmd, duration=env.policy_dt)
    obs = r["obs"]
    if r["status"] != "ok":
        print(f"[run_scene] {r['status']}: {r['reason']} -> reset")
        obs = env.reset()
    if headless and obs["timestamp"] - last_save >= args.save_every:
        cv2.imwrite(os.path.join(out_dir, "head_latest.png"), cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR))
        last_save = obs["timestamp"]
    if args.max_sim_s is not None and obs["timestamp"] >= args.max_sim_s:
        running = False
env.close()
