"""Preview test suites: for every scenario, the head-camera view at the robot's start pose and
a top view; plus suites/<suite>/previews/overview.png tiling the whole suite.

python scripts/testsuite/preview.py --suite t2_L1_barcode_size
python scripts/testsuite/preview.py --suite all            # every generated suite, one simulator process
"""
import argparse, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--suite", nargs="+", required=True, help="suite name(s) or 'all'")
ap.add_argument("--settle_s", type=float, default=0.5, help="sim time after reset before rendering (textures stream in)")
ap.add_argument("--max", type=int, default=None, help="only the first N scenarios of each suite")
ap.add_argument("--out_dir", default=None, help="write to <out_dir>/<suite>/ instead of suites/<suite>/previews")
ap.add_argument("--oblique", action="store_true", help="also four elevated diagonal views per scenario (robot + boxes; left_rear = behind the robot on its left)")
args = ap.parse_args()

from g1_barcode_sim.testsuite import SUITES_DIR, list_suites, load_suite  # noqa: E402
names = [m["suite"] for m in list_suites()] if args.suite == ["all"] else args.suite
SY = os.environ.get("PROGRESS_CLIENT_DIR", "")   # optional progress reporter (switchyard_progress.Run); unset = none
run = None
if SY and os.path.isdir(SY):
    sys.path.insert(0, SY)
    from switchyard_progress import Run
    run = Run("hbs-testsuite-preview", label="HumanoidBarcodeScan · test-suite previews",
              gpus=[int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.isdigit()])
total = sum(min(len(load_suite(n)), args.max or 10**9) for n in names)

from g1_barcode_sim.env import G1BarcodeEnv  # noqa: E402
env = G1BarcodeEnv("scenario", headless=True, seed=0)
try:
    import cv2
    from g1_barcode_sim.camera import look_at_ros

    def top_view(lay):
        """Straight down, covering the robot start and the table (image right = +x)."""
        r = np.asarray(lay["robot_init"]["pos"][:2], float)
        lo, hi = scene_bounds(lay)
        lo, hi = np.minimum(r - 0.5, lo - 0.2), np.maximum(r + 0.5, hi + 0.2)
        c = (lo + hi) / 2
        ext = max((hi - lo)[0], (hi - lo)[1] * 16 / 9)
        hfov = 60.0
        h = ext / 2 / math.tan(math.radians(hfov) / 2)
        T = look_at_ros(np.array([c[0], c[1], h]), np.array([c[0], c[1], 0.0]), world_up=(0.0, 1.0, 0.0))
        return env.render_aux(T, hfov_deg=hfov)["rgb"]

    def scene_bounds(lay):
        """xy bounds of the table (when there is one) or of the boxes (a stack on the floor)."""
        if lay["furniture"]:
            tp = lay["furniture"][0]["parts"][0]
            c, h = np.asarray(tp["center"][:2]), np.asarray(tp["size"][:2]) / 2
            return c - h, c + h
        P = np.array([np.asarray(b["pos"][:2]) + sx * np.asarray(b["size"][:2]) / 2 for b in lay["boxes"] for sx in ((-1, -1), (1, 1))])
        return P.min(axis=0), P.max(axis=0)

    def oblique_views(lay):
        """Four elevated views from the robot's diagonals (named as seen from the robot standing at its start pose:
        left_rear = behind the robot on its left), framing the robot start and every box."""
        r = np.asarray(lay["robot_init"]["pos"][:2], float)
        yaw = float(lay["robot_init"]["yaw"])
        lo, hi = scene_bounds(lay)
        P = np.vstack([r, lo, hi, [lo[0], hi[1]], [hi[0], lo[1]]])
        c = (P.min(axis=0) + P.max(axis=0)) / 2
        span = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
        top = max(b["pos"][2] + b["size"][2] / 2 for b in lay["boxes"])
        fwd, left = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
        hfov = 60.0
        dist = 0.75 * span / math.tan(math.radians(hfov) / 2) + 1.0
        out = {}
        for name, (sf, sl) in (("left_rear", (-1, 1)), ("right_rear", (-1, -1)), ("left_front", (1, 1)), ("right_front", (1, -1))):
            dxy = (sf * fwd + sl * left) / math.sqrt(2) * dist
            eye = np.array([c[0] + dxy[0], c[1] + dxy[1], 0.55 * dist + 0.5])
            T = look_at_ros(eye, np.array([c[0], c[1], top / 2]))
            out[name] = env.render_aux(T, hfov_deg=hfov)["rgb"]
        return out

    def caption(img, text):
        img = img.copy()
        for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, th + 1, cv2.LINE_AA)
        return img

    done = 0
    for name in names:
        scs = load_suite(name)[: args.max]
        out = os.path.join(args.out_dir, name) if args.out_dir else os.path.join(SUITES_DIR, name, "previews")
        os.makedirs(out, exist_ok=True)
        tiles = []
        for sc in scs:
            obs = env.reset(scenario=sc)
            r = env.stand_still(args.settle_s)
            head = cv2.cvtColor(r["obs"]["rgb"], cv2.COLOR_RGB2BGR)
            top = cv2.cvtColor(top_view(env.layout), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out, f"{sc['id']}_head.png"), head)
            cv2.imwrite(os.path.join(out, f"{sc['id']}_top.png"), top)
            if args.oblique:
                for vn, img in oblique_views(env.layout).items():
                    cv2.imwrite(os.path.join(out, f"{sc['id']}_{vn}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            fv = sc["factor"]["value"] if sc.get("factor") and "value" in sc["factor"] else ""
            fv = f"{fv:.3g}" if isinstance(fv, float) else str(fv)
            lbl = f"{sc['id'].split('_')[-1]} {sc['target'][0].upper()} {sc['background']['type']} {fv}"
            tiles.append(np.hstack([caption(cv2.resize(head, (320, 240)), lbl), cv2.resize(top, (427, 240))]))
            done += 1
            if run:
                run.progress(done, total, unit="scenarios", status=f"previews {name}")
        cols = 3
        blank = np.zeros_like(tiles[0])
        rows = [np.hstack(tiles[i:i + cols] + [blank] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
        cv2.imwrite(os.path.join(out, "overview.png"), np.vstack(rows))
        print(f"{name}: {len(scs)} previews -> {out}/overview.png", flush=True)
    if run:
        run.done()
except BaseException as e:
    import traceback; traceback.print_exc()
    if run:
        run.failed(repr(e))
    sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
sys.stdout.flush(); os._exit(0)
