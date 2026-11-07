"""Render overview images for every scene: top-down + oblique views (scene YAML `views`) + head camera.

python scripts/render_views.py [--scenes s1_table s2_shelf s3_stack]
Writes outputs/<scene>/{topdown,view_*,head_rgb,head_depth}.png. One process per scene (one Isaac app per process).
"""
import argparse, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--scenes", nargs="*", default=["s1_table", "s2_shelf", "s3_stack"])
ap.add_argument("--_worker", default=None)
args = ap.parse_args()

if args._worker:
    from g1_barcode_sim.env import G1BarcodeEnv
    env = G1BarcodeEnv(args._worker, headless=True, seed=0)
    try:
        out = os.path.join(ROOT, "outputs", args._worker)
        env.save_snapshot(out)
        print("WROTE", sorted(f for f in os.listdir(out) if f.endswith(".png")), flush=True)
    except BaseException:
        import traceback
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    env.close()
else:
    failed = []
    for sc in args.scenes:
        rc = subprocess.run([sys.executable, os.path.abspath(__file__), "--_worker", sc], cwd=ROOT).returncode
        print(f"{sc}: rc={rc}", flush=True)
        if rc:
            failed.append(sc)
    sys.exit(1 if failed else 0)
