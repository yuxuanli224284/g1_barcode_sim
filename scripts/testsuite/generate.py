"""Generate test suites.

python scripts/testsuite/generate.py --suite t1_L0
python scripts/testsuite/generate.py --suite t1_L1 --factor background
python scripts/testsuite/generate.py --suite t1_L2 --num 100 --seed 0
python scripts/testsuite/generate.py --suite all

Writes suites/<suite>/{manifest.json, generation_log.json, scenarios/<id>.json}. Every suite is generated twice in
memory and must come out byte-identical (else error). manifest.json is deterministic (no timestamp); the time
goes to generation_log.json. Pure Python: no simulator needed (task-2 suites read the measured head-camera
parameters from configs/testsuite/head_camera_nominal.yaml).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import yaml  # noqa: E402

from g1_barcode_sim.testsuite import SUITES_DIR, load_factors  # noqa: E402
from g1_barcode_sim.testsuite import geometry as G  # noqa: E402
from g1_barcode_sim.testsuite.sampler import expand_suite_names, generate  # noqa: E402

CODE_FILES = ["g1_barcode_sim/testsuite/sampler.py", "g1_barcode_sim/testsuite/scenario.py", "g1_barcode_sim/testsuite/geometry.py",
              "g1_barcode_sim/barcode.py", "g1_barcode_sim/camera.py", "configs/testsuite/factors.yaml"]


def code_hash(extra=()):
    h = hashlib.sha256()
    for f in list(CODE_FILES) + list(extra):
        h.update(open(os.path.join(ROOT, f), "rb").read())
    return h.hexdigest()[:16]


def dumps(obj):
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="t1_L0 | t1_L1 | t1_L2 | t2_L0 | t2_L1 | t2_L2 | all")
    ap.add_argument("--factor", default=None, help="L1 only: generate just this factor")
    ap.add_argument("--num", type=int, default=None, help="L2 only: number of scenarios (default from factors.yaml)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    F = load_factors()
    cam_path = os.path.join(ROOT, F["head_camera_nominal"])
    cam = G.head_camera(yaml.safe_load(open(cam_path))) if os.path.exists(cam_path) else None
    names = list(F["suites"]) if args.suite == "all" else [args.suite]
    all_contents = {}
    for name in names:
        for full, d in expand_suite_names(F, name, args.factor):
            t0 = time.time()
            runs = [generate(F, cam, full, d, args.seed, args.num) for _ in range(2)]
            blobs = [[dumps(sc) for sc in r[0]] for r in runs]
            if blobs[0] != blobs[1] or runs[0][1:] != runs[1][1:]:
                raise RuntimeError(f"{full}: two generations differ (RNG not fully seeded)")
            scs, reasons, warnings, skipped = runs[0]
            for sc in scs:
                for c in sc["barcodes"]:
                    if c["content"] in all_contents:
                        raise RuntimeError(f"barcode content {c['content']} in {sc['id']} also in {all_contents[c['content']]}")
                    all_contents[c["content"]] = sc["id"]
            out = os.path.join(SUITES_DIR, full)
            if os.path.isdir(os.path.join(out, "scenarios")):
                shutil.rmtree(os.path.join(out, "scenarios"))
            os.makedirs(os.path.join(out, "scenarios"), exist_ok=True)
            for sc, blob in zip(scs, blobs[0]):
                open(os.path.join(out, "scenarios", f"{sc['id']}.json"), "w").write(blob)
            manifest = {"suite": full, "base_suite": name, "version": F["version"], "seed": args.seed, "task": d["task"],
                        "level": d["level"], "count": len(scs), "definition": {k: v for k, v in d.items() if k != "factors"},
                        "rejections": dict(sorted(reasons.items())), "warnings": warnings, "skipped_infeasible": skipped,
                        "code_hash": code_hash([F["head_camera_nominal"]] if d["task"] == 2 else []),
                        "head_camera_nominal": F["head_camera_nominal"] if d["task"] == 2 else None,
                        "scenario_ids": [sc["id"] for sc in scs]}
            open(os.path.join(out, "manifest.json"), "w").write(dumps(manifest))
            open(os.path.join(out, "generation_log.json"), "w").write(dumps(
                {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - t0, 2),
                 "argv": sys.argv[1:], "identical_on_regeneration": True}))
            print(f"{full}: {len(scs)} scenarios, rejections {dict(sorted(reasons.items()))}, warnings {len(warnings)}, "
                  f"skipped infeasible {[(x['index'], x['factor_value'], x['reasons']) for x in skipped]} -> {out}", flush=True)


if __name__ == "__main__":
    main()
