"""Write the hand-specified box-wall suite `wall6x5_L0` (one scenario, task 3: scan every label of the wall).

python scripts/testsuite/make_wall_suite.py                       # defaults below (user spec 2026-11-10)
python scripts/testsuite/make_wall_suite.py --facing wall_left --robot_dist 0.70   # earlier side-on start

Geometry (user spec 2026-11-10): a single-depth wall of `cols` x `rows` boxes on the floor, boxes stacked without gaps
(static colliders, each box `on` the one below, as the other stack suites). Every box is box_h tall x box_w wide (the
face towards the robot) x box_d deep. The wall's front face is the -x face at x = factors.yaml world.table_front_x
(1.30 m, the open warehouse aisle every suite uses), centred on y = 0. Boxes are plain kraft colour (appearance
"kraft": a single-colour cube, no decal / tape / print). One code128 label per box on the front face, in its top-left
corner as seen by a viewer facing the face (left = +y), `label_margin` from the face's top and left edges; label = the
suite-generator nominal code128 (factors.yaml barcode: pattern 0.08 m wide, aspect 0.375, white backing with a 0.01 m
border -> 0.10 x 0.05 m). Contents: factors.yaml content_alphabet / nominal content_length, seeded, unique against
every existing suite.

Robot start (scenario robot_init, all parameters of this script): `--facing wall` (default, user request 2026-11-10:
"robot 初始应该正对 stack 面") = yaw 0 deg, the robot's forward axis (+x) along the wall face's inward normal (the face
is the -x face at x = 1.30), root `robot_dist` m from the face plane, `robot_along` m along the wall from its centre (y;
0 = centred on the wall). `--facing wall_right` / `wall_left` keep the earlier side-on starts (yaw +90 / -90 deg: the
wall on the robot's right / left); --yaw_deg overrides the yaw of any of them.

Stated choices (not from a spec):
  robot_dist 0.60 m: root-to-face distance inside the agent's scan-stance grid (g1_agent_system configs/task2.yaml stance
    min/max_face_dist_m 0.35-0.70 m) and within the 0.52-0.63 m the m5 task-2 runs chose; keeps the factors.yaml
    validity robot_footprint_radius (0.35 m) clear of the wall. Reach (T800 URDF FK, scripts/testsuite/measure_sizes.py):
    the right shoulder-pitch axis is 0.011 m behind the root; with the arm straight and raised forward the scanner tip
    (scanner.yaml tip on the robot_t800.yaml mount) is 0.654 m in front of the root, so the 0.20 m scan stand-off
    (task2.yaml offset_m) is reachable at shoulder height up to robot_dist = 0.85 m. At 0.60 m the tip needs 0.40 m of
    the arm's 0.67 m shoulder-to-tip length horizontally, leaving the elbow bent for up / down / sideways reach.
  label_margin: factors.yaml barcode.margin (0.01 m), the minimum in-face margin every suite label obeys.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from g1_barcode_sim.testsuite import SUITES_DIR, load_factors  # noqa: E402
from g1_barcode_sim.testsuite.scenario import label_size, to_layout  # noqa: E402

F = load_factors()
B = F["barcode"]

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="wall6x5_L0")
ap.add_argument("--cols", type=int, default=6, help="boxes along the wall (horizontal)")
ap.add_argument("--rows", type=int, default=5, help="boxes stacked (vertical)")
ap.add_argument("--box_h", type=float, default=0.25, help="box height (m): vertical edge of the robot-facing face")
ap.add_argument("--box_w", type=float, default=0.20, help="box width (m): horizontal edge of the robot-facing face")
ap.add_argument("--box_d", type=float, default=0.40, help="box depth (m), away from the robot")
ap.add_argument("--label_margin", type=float, default=B["margin"], help="label edge to face top / left edge (m)")
ap.add_argument("--robot_dist", type=float, default=0.60, help="robot root to the wall face plane (m)")
ap.add_argument("--robot_along", type=float, default=0.0, help="robot root along the wall from its centre, world y (m)")
FACING_YAW = {"wall": 0.0, "wall_right": 90.0, "wall_left": -90.0}   # yaw that points +x_robot at the -x face / along it
ap.add_argument("--facing", choices=tuple(FACING_YAW), default="wall",
                help="wall: face the wall squarely (yaw 0); wall_right / wall_left: side-on, the wall on that side")
ap.add_argument("--yaw_deg", type=float, default=None, help="start yaw (deg); default from --facing")
ap.add_argument("--seed", type=int, default=0, help="content RNG seed")
args = ap.parse_args()

yaw = args.yaw_deg if args.yaw_deg is not None else FACING_YAW[args.facing]
code_proto = {"type": "code128", "width": B["width"]["nominal"], "label_backing": True, "aspect": B["aspect"]["nominal"],
              "rot_deg": 0.0}
lw, lh = label_size(code_proto, B["backing_border"])
# label centre in face coordinates (barcode.FACE_AXES "-x": u = right of a viewer facing the face = -y, v = up)
u = -(args.box_w / 2 - args.label_margin - lw / 2)
v = args.box_h / 2 - args.label_margin - lh / 2
if u > 0 or v < 0:
    raise ValueError(f"label {lw} x {lh} m with margin {args.label_margin} does not fit a top-left corner of a "
                     f"{args.box_w} x {args.box_h} m face")

# contents already used by any suite (load_scenario resolves by id, the agent's logs key on content)
used = set()
for name in sorted(os.listdir(SUITES_DIR)):
    d = os.path.join(SUITES_DIR, name, "scenarios")
    if name == args.suite or not os.path.isdir(d):
        continue
    for f in os.listdir(d):
        if f.endswith(".json"):
            used.update(c["content"] for c in json.load(open(os.path.join(d, f)))["barcodes"])
rng = np.random.default_rng(args.seed)
alphabet = F["barcode"]["content_alphabet"]
n_chars = B["content_length"]["nominal"]


def new_content():
    while True:
        c = "".join(alphabet[i] for i in rng.integers(0, len(alphabet), n_chars))
        if c not in used:
            used.add(c)
            return c


sid = f"{args.suite}_0000"
boxes, barcodes = [], []
# column c = 0 is the leftmost seen from in front of the face (+y end), as the task-3 tour reads a face left to right
for c in range(args.cols):
    y = (args.cols - 1) / 2 * args.box_w - c * args.box_w
    for r in range(args.rows):
        bid = f"C{c}R{r}"
        boxes.append({"id": bid, "size_wdh": [args.box_w, args.box_d, args.box_h], "appearance": "kraft", "setback": 0.0,
                      "y_center": round(y, 6), "on": f"C{c}R{r - 1}" if r else None})
        barcodes.append(dict(code_proto, box_id=bid, face="-x", u=round(u, 6), v=round(v, 6), content=new_content(),
                             content_id=f"WALL-0000-{bid}"))

note = (f"hand-specified 2026-11-10 (scripts/testsuite/make_wall_suite.py): a {args.cols} x {args.rows} wall of "
        f"{args.cols * args.rows} plain kraft boxes on the floor, no gaps, each {args.box_h} m tall x {args.box_w} m wide "
        f"(face towards the robot) x {args.box_d} m deep; wall {args.cols * args.box_w:.2f} m wide x "
        f"{args.rows * args.box_h:.2f} m tall, front face x = {F['world']['table_front_x']} m, centred on y = 0. One code128 "
        f"label ({lw:.2f} x {lh:.2f} m incl. backing) per box in the top-left corner of the front face (seen facing it), "
        f"{args.label_margin} m from the top and left edges. Robot start: "
        + ("facing the wall face squarely" if args.facing == "wall" else f"side-on, the wall on its {args.facing[5:]}")
        + f", root {args.robot_dist} m from the face, {args.robot_along} m along the wall from its centre, yaw {yaw} deg.")
sc = {"id": sid, "suite": args.suite, "task": 3, "seed": args.seed, "index": 0, "level": "L0", "factors_varied": [],
      "factor": {"note": note, "params": {k: getattr(args, k) for k in ("cols", "rows", "box_h", "box_w", "box_d",
                                                                         "label_margin", "robot_dist", "robot_along",
                                                                         "facing")} | {"yaw_deg": yaw}},
      "background": {"type": "warehouse", "light_scale": 1.0}, "table": None,
      "robot_init": {"dist_to_table": args.robot_dist, "lateral_offset": args.robot_along, "yaw_deg": yaw},
      "boxes": boxes, "barcodes": barcodes, "target": "all", "task2_start": None}
lay = to_layout(sc, F)            # the same validity checks every suite passes (labels fit, outer face, no overlap)

out = os.path.join(SUITES_DIR, args.suite)
os.makedirs(os.path.join(out, "scenarios"), exist_ok=True)


def dumps(obj):
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


open(os.path.join(out, "scenarios", f"{sid}.json"), "w").write(dumps(sc))
manifest = {"suite": args.suite, "base_suite": args.suite, "version": F["version"], "seed": args.seed, "task": 3,
            "level": "L0", "count": 1, "definition": {"note": "hand-specified (scripts/testsuite/make_wall_suite.py)"},
            "rejections": {}, "warnings": [], "skipped_infeasible": [], "code_hash": None, "head_camera_nominal": None,
            "scenario_ids": [sid]}
open(os.path.join(out, "manifest.json"), "w").write(dumps(manifest))
zs = sorted({round(bc["center"][2], 3) for bc in lay["barcodes"]})
print(f"{sid}: {len(lay['boxes'])} boxes, {len(lay['barcodes'])} labels at u={u:.3f} v={v:.3f} (label centre heights {zs} m), "
      f"robot_init {lay['robot_init']} -> {out}")
