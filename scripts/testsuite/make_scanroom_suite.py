"""Suite scanroom6x5_L0: the lab's scan room (photogrammetry reconstruction, 2026-11-17) with our barcode labels.

    python scripts/testsuite/make_scanroom_suite.py [--robot_dist 0.65]

Subset suites (same boxes and labels; a task-3 episode that scans only some boxes), e.g. scanroom6x5_sub4_L0:

    python scripts/testsuite/make_scanroom_suite.py --suite scanroom6x5_sub4_L0 --robot_dist 0.60 --lateral 0.40 \
        --targets C1R4,C2R4,C1R3,C2R3 --order rows

    python scripts/testsuite/make_scanroom_suite.py --suite scanroom6x5_r234_L0 --robot_dist 0.60 --lateral 0.0 \
        --targets C1R4,C2R4,C3R4,C4R4,C5R4,C6R4,C1R3,C2R3,C3R3,C4R3,C5R3,C6R3,C1R2,C2R2,C3R2,C4R2,C5R2,C6R2 \
        --order rows --rows_from bottom --all_columns --head_views upright --phase1_view obs_pose

  - "target_boxes": the GT box ids (evaluation only; the agent never sees box ids);
  - "scan_targets": the same boxes as the task instruction in robot terms, derived from the ids (C<col>R<row>, column 1
    = the robot's left, row 1 = the bottom row): {<columns>, <rows>, "order": ...}, with <columns> =
    "columns_from_left": [...] or, with --all_columns, "columns": "all" (the targets must then cover every column), and
    <rows> = "rows_from_top": [...] (--rows_from top, the default) or "rows_from_bottom": [...] (--rows_from bottom).
    The targets must form a full block of columns x rows (otherwise no such instruction exists). order: "rows" (rows
    from the top, left to right inside a row) or "columns" (the default task-3 order).
  - "head_views" (--head_views upright): the agent takes upright head views only (no lean-back looks); absent = the
    camera model's default. "phase1_view" (--phase1_view start_pose | obs_pose): with obs_pose the robot may walk (also
    back) to an observation pose before planning its scans; absent = start_pose for a subset task. Both are written only
    when given; the simulator does not read them (task fields for the agent, like "scan_targets").
  - "target" stays "all" (testsuite scenario.target_box_id: any other value would select the rightmost box).

Scene (assets/scenes/scan_room_20261117, from scan_room_same_size_20261117.zip / isaac_environment): a 6 x 5 wall of
30 boxes (0.20 m wide x 0.40 m deep x 0.25 m tall, calibrated), the striped carpet floor and the room wall behind, with
the photographed textures. It is referenced as the scenario's `scene_usd`, visual only; the scenario's boxes are the
physics and ground truth (same grid as the reconstruction's colliders) and have no visual of their own.

Labels: one per box at the position, size and in-plane rotation of the paper label photographed on that box
(labels_measured.json: the four paper corners per box, traced in the reconstruction's texture atlas). Each is replaced
by one of our labels: a white backed Code128 label of the photographed paper's size, position and rotation (rectangle
fitted to its four corners, label_rect), the code = label minus the backing border (factors.yaml
barcode.backing_border), content = random 9 characters of factors.yaml barcode.content_alphabet (seeded).

Frames: the reconstruction has the box faces in its y = 0 plane facing -y, columns along +x, z up. It is placed with
rotate_z -90 deg and translate (1.30, 0, 0): asset +x -> world -y, asset +y -> world +x, so the box fronts lie in the
world plane x = 1.30 (factors.yaml world.table_front_x) facing the robot, column 1 (C1) at world y = +0.5 (the robot's
left). Face coordinates of a -x face: u along world -y (= asset +x), v = up.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from g1_barcode_sim.testsuite import SUITES_DIR, load_factors  # noqa: E402

SUITE = "scanroom6x5_L0"
SCENE_DIR = "assets/scenes/scan_room_20261117"
COLS, ROWS = 6, 5
BOX_W, BOX_D, BOX_H = 0.20, 0.40, 0.25          # scene_regularized.usda boxDimensionsMeters / its collider cubes


def label_rect(pts_asset, col, row):
    """Rectangle fitted to a photographed paper label on the -x face: (u, v, rot_deg, w, h). The traced quads are
    rectangles up to ~1 mm (edges within 3.5 deg of the box axes, opposite edges within 4 mm in length): centre = mean
    of the corners, rotation = mean direction of the top and bottom edges, w / h = mean lengths of the opposite edges."""
    bx = (col - (COLS - 1) / 2) * BOX_W            # box centre, asset x
    bz = (row + 0.5) * BOX_H
    P = np.array([[p[0] - bx, p[2] - bz] for p in pts_asset])    # (u, v); corners TL, TR, BR, BL seen from the front
    top, bot = P[1] - P[0], P[2] - P[3]
    a = math.atan2(top[1] + bot[1], top[0] + bot[0])
    w = (np.linalg.norm(top) + np.linalg.norm(bot)) / 2
    h = (np.linalg.norm(P[3] - P[0]) + np.linalg.norm(P[2] - P[1])) / 2
    c = P.mean(0)
    return float(c[0]), float(c[1]), math.degrees(a), float(w), float(h)


def subset_spec(targets: list[str], order: str, rows_from: str = "top", columns: str = "list") -> dict:
    """Robot-terms target spec of a block of boxes C<col>R<row>: columns counted from the robot's left (C1 = world
    y +0.5 = left), rows from the top (rows_from "top": R5 = row 1) or from the bottom (rows_from "bottom": R1 = row 1).
    columns "list" names the columns ("columns_from_left"); columns "all" writes "columns": "all" and requires the
    targets to cover every column 1..COLS. Key order: columns, rows, order."""
    if rows_from not in ("top", "bottom"):
        raise ValueError(f"rows_from must be top | bottom, got {rows_from!r}")
    if columns not in ("list", "all"):
        raise ValueError(f"columns must be list | all, got {columns!r}")
    cells = []
    for t in targets:
        if not (t.startswith("C") and "R" in t):
            raise ValueError(f"target box id {t!r} is not C<col>R<row>")
        col, row = (int(v) for v in t[1:].split("R"))
        if not (1 <= col <= COLS and 1 <= row <= ROWS):
            raise ValueError(f"target box {t} outside the {COLS} x {ROWS} wall")
        cells.append((col, ROWS + 1 - row if rows_from == "top" else row))
    cols, rows = sorted({c for c, _ in cells}), sorted({r for _, r in cells})
    if len(set(cells)) != len(cells) or set(cells) != {(c, r) for c in cols for r in rows}:
        raise ValueError(f"targets {targets} are not a full block of columns x rows (no columns / rows instruction)")
    if order not in ("rows", "columns"):
        raise ValueError(f"order must be rows | columns, got {order!r}")
    if columns == "all" and cols != list(range(1, COLS + 1)):
        raise ValueError(f"targets {targets} cover columns {cols}, not all {COLS} columns (no \"all columns\" instruction)")
    col_part = {"columns": "all"} if columns == "all" else {"columns_from_left": cols}
    return {**col_part, f"rows_from_{rows_from}": rows, "order": order}


def task_note(targets: list[str], spec: dict, head_views: str | None, phase1_view: str | None) -> str:
    """The scenario note's task sentence, built from the spec parts (reproduces the 4-box suite's text exactly)."""
    cols = "all columns" if spec.get("columns") == "all" else f"columns {spec['columns_from_left']} from the left"
    edge = "top" if "rows_from_top" in spec else "bottom"
    upright = head_views == "upright"
    if phase1_view == "obs_pose":
        where = f"may walk back to an {'upright ' if upright else ''}observation pose"
    else:
        where = "works from near its start pose"
    views = " (no lean-back head views)" if upright else ""
    return (f" Task: scan only the labels of boxes {', '.join(targets)} ({cols}, rows {spec[f'rows_from_{edge}']} from "
            f"the {edge} of the wall), order {spec['order']}; the robot {where}{views}.")


def build(a) -> tuple[dict, dict]:
    """(scenario, manifest) of the suite for the parsed arguments (main writes them)."""
    F = load_factors()
    B = F["barcode"]
    labels = {x["id"]: x for x in json.load(open(os.path.join(ROOT, SCENE_DIR, "labels_measured.json")))}
    rng = np.random.default_rng(a.seed)
    alpha = list(B["content_alphabet"])
    suite = a.suite
    sid = f"{suite}_0000"
    boxes, codes = [], []
    for row in range(ROWS):                          # bottom row first: a stacked box rests on an earlier one
        for col in range(COLS):
            bid = f"C{col + 1}R{row + 1}"
            boxes.append({"id": bid, "size_wdh": [BOX_W, BOX_D, BOX_H], "appearance": "invisible", "setback": 0.0,
                          "y_center": round(((COLS - 1) / 2 - col) * BOX_W, 6),
                          "on": None if row == 0 else f"C{col + 1}R{row}"})
            L = labels[bid]
            if (L["row"], L["col"]) != (row, col):
                raise ValueError(f"labels_measured.json: {bid} has row/col {L['row']}/{L['col']}")
            u, v, rot, w, h = label_rect(L["points_asset_m"], col, row)
            pw = w - 2 * B["backing_border"]
            codes.append({"type": "code128", "width": round(pw, 5), "label_backing": True,
                          "aspect": round((h - 2 * B["backing_border"]) / pw, 5), "rot_deg": round(rot, 3),
                          "box_id": bid, "face": "-x", "u": round(u, 5), "v": round(v, 5),
                          "content": "".join(rng.choice(alpha, size=int(B["content_length"]["nominal"]))),
                          "content_id": f"SCANROOM-0000-{bid}",
                          "photo_label_size_m": [round(x, 5) for x in L["label_size_m"]]})
    fx = F["world"]["table_front_x"]
    start = f"root {a.robot_dist} m from the box faces"
    if a.lateral:
        start += f", {a.lateral} m to the left of the wall centre (world y = {a.lateral})"
    params = {"robot_dist": a.robot_dist, "seed": a.seed}
    targets = [t.strip() for t in a.targets.split(",")] if a.targets else None
    note = ""
    if targets:
        spec = subset_spec(targets, a.order, rows_from=a.rows_from, columns="all" if a.all_columns else "list")
        params.update(lateral=a.lateral, targets=targets, order=a.order)
        # the new options enter params only when they differ from the defaults: the existing suites stay byte-identical
        if a.rows_from != "top":
            params["rows_from"] = a.rows_from
        if a.all_columns:
            params["columns"] = "all"
        for k in ("head_views", "phase1_view"):
            if getattr(a, k) is not None:
                params[k] = getattr(a, k)
        note = task_note(targets, spec, a.head_views, a.phase1_view)
    sc = {"id": sid, "suite": suite, "task": 3, "seed": a.seed, "index": 0, "level": "L0", "factors_varied": [],
          "factor": {"note": f"scan room reconstruction (2026-11-17) with our labels: 6 x 5 boxes {BOX_W} x {BOX_D} x {BOX_H} m "
                             f"(w x d x h), front x = {fx} m, centred on y = 0; one backed code128 label per box at the "
                             f"photographed paper label (position, size, rotation). Robot start: facing the wall squarely, "
                             f"{start}, yaw 0 (scripts/testsuite/make_scanroom_suite.py).{note}",
                     "params": params},
          "background": {"type": "empty", "light_scale": 1.0},
          "scene_usd": {"usd": f"{SCENE_DIR}/scene.usda", "translate": [fx, 0.0, 0.0], "rotate_z_deg": -90.0},
          # the photographed paper labels reach the top and left edges of their boxes, and the fitted rectangle of a slightly
          # rotated paper can stand < 1 mm beyond them (factors.yaml margin 0.01 would reject them); -2 mm as in the
          # scene's own adapter (scan_room_same_size_20261117/scene_adapter.py)
          "label_margin": -0.002,
          "table": None, "robot_init": {"dist_to_table": a.robot_dist, "lateral_offset": a.lateral, "yaw_deg": 0.0},
          "boxes": boxes, "barcodes": codes, "target": "all", "task2_start": None}
    if targets:
        # GT box ids (evaluation only) and the same targets in robot terms (the task instruction the agent gets)
        sc["target_boxes"] = targets
        sc["scan_targets"] = spec
        for k in ("head_views", "phase1_view"):           # agent task fields, written only when given
            if getattr(a, k) is not None:
                sc[k] = getattr(a, k)
    from g1_barcode_sim.testsuite.scenario import to_layout
    to_layout(sc, F)                                  # validates boxes, stacking, label fit and overlaps
    man = {"suite": suite, "base_suite": suite, "version": 1, "seed": a.seed, "task": 3, "level": "L0", "count": 1,
           "definition": {"note": "hand-specified (scripts/testsuite/make_scanroom_suite.py)"}, "rejections": {},
           "warnings": [], "skipped_infeasible": [], "code_hash": None, "head_camera_nominal": None, "scenario_ids": [sid]}
    return sc, man


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=SUITE, help="suite name (scenario id = <suite>_0000)")
    ap.add_argument("--robot_dist", type=float, default=0.65, help="root distance from the box faces (m)")
    ap.add_argument("--lateral", type=float, default=0.0, help="root start y (m, + = the robot's left of the wall centre)")
    ap.add_argument("--targets", default=None, help="comma-separated box ids C<col>R<row> to scan (subset task)")
    ap.add_argument("--order", default="columns", choices=["columns", "rows"], help="scan order of a subset task")
    ap.add_argument("--rows_from", default="top", choices=["top", "bottom"],
                    help="subset task: count the target rows from the top (default) or from the bottom of the wall")
    ap.add_argument("--all_columns", action="store_true",
                    help='subset task: write "columns": "all" (the targets must cover every column)')
    ap.add_argument("--head_views", default=None, choices=["upright"],
                    help="subset task: scenario head_views (upright = no lean-back head views); absent = camera default")
    ap.add_argument("--phase1_view", default=None, choices=["start_pose", "obs_pose"],
                    help="subset task: scenario phase1_view (obs_pose = the robot may walk to an observation pose)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if not a.targets:
        given = [f"--{k}" for k, v in (("order", a.order != "columns"), ("rows_from", a.rows_from != "top"),
                                       ("all_columns", a.all_columns), ("head_views", a.head_views is not None),
                                       ("phase1_view", a.phase1_view is not None)) if v]
        if given:
            ap.error(f"{', '.join(given)} appl{'ies' if len(given) == 1 else 'y'} to a subset task (--targets)")
    return a


def main():
    a = parse_args()
    sc, man = build(a)
    d = os.path.join(SUITES_DIR, a.suite)
    sid, codes, B = sc["id"], sc["barcodes"], load_factors()["barcode"]
    os.makedirs(os.path.join(d, "scenarios"), exist_ok=True)
    json.dump(sc, open(os.path.join(d, "scenarios", f"{sid}.json"), "w"), indent=1)
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1)
    sizes = np.array([[c["width"] + 2 * B["backing_border"], c["width"] * c["aspect"] + 2 * B["backing_border"]] for c in codes])
    photo = np.array([c["photo_label_size_m"] for c in codes])
    print(f"wrote {sid}: {len(sc['boxes'])} boxes, {len(codes)} labels; label w {sizes[:, 0].min():.4f}-{sizes[:, 0].max():.4f} m, "
          f"h {sizes[:, 1].min():.4f}-{sizes[:, 1].max():.4f} m (photographed paper w {photo[:, 0].min():.4f}-{photo[:, 0].max():.4f}, "
          f"h {photo[:, 1].min():.4f}-{photo[:, 1].max():.4f}); |rot| max {max(abs(c['rot_deg']) for c in codes):.2f} deg")


if __name__ == "__main__":
    main()
