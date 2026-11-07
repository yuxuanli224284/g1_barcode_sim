"""Scenario JSON -> layout dict in the format of layout.expand_scene (furniture parts, boxes,
barcodes with box-local and world placement, robot_init), so the simulator, GT export, clearance model and
visibility code work unchanged. Pure numpy.

Conventions
- boxes[i].size_wdh = [w (along y), d (along x), h]; layout box size = [d, w, h].
- barcode face: "-x" (default, facing the robot start; legacy suites have no "face"), "+x", "-y" or "+y".
- barcode u: along the face's "right" axis as seen from in front of the face (barcode.FACE_AXES; for the -x face: -y,
  i.e. the robot's right); v: up. Both measured from the face centre, metres. rot_deg: counter-clockwise seen from in
  front. The label (rotated extent + factors barcode.margin) must fit inside its face, and no other box may reach into
  the slab face_offset + label_thickness in front of it (an inner face of a stack: the label would be unreadable).
- label size = pattern size + 2 * backing_border on each side when label_backing, else the pattern size.
"""
from __future__ import annotations

import math

import numpy as np

from ..barcode import placement_in_box
from ..config import load_barcode_cfg
from .geometry import TOUR_FACES, _face_frame, blockers

TABLE_COLOR = [0.92, 0.92, 0.92]       # white matte (fixed in L0-L2)
TABLE_ROUGHNESS = 0.9
# USD prim name of a label per face (scenario_builder), so one box can carry labels on several faces
LABEL_PRIM = {"-x": "BC_mx", "+x": "BC_px", "-y": "BC_my", "+y": "BC_py"}


def pattern_size(code: dict) -> tuple[float, float]:
    if code["type"] == "qr":
        return float(code["width"]), float(code["width"])
    return float(code["width"]), float(code["width"]) * float(code["aspect"])


def label_size(code: dict, border: float) -> tuple[float, float]:
    pw, ph = pattern_size(code)
    if code["label_backing"]:
        return pw + 2 * border, ph + 2 * border
    return pw, ph


def table_parts(t: dict, front_x: float, top_thickness: float, leg: float):
    d, w, h = t["depth"], t["width"], t["height"]
    parts = [{"name": "top", "center": [front_x + d / 2, 0.0, h - top_thickness / 2], "size": [d, w, top_thickness],
              "color": TABLE_COLOR, "roughness": TABLE_ROUGHNESS, "semantic": "table"}]
    for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        parts.append({"name": f"leg{i}", "center": [front_x + d / 2 + sx * (d / 2 - leg / 2), sy * (w / 2 - leg / 2), (h - top_thickness) / 2],
                      "size": [leg, leg, h - top_thickness], "color": TABLE_COLOR, "roughness": TABLE_ROUGHNESS, "semantic": "table"})
    return parts


def rotated_half_extent(lw: float, lh: float, rot_deg: float) -> tuple[float, float]:
    """Half extents along the face's u / v axes of a lw x lh label rotated by rot_deg in the face plane."""
    a = math.radians(rot_deg)
    return (lw * abs(math.cos(a)) + lh * abs(math.sin(a))) / 2, (lw * abs(math.sin(a)) + lh * abs(math.cos(a))) / 2


def to_layout(sc: dict, fcfg: dict, label_thickness: float | None = None) -> dict:
    """fcfg: configs/testsuite/factors.yaml. label_thickness: label plate thickness (default configs/barcode.yaml, the
    value scenario_builder authors the plate with)."""
    if label_thickness is None:
        label_thickness = load_barcode_cfg()["label_thickness"]
    fx = fcfg["world"]["table_front_x"]
    t = sc["table"]
    tcfg = fcfg["table"]
    if t is None:                       # boxes on the floor; fx is then the front plane the setbacks are measured from
        furniture, support_z = [], 0.0
    else:
        furniture = [{"id": "T0", "kind": "table", "parts": table_parts(t, fx, tcfg["top_thickness"], tcfg["leg_size"]),
                      "support_z": [t["height"]], "front_x": fx, "depth": t["depth"], "y_range": [-t["width"] / 2, t["width"] / 2],
                      "clearance": [None]}]
        support_z = t["height"]
    boxes = {}
    for b in sc["boxes"]:
        w, d, h = b["size_wdh"]
        x, y = fx + b["setback"] + d / 2, b["y_center"]
        on = b.get("on")
        if on is None:
            z0 = support_z
        else:                           # stacked: rests on the top face of an earlier box (static colliders, no gap)
            if on not in boxes:
                raise ValueError(f"box {b['id']} is on {on!r}, which is not an earlier box of the scenario")
            lo = boxes[on]
            ld, lw, lh = lo["size"]
            if abs(x - lo["pos"][0]) + d / 2 > ld / 2 + 1e-9 or abs(y - lo["pos"][1]) + w / 2 > lw / 2 + 1e-9:
                raise ValueError(f"box {b['id']} overhangs the box {on} it rests on")
            z0 = lo["pos"][2] + lh / 2
        boxes[b["id"]] = {"id": b["id"], "size": [d, w, h], "pos": [x, y, z0 + h / 2],
                          "yaw": 0.0, "appearance": b["appearance"], "distractor": False, "barcodes": []}
    border = fcfg["barcode"]["backing_border"]
    face_offset, margin = fcfg["barcode"]["face_offset"], fcfg["barcode"]["margin"]
    wdh = {b["id"]: b["size_wdh"] for b in sc["boxes"]}
    barcodes = []
    on_face = {}                        # (box, face) -> [(u, v, eu, ev)] of the labels already placed there
    for code in sc["barcodes"]:
        b = boxes[code["box_id"]]
        face = code.get("face", "-x")
        if face not in TOUR_FACES:
            raise ValueError(f"{sc['id']} {code['content']}: face {face!r} is not one of {TOUR_FACES}")
        lw, lh = label_size(code, border)
        # the label must fit inside its face (same rule as the sampler: rotated extent + margin on every side; 1e-9 m
        # float slack as the overhang test above, so a label exactly at the margin is accepted)
        w, d, h = wdh[code["box_id"]]
        fw = w if face in ("-x", "+x") else d
        eu, ev = rotated_half_extent(lw, lh, code["rot_deg"])
        if abs(code["u"]) + eu > fw / 2 - margin + 1e-9 or abs(code["v"]) + ev > h / 2 - margin + 1e-9:
            raise ValueError(f"{sc['id']} {code['content']}: label (half extent {eu:.4f} x {ev:.4f} at u={code['u']}, "
                             f"v={code['v']}) does not fit the {face} face ({fw} x {h}) of box {code['box_id']} with the "
                             f"{margin} m margin")
        # several labels may share a box face; their (rotated, axis-aligned) extents must not overlap
        for (u0, v0, eu0, ev0) in on_face.get((code["box_id"], face), []):
            if abs(code["u"] - u0) < eu + eu0 - 1e-9 and abs(code["v"] - v0) < ev + ev0 - 1e-9:
                raise ValueError(f"{sc['id']} {code['content']}: label at u={code['u']}, v={code['v']} overlaps another "
                                 f"label on the {face} face of box {code['box_id']} (at u={u0}, v={v0})")
        on_face.setdefault((code["box_id"], face), []).append((code["u"], code["v"], eu, ev))
        c, n, r, u = placement_in_box(b["size"], face, (code["u"], code["v"]), code["rot_deg"], face_offset)
        wc = np.asarray(b["pos"]) + c
        # the label must not sit on an inner face: no other box may reach into the slab it occupies
        _, fr, _ = _face_frame(face)
        lat = np.abs(fr)
        plane = float(n @ (np.asarray(b["pos"]) + n * (np.abs(n) @ np.asarray(b["size"]) / 2)))
        lo, hi = np.array([lat @ wc - eu, wc[2] - ev]), np.array([lat @ wc + eu, wc[2] + ev])
        cov = blockers(list(boxes.values()), face, plane, lo, hi, face_offset + label_thickness, exclude=code["box_id"])
        if cov:
            raise ValueError(f"{sc['id']} {code['content']}: the {face} face of box {code['box_id']} is covered by another "
                             f"box where the label sits (inner face of the stack; the label would be unreadable)")
        barcodes.append({"content": code["content"], "type": code["type"], "box_id": code["box_id"], "face": face,
                         "uv": [code["u"], code["v"]], "rot_deg": code["rot_deg"], "distractor": False,
                         "local": {"center": c.tolist(), "normal": n.tolist(), "right": r.tolist(), "up": u.tolist()},
                         "center": wc.tolist(), "normal": n.tolist(), "up": u.tolist(), "right": r.tolist(),
                         "size": [lw, lh], "pattern_size": list(pattern_size(code)), "label_backing": code["label_backing"],
                         "code": code})
    ri = sc["robot_init"]
    if sc.get("task2_start"):
        p = sc["task2_start"]["pose_world"]
        robot_init = {"pos": [p["x"], p["y"]], "yaw": math.radians(p["yaw_deg"])}
    else:
        robot_init = {"pos": [fx - ri["dist_to_table"], ri["lateral_offset"]], "yaw": math.radians(ri["yaw_deg"])}
    return {"scene": sc["id"], "scene_id": sc["id"], "seed": sc["seed"], "randomized": sc["level"] != "L0",
            "furniture": furniture, "boxes": list(boxes.values()), "barcodes": barcodes, "robot_init": robot_init,
            "background": sc["background"], "target_box_id": target_box_id(sc), "scenario": sc}


def target_box_id(sc: dict) -> str:
    """"left"/"right" from the robot's view (it faces +x, so left = larger y); "all" (task 3: every box) is kept."""
    if sc["target"] == "all":
        return "all"
    ys = sorted(sc["boxes"], key=lambda b: b["y_center"])
    return (ys[-1] if sc["target"] == "left" else ys[0])["id"]
