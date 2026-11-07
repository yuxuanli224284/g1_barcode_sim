"""Expand a scene YAML into concrete geometry: furniture parts, boxes and barcode placements.

Pure Python / numpy (no Isaac). Everything is axis-aligned except box yaw. The same layout is used
to author the USD (scene_builder.py) and to compute ground truth (ground_truth.py), and the check
script verifies the authored USD transforms agree with it.
"""
from __future__ import annotations

import math

import numpy as np

from .barcode import barcode_content, placement_in_box

SIDE_TO_FACE = {"front": "-x", "back": "+x", "left": "+y", "right": "-y"}


def yaw_quat_wxyz(yaw: float):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _part(name, center, size, color, roughness, sem):
    return {"name": name, "center": [float(x) for x in center], "size": [float(x) for x in size],
            "color": list(color), "roughness": roughness, "semantic": sem}


def _table(f):
    d, w, h = f["size"]
    t, leg = f["top_thickness"], f["leg_size"]
    x0, cy = f["front_x"], f["center_y"]
    parts = [_part("top", (x0 + d / 2, cy, h - t / 2), (d, w, t), f["color"], f["roughness"], "table")]
    for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        parts.append(_part(f"leg{i}", (x0 + d / 2 + sx * (d / 2 - leg / 2), cy + sy * (w / 2 - leg / 2), (h - t) / 2),
                           (leg, leg, h - t), f["color"], f["roughness"], "table"))
    return {"id": f["id"], "kind": "table", "parts": parts, "support_z": [h], "front_x": x0, "depth": d,
            "y_range": [cy - w / 2, cy + w / 2], "clearance": [None]}


def _shelf(f):
    d, w, H = f["size"]
    t, p = f["board_thickness"], f["post_size"]
    x0, cy = f["front_x"], f["center_y"]
    levels = list(f["level_heights"])
    parts = [_part(f"board{i}", (x0 + d / 2, cy, z - t / 2), (d, w, t), f["color"], f["roughness"], "shelf")
             for i, z in enumerate(levels)]
    for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        parts.append(_part(f"post{i}", (x0 + d / 2 + sx * (d / 2 - p / 2), cy + sy * (w / 2 - p / 2), H / 2),
                           (p, p, H), f["color"], f["roughness"], "shelf"))
    clearance = [levels[i + 1] - t - levels[i] for i in range(len(levels) - 1)] + [H - levels[-1]]
    return {"id": f["id"], "kind": "shelf", "parts": parts, "support_z": levels, "front_x": x0, "depth": d,
            "y_range": [cy - w / 2 + p, cy + w / 2 - p], "clearance": clearance}


def _place_on(box, furn):
    on = box["place_on"]
    level = on.get("level", 0)
    bx, by, bz = box["size"]
    if on["front_setback"] + bx > furn["depth"] + 1e-9:
        raise ValueError(f"{box['id']}: depth {bx} + setback does not fit {furn['id']}")
    lo, hi = on["y"] - by / 2, on["y"] + by / 2
    if lo < furn["y_range"][0] - 1e-9 or hi > furn["y_range"][1] + 1e-9:
        raise ValueError(f"{box['id']}: y span [{lo:.3f},{hi:.3f}] outside {furn['id']} {furn['y_range']}")
    clr = furn["clearance"][level]
    if clr is not None and bz > clr + 1e-9:
        raise ValueError(f"{box['id']}: height {bz} exceeds clearance {clr:.3f} on {furn['id']} level {level}")
    z0 = furn["support_z"][level]
    return [furn["front_x"] + on["front_setback"] + bx / 2, on["y"], z0 + bz / 2]


def expand_scene(scfg: dict, bcfg: dict, seed: int = 0, randomize: bool | None = None) -> dict:
    sid = scfg["scene_id"]
    furniture = []
    for f in scfg.get("furniture", []):
        furniture.append({"table": _table, "shelf": _shelf}[f["kind"]](f))
    fmap = {f["id"]: f for f in furniture}

    boxes = []
    for b in scfg.get("boxes", []):
        pos = _place_on(b, fmap[b["place_on"]["furniture"]]) if "place_on" in b else list(b["pos"])
        boxes.append({"id": b["id"], "size": list(b["size"]), "pos": pos, "place_on": b.get("place_on"), "yaw": math.radians(b.get("yaw_deg", 0.0)),
                      "barcodes": b.get("barcodes", []), "distractor": bool(b.get("distractor", False))})
    for st in scfg.get("stacks", []):
        nx, ny, nz = st["counts"]
        s, g = st["box_size"], st["gap"]
        pitch = s + g
        sides = {SIDE_TO_FACE[x] for x in st["barcode_sides"]}
        k = len(boxes)
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    k += 1
                    faces = []
                    if ix == 0 and "-x" in sides: faces.append({"face": "-x"})
                    if ix == nx - 1 and "+x" in sides: faces.append({"face": "+x"})
                    if iy == 0 and "-y" in sides: faces.append({"face": "-y"})
                    if iy == ny - 1 and "+y" in sides: faces.append({"face": "+y"})
                    boxes.append({"id": f"B{k:02d}", "size": [s, s, s],
                                  "pos": [st["front_x"] + s / 2 + ix * pitch, st["center_y"] + (iy - (ny - 1) / 2) * pitch, s / 2 + iz * pitch],
                                  "yaw": 0.0, "barcodes": faces, "distractor": False, "stack": st["id"]})

    rnd = bcfg["randomize"]
    do_rand = rnd["enabled"] if randomize is None else randomize
    rng = np.random.default_rng(seed)
    btype = bcfg["type"]
    lw, lh = bcfg["label_size"][btype]
    barcodes = []
    for b in boxes:
        num = int(b["id"][1:])
        R = rot_z(b["yaw"])
        for spec in b["barcodes"]:
            uv = list(spec.get("uv", [0.0, 0.0]))
            rot = float(spec.get("rot_deg", 0.0))
            if do_rand:
                uv = [uv[0] + rng.uniform(-rnd["offset_uv"], rnd["offset_uv"]), uv[1] + rng.uniform(-rnd["offset_uv"], rnd["offset_uv"])]
                rot = rot + rng.uniform(-rnd["rot_deg"], rnd["rot_deg"])
            c, n, r, u = placement_in_box(b["size"], spec["face"], uv, rot, bcfg["face_offset"])
            # the rotated label must stay inside the face
            n_abs = np.abs(n)
            face_dims = [d for d, a in zip(b["size"], n_abs) if a < 0.5]
            corners = [c + sx * lw / 2 * r + sy * lh / 2 * u for sx in (-1, 1) for sy in (-1, 1)]
            half = np.asarray(b["size"]) / 2
            for q in corners:
                if np.any((np.abs(q) - half)[n_abs < 0.5] > 1e-6):
                    raise ValueError(f"{b['id']} face {spec['face']}: label leaves the face (uv={uv}, rot={rot:.1f}, face={face_dims})")
            content = barcode_content(bcfg["content_format"], sid, num, spec["face"], bcfg["face_codes"])
            wc = np.asarray(b["pos"]) + R @ c
            barcodes.append({"content": content, "type": btype, "box_id": b["id"], "face": spec["face"],
                             "uv": [float(x) for x in uv], "rot_deg": rot, "distractor": b["distractor"],
                             "local": {"center": c.tolist(), "normal": n.tolist(), "right": r.tolist(), "up": u.tolist()},
                             "center": wc.tolist(), "normal": (R @ n).tolist(), "up": (R @ u).tolist(),
                             "right": (R @ r).tolist(), "size": [lw, lh]})
    contents = [x["content"] for x in barcodes]
    if len(set(contents)) != len(contents):
        raise ValueError("barcode contents are not unique")
    return {"scene": scfg["name"], "scene_id": sid, "seed": seed, "randomized": bool(do_rand),
            "furniture": furniture, "boxes": boxes, "barcodes": barcodes, "robot_init": scfg["robot_init"]}
