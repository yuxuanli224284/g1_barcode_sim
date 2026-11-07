"""Ground-truth export and the reference standing pose for a barcode.

World frame, metres, quaternions wxyz. `target_root_pose(center, normal, d)`:
  n_h = normalize(normal projected on the horizontal plane); pos = center_xy + d * n_h (z = 0);
  yaw faces the label, i.e. along -n_h.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .layout import yaw_quat_wxyz


def target_root_pose(center, normal, d: float = 0.20) -> dict:
    n = np.asarray(normal, dtype=float)
    nh = np.array([n[0], n[1], 0.0])
    norm = np.linalg.norm(nh)
    if norm < 1e-9:
        raise ValueError("barcode normal is vertical; no horizontal standing direction")
    nh /= norm
    c = np.asarray(center, dtype=float)
    pos = [c[0] + d * nh[0], c[1] + d * nh[1], 0.0]
    return {"pos": [float(x) for x in pos], "yaw": float(math.atan2(-nh[1], -nh[0]))}


def build_gt(layout: dict, d: float = 0.20, robot_init: dict | None = None) -> dict:
    ri = robot_init or layout["robot_init"]
    key = f"target_root_pose_d{int(round(d * 100)):03d}"
    gt = {
        "scene": layout["scene"], "seed": layout["seed"], "randomized": layout["randomized"],
        "robot_init": {"pos": [float(ri["pos"][0]), float(ri["pos"][1]), 0.0], "yaw": float(ri["yaw"])},
        "boxes": [{"id": b["id"], "size": b["size"], "pos": [float(x) for x in b["pos"]], "quat_wxyz": yaw_quat_wxyz(b["yaw"]),
                   "distractor": b["distractor"]} for b in layout["boxes"]],
        "barcodes": [],
    }
    for bc in layout["barcodes"]:
        gt["barcodes"].append({
            "content": bc["content"], "type": bc["type"], "box_id": bc["box_id"], "face": bc["face"],
            "center": bc["center"], "normal": bc["normal"], "up": bc["up"], "size": bc["size"],
            # drawn symbol (bars / modules) without the white backing; None for layouts that do not record it
            "pattern_size": bc.get("pattern_size"),
            "height_above_ground": float(bc["center"][2]),
            "distractor": bc["distractor"],
            key: target_root_pose(bc["center"], bc["normal"], d),
        })
    return gt


def save_gt(gt: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(gt, f, indent=1)
