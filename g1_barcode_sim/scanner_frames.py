"""Fixed frames of the handheld scanner (configs/scanner.yaml, robot file `scanner_mount`), pure numpy.

Split out of scanner.py (which imports Isaac Lab for the USD spawner) so code without the Isaac Sim app can build the
same transforms, e.g. g1_agent_system scripts/export_urdf.py --source usd (offline URDF export)."""
from __future__ import annotations

import numpy as np

from .camera import make_T

# ROS optical axes in the G / tip frame: +x_opt (image right) = -y_G, +y_opt (image down) = -z_G,
# +z_opt (viewing direction) = +x_G (scan direction). Columns of R_G_opt.
R_G_OPT = np.array([[0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0]])


def transforms(mount_cfg: dict, scfg: dict) -> dict:
    """Fixed transforms of the scanner chain, all relative to the hand link."""
    T_hand_G = make_T(mount_cfg["pos"], mount_cfg["quat_wxyz"])
    T_G_tip = np.eye(4); T_G_tip[:3, 3] = scfg["tip"]["pos"]
    T_tip_cam = np.eye(4); T_tip_cam[:3, :3] = R_G_OPT; T_tip_cam[:3, 3] = scfg["camera"]["pos_in_tip"]
    T_hand_tip = T_hand_G @ T_G_tip
    return {"T_hand_G": T_hand_G, "T_hand_tip": T_hand_tip, "T_hand_cam": T_hand_tip @ T_tip_cam}


def part_corners_G(scfg: dict) -> np.ndarray:
    """8 corners of every gun part (G frame), for the clearance model."""
    out = []
    for p in scfg["parts"]:
        c, h = np.asarray(p["center"], float), np.asarray(p["size"], float) / 2
        out += [c + h * np.array([sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return np.array(out).reshape(-1, 3)
