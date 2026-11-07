"""Handheld barcode scanner held by the left Dex1 gripper, with a camera at the scanner head.

Frames (configs/scanner.yaml, robot.yaml `scanner_mount`):
  hand -> G (scanner_mount, grip point)  -> tip (exit-window centre)  -> cam (ROS optical frame)
The gun is a rigid, visual-only child of the hand link, so it follows the link exactly. It has no
collider and no mass; its parts are added to the robot clearance model so `step()` reports a
"collision" when the gun runs into the scene. The left gripper is commanded to close on the
handle faces (joint target computed from the live finger-pad geometry, `grip_joint_target`).
"""
from __future__ import annotations

from dataclasses import MISSING

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

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


# ------------------------------------------------------------------ USD spawner (runs before the sim starts)
@clone
def _spawn_scanner(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import omni.usd
    from pxr import Gf, UsdGeom
    from .scene_builder import _cuboid
    stage = omni.usd.get_context().get_stage()
    xf = UsdGeom.Xform.Define(stage, prim_path)   # = scanner_mount
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    w, x, y, z = (float(v) for v in orientation)
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))
    for p in cfg.parts:
        # visual only: no collider (a collider under the link would become part of the articulation's link)
        _cuboid(stage, f"{prim_path}/{p['name']}", p["center"], p["size"], color=p["color"], collider=False)
    return xf.GetPrim()


@configclass
class ScannerSpawnerCfg(sim_utils.SpawnerCfg):
    """parts = [] spawns only the empty scanner_mount Xform (scanner disabled)."""
    func = _spawn_scanner
    parts: list = MISSING


# ------------------------------------------------------------------ gripper hold
def _pad_x(robot, collision_model, side, pad_links):
    """x_hand coordinates of every pad collision vertex per finger (live poses), and R_hand_world."""
    from .camera import inv_T
    bodies = list(robot.data.body_names)
    P = robot.data.body_link_pose_w[0].cpu().numpy()
    hb = bodies.index(f"{side}_hand_base_link")
    T_hw = inv_T(make_T(P[hb, :3], P[hb, 3:]))
    W = collision_model.world_points()
    x = {}
    for finger, links in pad_links.items():
        m = np.isin(collision_model.owner, [bodies.index(ln.format(side=side)) for ln in links])
        if not m.any():
            raise RuntimeError(f"no collision geometry on pad links {links}")
        x[finger] = ((T_hw[:3, :3] @ W[m].T).T + T_hw[:3, 3])[:, 0]
    return x, T_hw[:3, :3]


def pad_gap(robot, collision_model, side: str, pad_links: dict) -> float:
    """Current distance between the two pads' inner faces along x_hand (m). finger1 is on +x_hand."""
    x, _ = _pad_x(robot, collision_model, side, pad_links)
    return float(x["finger1"].min() - x["finger2"].max())


def grip_joint_target(robot, collision_model, side: str, pad_links: dict, joints: list, width: float) -> float:
    """Gripper joint target that puts the inner faces of the two finger pads `width` apart.

    Each pad's inner face is its extreme collision vertex along x_hand (the closing direction); how a face
    moves per unit joint motion comes from the PhysX jacobian of the finger link (exact for prismatic
    fingers). Both finger joints receive the same target (as the upstream gripper command does), so q solves
        face1(q1) - face2(q2) = width,  face_i(q) = face_i_now + s_i (q - q_i_now).
    """
    x, R_hw = _pad_x(robot, collision_model, side, pad_links)
    face1, face2 = x["finger1"].min(), x["finger2"].max()
    bodies, names = list(robot.data.body_names), list(robot.data.joint_names)
    jidx = [names.index(j) for j in joints]
    q = robot.data.joint_pos[0, jidx].cpu().numpy()
    J = robot.root_physx_view.get_jacobians()[0].cpu().numpy()
    s = []
    for finger, j in zip(("finger1", "finger2"), jidx):
        b = bodies.index(pad_links[finger][0].format(side=side))
        s.append(float((R_hw @ J[b, :3, j + 6])[0]))   # floating base: joint j is jacobian column j + 6
    if abs(s[0] - s[1]) < 1e-6:
        raise RuntimeError(f"finger joints do not move the pads apart (dx/dq = {s})")
    return float((width - face1 + face2 + s[0] * q[0] - s[1] * q[1]) / (s[0] - s[1]))
