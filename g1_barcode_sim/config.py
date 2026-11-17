"""YAML config loading: configs/scenes/_common.yaml <- scene file <- overrides (deep merge)."""
from __future__ import annotations

import copy
import os

import yaml

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_DIR = os.path.join(PKG_ROOT, "configs")
SCENES = ("s1_table", "s2_shelf", "s3_stack")


def deep_merge(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_scene_cfg(scene: str, overrides: dict | None = None) -> dict:
    if scene not in SCENES and not os.path.exists(os.path.join(CFG_DIR, "scenes", f"{scene}.yaml")):
        raise ValueError(f"unknown scene {scene!r}; known: {SCENES}")
    cfg = deep_merge(_load(os.path.join(CFG_DIR, "scenes", "_common.yaml")), _load(os.path.join(CFG_DIR, "scenes", f"{scene}.yaml")))
    return deep_merge(cfg, overrides)


def load_barcode_cfg(overrides: dict | None = None) -> dict:
    return deep_merge(_load(os.path.join(CFG_DIR, "barcode.yaml")), overrides)


ROBOT_FILES = {"g1": "robot.yaml", "t800": "robot_t800.yaml"}


def load_robot_cfg(overrides: dict | None = None) -> dict:
    """The robot is chosen per call, never by editing a config file: overrides {"model": "g1" | "t800"} (default g1)
    selects the robot file, g1 -> robot.yaml, t800 -> robot_t800.yaml (EngineAI T800 Pro, loaded instead of
    robot.yaml); the overrides are merged over that file. Each file declares its own `model` and a mismatch raises, so
    robot.yaml stays the G1 file: callers that do not pass a model (all G1 code, tests and scripts) always get G1."""
    model = (overrides or {}).get("model", "g1")
    if model not in ROBOT_FILES:
        raise ValueError(f"robot model {model!r} unknown; known: {sorted(ROBOT_FILES)}")
    base = _load(os.path.join(CFG_DIR, ROBOT_FILES[model]))
    if base.get("model") != model:
        raise ValueError(f"configs/{ROBOT_FILES[model]} must declare model: {model} (got {base.get('model')!r}); select "
                         f"another robot with overrides {{'model': ...}} instead of editing the file")
    return deep_merge(base, overrides)


def head_camera_mount(robot_cfg: dict) -> dict | None:
    """Camera-cfg entries that place the head camera on a robot whose robot file has a `head_camera` block
    (T800; G1 has none and keeps camera_head.yaml as is). Intrinsics / resolution / depth stay those of
    camera_head.yaml. The mount is the product of the block's URDF fixed-joint `chain` (xyz / rpy, parent -> child)
    starting at `parent_link`; its last frame must be an optical frame (`chain_frame: ros`)."""
    hc = robot_cfg.get("head_camera")
    if hc is None:
        return None
    import numpy as np
    from .camera import R_to_quat_wxyz, T_from_xyz_rpy
    if hc["chain_frame"] != "ros":
        raise ValueError(f"head_camera.chain_frame must be 'ros' (optical frame), got {hc['chain_frame']!r}")
    T = np.eye(4)
    for j in hc["chain"]:
        T = T @ T_from_xyz_rpy(j["xyz"], j["rpy"])
    return {"prim_parent_link": hc["parent_link"], "offset_pos": T[:3, 3].tolist(),
            "offset_rot_wxyz": R_to_quat_wxyz(T[:3, :3]), "offset_convention": "ros",
            "mount": {"parent_link": hc["parent_link"], "chain": [j["joint"] for j in hc["chain"]],
                      "T_link_cam": np.round(T, 6).tolist()}}


def load_camera_cfg(overrides: dict | None = None) -> dict:
    return deep_merge(_load(os.path.join(CFG_DIR, "camera_head.yaml")), overrides)


def load_scanner_cfg(overrides: dict | None = None) -> dict:
    return deep_merge(_load(os.path.join(CFG_DIR, "scanner.yaml")), overrides)


def load_all(scene: str, overrides: dict | None = None) -> dict:
    """overrides: {"scene": {...}, "barcode": {...}, "robot": {...}, "camera": {...}, "scanner": {...}}"""
    o = overrides or {}
    robot = load_robot_cfg(o.get("robot"))
    camera = load_camera_cfg(o.get("camera"))
    mount = head_camera_mount(robot)
    if mount is not None:          # robot-specific mount (T800); the camera overrides still win
        camera = deep_merge(deep_merge(camera, mount), o.get("camera"))
    return {"scene": load_scene_cfg(scene, o.get("scene")), "barcode": load_barcode_cfg(o.get("barcode")),
            "robot": robot, "camera": camera, "scanner": load_scanner_cfg(o.get("scanner"))}
