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


def load_robot_cfg(overrides: dict | None = None) -> dict:
    return deep_merge(_load(os.path.join(CFG_DIR, "robot.yaml")), overrides)


def load_camera_cfg(overrides: dict | None = None) -> dict:
    return deep_merge(_load(os.path.join(CFG_DIR, "camera_head.yaml")), overrides)


def load_scanner_cfg(overrides: dict | None = None) -> dict:
    return deep_merge(_load(os.path.join(CFG_DIR, "scanner.yaml")), overrides)


def load_all(scene: str, overrides: dict | None = None) -> dict:
    """overrides: {"scene": {...}, "barcode": {...}, "robot": {...}, "camera": {...}, "scanner": {...}}"""
    o = overrides or {}
    return {"scene": load_scene_cfg(scene, o.get("scene")), "barcode": load_barcode_cfg(o.get("barcode")),
            "robot": load_robot_cfg(o.get("robot")), "camera": load_camera_cfg(o.get("camera")),
            "scanner": load_scanner_cfg(o.get("scanner"))}
