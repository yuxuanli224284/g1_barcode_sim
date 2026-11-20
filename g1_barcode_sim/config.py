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
    from .camera import resolve_head_model
    return resolve_head_model(deep_merge(_load(os.path.join(CFG_DIR, "camera_head.yaml")), overrides))


def load_scanner_cfg(overrides: dict | None = None) -> dict:
    return deep_merge(_load(os.path.join(CFG_DIR, "scanner.yaml")), overrides)


def load_variant(name: str) -> dict:
    """Override bundle {"robot": .., "scanner": .., "camera": ..} of a robot variant (configs/robot_variants.yaml)."""
    variants = _load(os.path.join(CFG_DIR, "robot_variants.yaml"))
    if name not in variants:
        raise ValueError(f"unknown robot variant {name!r}; known: {sorted(variants)}")
    return variants[name] or {}


def load_all(scene: str, overrides: dict | None = None) -> dict:
    """overrides: {"scene": {...}, "barcode": {...}, "robot": {...}, "camera": {...}, "scanner": {...}}.
    The robot variant (robot.yaml `variant`, or overrides["robot"]["variant"]) is merged between the yaml files and the
    overrides."""
    o = overrides or {}
    variant = (o.get("robot") or {}).get("variant") or _load(os.path.join(CFG_DIR, "robot.yaml"))["variant"]
    v = load_variant(variant)
    robot = load_robot_cfg(deep_merge(v.get("robot") or {}, o.get("robot")))
    robot["variant"] = variant
    return {"scene": load_scene_cfg(scene, o.get("scene")), "barcode": load_barcode_cfg(o.get("barcode")),
            "robot": robot, "camera": load_camera_cfg(deep_merge(v.get("camera") or {}, o.get("camera"))),
            "scanner": load_scanner_cfg(deep_merge(v.get("scanner") or {}, o.get("scanner")))}
