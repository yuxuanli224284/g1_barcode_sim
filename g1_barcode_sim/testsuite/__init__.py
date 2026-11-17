"""Test suites: scenario generation, loading, and the scenario -> layout conversion.

    from g1_barcode_sim.testsuite import load_suite, load_scenario
    for sc in load_suite("t1_L0"): ...
    sc = load_scenario("t2_L1_barcode_pos_0003")
"""
from __future__ import annotations

import json
import os

from ..config import PKG_ROOT

SUITES_DIR = os.path.join(PKG_ROOT, "suites")
FACTORS_YAML = os.path.join(PKG_ROOT, "configs", "testsuite", "factors.yaml")


# Robot-specific factor files, deep-merged over factors.yaml for that robot (config.load_robot_cfg selects the robot the
# same way). G1 has none: load_factors() / load_factors("g1") return factors.yaml unchanged.
ROBOT_FACTOR_FILES = {"t800": os.path.join(PKG_ROOT, "configs", "testsuite", "factors_t800.yaml")}


def load_factors(robot: str = "g1") -> dict:
    """factors.yaml, with configs/testsuite/factors_<robot>.yaml merged over it for a non-G1 robot (head-camera nominal
    file, observation-distance range). The merged dict records the robot under "robot" (G1: no key, as before)."""
    import yaml
    with open(FACTORS_YAML) as f:
        F = yaml.safe_load(f)
    if robot == "g1":
        return F
    if robot not in ROBOT_FACTOR_FILES:
        raise ValueError(f"no test-suite factor file for robot {robot!r}; known: g1, {sorted(ROBOT_FACTOR_FILES)}")
    from ..config import deep_merge
    with open(ROBOT_FACTOR_FILES[robot]) as f:
        over = yaml.safe_load(f)
    if over.get("robot") != robot:
        raise ValueError(f"{ROBOT_FACTOR_FILES[robot]} must declare robot: {robot} (got {over.get('robot')!r})")
    unknown = [k for k in over if k != "robot" and k not in F]
    if unknown:
        raise ValueError(f"{ROBOT_FACTOR_FILES[robot]}: keys {unknown} are not factors.yaml keys")
    return deep_merge(F, over)


def list_suites() -> list[dict]:
    out = []
    if not os.path.isdir(SUITES_DIR):
        return out
    for name in sorted(os.listdir(SUITES_DIR)):
        mp = os.path.join(SUITES_DIR, name, "manifest.json")
        if os.path.exists(mp):
            m = json.load(open(mp))
            lg = os.path.join(SUITES_DIR, name, "generation_log.json")
            m["generated_at"] = json.load(open(lg)).get("generated_at") if os.path.exists(lg) else None
            out.append(m)
    return out


def load_suite(name: str) -> list[dict]:
    """Scenario dicts of a generated suite, in id order."""
    d = os.path.join(SUITES_DIR, name, "scenarios")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"suite {name!r} not generated (python scripts/testsuite/generate.py --suite ...)")
    return [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.endswith(".json")]


def load_scenario(scenario_id: str) -> dict:
    """'t2_L1_barcode_pos_0003' -> scenario dict (suite = id without the trailing _NNNN)."""
    suite = scenario_id.rsplit("_", 1)[0]
    p = os.path.join(SUITES_DIR, suite, "scenarios", f"{scenario_id}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return json.load(open(p))
