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


def load_factors() -> dict:
    import yaml
    with open(FACTORS_YAML) as f:
        return yaml.safe_load(f)


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
