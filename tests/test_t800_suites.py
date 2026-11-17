"""CPU checks of the robot-specific test-suite factors (no Isaac Sim): G1 factors / suites unchanged, the T800 factors
(own head-camera nominal file, observation range from backing up with the head at 0), T800 suite generation.
run: python -m pytest tests/test_t800_suites.py"""
import json
import math
import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "testsuite"))
from g1_barcode_sim.testsuite import FACTORS_YAML, SUITES_DIR, load_factors, load_scenario  # noqa: E402
from g1_barcode_sim.testsuite import geometry as G  # noqa: E402
from g1_barcode_sim.testsuite.sampler import expand_suite_names, generate  # noqa: E402
from g1_barcode_sim.testsuite.scenario import to_layout  # noqa: E402


def _cam(F):
    return G.head_camera(yaml.safe_load(open(os.path.join(ROOT, F["head_camera_nominal"]))))


def test_g1_factors_are_the_file():
    assert load_factors() == load_factors("g1") == yaml.safe_load(open(FACTORS_YAML))
    assert "robot" not in load_factors()


def test_t800_factors():
    F, G1 = load_factors("t800"), load_factors()
    assert F["robot"] == "t800" and F["head_camera_nominal"] == "configs/testsuite/head_camera_nominal_t800.yaml"
    assert F["task2_start"]["d_obs_range"] == [0.60, 1.50]
    rest = {k: v for k, v in F["task2_start"].items() if k != "d_obs_range"}
    assert rest == {k: v for k, v in G1["task2_start"].items() if k != "d_obs_range"}
    assert {k: v for k, v in F.items() if k not in ("robot", "head_camera_nominal", "task2_start")} == \
        {k: v for k, v in G1.items() if k not in ("head_camera_nominal", "task2_start")}
    with pytest.raises(ValueError):
        load_factors("h1")


def test_t800_obs_bound_derivation():
    """d_obs_range upper bound = first grid point at which a face's floor line is inside the image with the margin."""
    F = load_factors("t800")
    S = F["task2_start"]
    nom = yaml.safe_load(open(os.path.join(ROOT, F["head_camera_nominal"])))
    T, K = np.asarray(nom["T_root_cam"], float), np.asarray(nom["K"], float)
    a = math.asin(-T[2, 2]) + math.atan((nom["height"] * (1 - S["fov_margin_frac"]) - K[1, 2]) / K[1, 1])
    d = (nom["root_height"] + T[2, 3]) / math.tan(a) + T[0, 3]
    k = math.ceil((d - S["d_obs_range"][0]) / S["d_obs_step"] - 1e-9)
    assert S["d_obs_range"][1] == pytest.approx(S["d_obs_range"][0] + k * S["d_obs_step"])
    assert math.degrees(math.asin(-T[2, 2])) == pytest.approx(8.0, abs=0.05)          # head at 0: the O2S mount pitch


def test_wall_obs_pose_t800_backs_up_g1_does_not_fit():
    for robot, d_exp, fits in (("t800", 1.50, True), ("g1", 1.00, False)):
        F = load_factors(robot)
        lay = to_layout(load_scenario("wall6x5_L0_0000"), F)
        fc, n, c = G.front_face_union(lay["boxes"], F["barcode"]["face_offset"] + 0.001)
        S = F["task2_start"]
        ob = G.obs_pose(fc, n, c, _cam(F), S["d_obs_range"], S["d_obs_step"], S["fov_margin_frac"])
        assert ob["d_obs"] == pytest.approx(d_exp) and ob["fits"] is fits


def test_g1_suites_regenerate_identically():
    import generate as GEN
    F = load_factors()
    for name in ("t1_L0", "t2_L0"):
        for full, d in expand_suite_names(F, name, None):
            for sc in generate(F, _cam(F), full, d, 0, None)[0]:
                ref = open(os.path.join(SUITES_DIR, full, "scenarios", f"{sc['id']}.json")).read()
                assert GEN.dumps(sc) == ref, sc["id"]


def test_t800_suite_same_scenes_own_start_pose():
    F, G1 = load_factors("t800"), load_factors()
    (full, d), = expand_suite_names(F, "t2_L0", None)
    t8 = generate(F, _cam(F), full + "_t800", d, 0, None, rng_name=full)[0]
    g1 = generate(G1, _cam(G1), full, d, 0, None)[0]
    assert len(t8) == len(g1)
    for a, b in zip(t8, g1):
        assert a["id"] == b["id"].replace("t2_L0_", "t2_L0_t800_") and a["robot"] == "t800" and "robot" not in b
        assert a["boxes"] == b["boxes"] and a["barcodes"] == b["barcodes"] and a["robot_init"] == b["robot_init"]
        assert a["task2_start"]["obs_fits"]
        assert a["task2_start"]["obs_d"] > b["task2_start"]["obs_d"]       # level head camera: farther than the G1


def test_scanner_frames_without_isaac():
    """The pure scanner transforms import without Isaac Lab (offline URDF export) and are what scanner.py exports."""
    from g1_barcode_sim.config import load_robot_cfg, load_scanner_cfg
    from g1_barcode_sim.scanner_frames import transforms
    st = transforms(load_robot_cfg({"model": "t800"})["scanner_mount"], load_scanner_cfg())
    np.testing.assert_allclose(st["T_hand_tip"][:3, 3], [0.080, 0.038, -0.141], atol=1e-9)
    assert json.dumps(np.round(st["T_hand_G"], 6).tolist())
