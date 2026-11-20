"""scripts/testsuite/make_scanroom_suite.py: the default suite is reproduced byte for byte, the 4-box subset suite has
the same boxes / labels with its own start pose, GT target boxes and the same targets in robot terms."""
import importlib.util
import json
import math
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("make_scanroom_suite", os.path.join(ROOT, "scripts", "testsuite", "make_scanroom_suite.py"))
MS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MS)

SUB4 = ["--suite", "scanroom6x5_sub4_L0", "--robot_dist", "0.60", "--lateral", "0.40",
        "--targets", "C1R4,C2R4,C1R3,C2R3", "--order", "rows"]


def _stored(suite):
    d = os.path.join(ROOT, "suites", suite)
    return (open(os.path.join(d, "scenarios", f"{suite}_0000.json")).read(), open(os.path.join(d, "manifest.json")).read())


@pytest.mark.parametrize("argv,suite", [([], "scanroom6x5_L0"), (SUB4, "scanroom6x5_sub4_L0")])
def test_generator_reproduces_the_stored_suites(argv, suite):
    sc, man = MS.build(MS.parse_args(argv))
    want_sc, want_man = _stored(suite)
    assert json.dumps(sc, indent=1) == want_sc and json.dumps(man, indent=1) == want_man


def test_subset_suite_targets_and_start():
    from g1_barcode_sim.testsuite import load_factors
    from g1_barcode_sim.testsuite.scenario import to_layout
    full, _ = MS.build(MS.parse_args([]))
    sub, _ = MS.build(MS.parse_args(SUB4))
    assert sub["boxes"] == full["boxes"] and sub["barcodes"] == full["barcodes"] and sub["target"] == "all"
    assert sub["target_boxes"] == ["C1R4", "C2R4", "C1R3", "C2R3"]
    assert sub["scan_targets"] == {"columns_from_left": [1, 2], "rows_from_top": [2, 3], "order": "rows"}
    assert "target_boxes" not in full and "scan_targets" not in full
    lay = to_layout(sub, load_factors())
    assert lay["robot_init"]["pos"] == pytest.approx([0.70, 0.40]) and lay["robot_init"]["yaw"] == 0.0
    assert lay["target_box_id"] == "all"
    by_box = {b["box_id"]: b for b in lay["barcodes"]}
    # C1 is the robot's left (+y), R5 the top row: the targets are the second and third rows from the top
    assert by_box["C1R4"]["center"][1] > by_box["C2R4"]["center"][1] and by_box["C1R4"]["center"][2] > by_box["C1R3"]["center"][2]
    assert by_box["C1R5"]["center"][2] > by_box["C1R4"]["center"][2]
    assert math.isclose(lay["robot_init"]["pos"][0], 1.30 - 0.60)


def test_subset_spec_needs_a_block():
    assert MS.subset_spec(["C3R1"], "columns") == {"columns_from_left": [3], "rows_from_top": [5], "order": "columns"}
    with pytest.raises(ValueError):
        MS.subset_spec(["C1R4", "C2R3"], "rows")          # a diagonal pair is no columns x rows block
    with pytest.raises(ValueError):
        MS.subset_spec(["C7R1"], "rows")
