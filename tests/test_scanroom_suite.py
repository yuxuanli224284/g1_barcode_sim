"""scripts/testsuite/make_scanroom_suite.py: the default suite is reproduced byte for byte, the subset suites (4-box,
rows 2-4 from the bottom x all columns, the same with the wall layout and the serpentine order) have the same boxes /
labels with their own start pose, GT target boxes and the same targets in robot terms."""
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
R234_BOXES = [f"C{c}R{r}" for r in (4, 3, 2) for c in range(1, 7)]
R234 = ["--suite", "scanroom6x5_r234_L0", "--robot_dist", "0.60", "--lateral", "0.0", "--targets", ",".join(R234_BOXES),
        "--order", "rows", "--rows_from", "bottom", "--all_columns", "--head_views", "upright", "--phase1_view", "obs_pose"]
LAYOUT = ["--suite", "scanroom6x5_r234_layout_L0", "--robot_dist", "0.60", "--lateral", "0.0",
          "--targets", ",".join(R234_BOXES), "--order", "serpentine", "--rows_from", "bottom", "--all_columns",
          "--head_views", "upright", "--phase1_view", "obs_pose", "--wall_layout"]


def _stored(suite):
    d = os.path.join(ROOT, "suites", suite)
    return (open(os.path.join(d, "scenarios", f"{suite}_0000.json")).read(), open(os.path.join(d, "manifest.json")).read())


@pytest.mark.parametrize("argv,suite", [([], "scanroom6x5_L0"), (SUB4, "scanroom6x5_sub4_L0"), (R234, "scanroom6x5_r234_L0"),
                                        (LAYOUT, "scanroom6x5_r234_layout_L0")])
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


def test_subset_spec_rows_from_bottom_and_all_columns():
    assert MS.subset_spec(["C1R4", "C2R4", "C1R3", "C2R3"], "rows", rows_from="bottom") == {
        "columns_from_left": [1, 2], "rows_from_bottom": [3, 4], "order": "rows"}
    assert MS.subset_spec(R234_BOXES, "rows", rows_from="bottom", columns="all") == {
        "columns": "all", "rows_from_bottom": [2, 3, 4], "order": "rows"}
    assert list(MS.subset_spec(R234_BOXES, "rows", rows_from="bottom", columns="all")) == ["columns", "rows_from_bottom", "order"]
    with pytest.raises(ValueError):
        MS.subset_spec(["C1R4", "C2R4", "C1R3", "C2R3"], "rows", columns="all")      # a partial column set is not "all"
    with pytest.raises(ValueError):
        MS.subset_spec(["C1R4"], "rows", rows_from="middle")
    with pytest.raises(ValueError):
        MS.subset_spec(["C1R4"], "rows", columns="every")
    with pytest.raises(ValueError):                                                 # the full-block check still holds
        MS.subset_spec([b for b in R234_BOXES if b != "C6R3"], "rows", rows_from="bottom", columns="all")


def test_r234_suite_fields_and_options_only_when_given():
    full, _ = MS.build(MS.parse_args([]))
    sub, _ = MS.build(MS.parse_args(SUB4))
    r234, _ = MS.build(MS.parse_args(R234))
    assert r234["boxes"] == full["boxes"] and r234["barcodes"] == full["barcodes"] and r234["target"] == "all"
    assert r234["target_boxes"] == R234_BOXES
    assert r234["scan_targets"] == {"columns": "all", "rows_from_bottom": [2, 3, 4], "order": "rows"}
    assert r234["head_views"] == "upright" and r234["phase1_view"] == "obs_pose"
    assert r234["robot_init"] == {"dist_to_table": 0.60, "lateral_offset": 0.0, "yaw_deg": 0.0}
    note = r234["factor"]["note"]
    assert "(all columns, rows [2, 3, 4] from the bottom of the wall), order rows;" in note
    assert note.endswith("the robot may walk back to an upright observation pose (no lean-back head views).")
    for sc in (full, sub):                                # the new task fields only when given
        assert "head_views" not in sc and "phase1_view" not in sc
        assert not {"rows_from", "columns", "head_views", "phase1_view"} & set(sc["factor"]["params"])
    assert sub["factor"]["note"].endswith(" Task: scan only the labels of boxes C1R4, C2R4, C1R3, C2R3 (columns [1, 2] "
                                          "from the left, rows [2, 3] from the top of the wall), order rows; the robot "
                                          "works from near its start pose.")
    one, _ = MS.build(MS.parse_args(SUB4 + ["--phase1_view", "start_pose"]))
    assert one["phase1_view"] == "start_pose" and "head_views" not in one
    for extra in (["--all_columns"], ["--rows_from", "bottom"], ["--head_views", "upright"], ["--phase1_view", "obs_pose"],
                  ["--order", "rows"]):
        with pytest.raises(SystemExit):                   # a subset-task option without --targets
            MS.parse_args(extra)
    with pytest.raises(SystemExit):
        MS.parse_args(SUB4 + ["--head_views", "leaned"])



def test_layout_suite_fields_and_options_only_when_given():
    """scanroom6x5_r234_layout_L0 = r234 with order serpentine and the wall layout (from the scene constants); the layout
    field and param appear only when --wall_layout is given; --wall_layout needs a subset task planned from the
    observation pose."""
    from g1_barcode_sim.testsuite import load_factors
    from g1_barcode_sim.testsuite.scenario import to_layout
    full, _ = MS.build(MS.parse_args([]))
    sub, _ = MS.build(MS.parse_args(SUB4))
    r234, _ = MS.build(MS.parse_args(R234))
    lay, man = MS.build(MS.parse_args(LAYOUT))
    assert lay["wall_layout"] == {"columns": 6, "rows": 5, "box_w": 0.20, "box_h": 0.25, "base": "floor"}
    assert (lay["wall_layout"]["columns"], lay["wall_layout"]["rows"]) == (MS.COLS, MS.ROWS)
    assert (lay["wall_layout"]["box_w"], lay["wall_layout"]["box_h"]) == (MS.BOX_W, MS.BOX_H)
    assert list(lay["wall_layout"]) == ["columns", "rows", "box_w", "box_h", "base"]
    assert lay["scan_targets"] == {"columns": "all", "rows_from_bottom": [2, 3, 4], "order": "serpentine"}
    assert lay["factor"]["params"]["wall_layout"] is True and lay["factor"]["params"]["order"] == "serpentine"
    assert man["scenario_ids"] == ["scanroom6x5_r234_layout_L0_0000"]
    # the same scene, targets and start as r234: only the task fields differ
    for k in ("boxes", "barcodes", "target", "target_boxes", "robot_init", "head_views", "phase1_view", "scene_usd",
              "label_margin", "background", "table", "task2_start"):
        assert lay[k] == r234[k], k
    assert set(lay) - set(r234) == {"wall_layout"}
    note = lay["factor"]["note"]
    assert ("order serpentine (rows from the top, left to right, then right to left, alternating); the robot may walk "
            "back to an upright observation pose (no lean-back head views); the wall layout (6 x 5 boxes of 0.2 x 0.25 m "
            "on the floor) is given to the agent.") in note
    assert r234["factor"]["note"].startswith(note[:note.index(" order serpentine")] + " order rows;")
    lay_layout = to_layout(lay, load_factors())                    # the simulator accepts (ignores) the task field
    assert lay_layout["robot_init"]["pos"] == pytest.approx([0.70, 0.0])
    for sc in (full, sub, r234):                                    # absent unless given
        assert "wall_layout" not in sc and "wall_layout" not in sc["factor"]["params"]
    assert "wall layout" not in r234["factor"]["note"] and "serpentine" not in r234["factor"]["note"]
    for argv in (["--wall_layout"],                                 # no subset task
                 R234[:-2] + ["--wall_layout"],                     # no observation pose
                 R234[:-1] + ["start_pose", "--wall_layout"]):
        with pytest.raises(SystemExit):
            MS.parse_args(argv)
    with pytest.raises(SystemExit):                                 # serpentine is a subset-task order too
        MS.parse_args(["--order", "serpentine"])


def test_serpentine_order_spec():
    assert MS.subset_spec(["C1R4", "C2R4", "C1R3", "C2R3"], "serpentine") == {
        "columns_from_left": [1, 2], "rows_from_top": [2, 3], "order": "serpentine"}
    with pytest.raises(ValueError):
        MS.subset_spec(["C1R4"], "zigzag")
